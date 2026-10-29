// arch/parakeet/decoder.cpp - Parakeet TDT decoder implementation.
//
// See decoder.h for the API contract. The predictor LSTM, encoder
// projection, and joint network all run as ggml backend graphs on a
// single shared CPU backend + persistent threadpool (owned by PredGraph).
// Decode weights are uploaded to resident ggml tensors at load and the
// host fp32 mirrors then freed.
//
// Numerical note: ggml's reduction order differs slightly from a naive
// host loop, so per-step values drift ~1e-7 vs the reference; over an
// utterance this stays within ~1e-4 and the greedy transcript is
// unchanged.

#include "decoder.h"

#include "parakeet.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"
#include "vocabulary.h"
#include "weights.h"

// ggml-backend.h, not ggml-cpu.h: under GGML_BACKEND_DL the CPU backend
// is a loadable module, so backend-specific entry points must be reached
// through the registry (ggml_backend_init_by_type / get_proc_address),
// never by direct link-time reference.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"  // ggml_threadpool_params_default (rest via registry)
#include "ggml.h"

#include <thread>

#if TRANSCRIBE_HAS_BLAS
#    ifdef __APPLE__
#        include <Accelerate/Accelerate.h>
#    else
#        include <cblas.h>
#    endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace transcribe::parakeet {

// ---------------------------------------------------------------------------
// Host weight extraction
// ---------------------------------------------------------------------------

namespace {

// Read all elements of a ggml_tensor into a host fp32 vector,
// dequantizing via ggml_get_type_traits()->to_float. Source dtype
// support is whatever ggml registers a to_float for (F32, F16, BF16, all
// Q*/IQ*). ggml_backend_tensor_get is universal (memcpy on host buffers,
// readback on discrete GPUs); we stage the raw bytes then materialize
// fp32 in `out`. Returns false on missing-trait / size errors.
bool read_tensor_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: read_tensor_to_f32: null tensor");
        return false;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" has 0 bytes", t->name);
        return false;
    }
    const int64_t nelem = ggml_nelements(t);
    if (nelem <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" has nelem=%lld", t->name,
                static_cast<long long>(nelem));
        return false;
    }

    out.resize(static_cast<size_t>(nelem));

    // F32 fast path. type_traits[GGML_TYPE_F32] has no to_float (identity
    // is left null upstream), so short-circuit rather than dispatch.
    if (t->type == GGML_TYPE_F32) {
        if (nbytes != static_cast<size_t>(nelem) * sizeof(float)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: tensor \"%s\" f32 nbytes %zu "
                    "!= nelem*sizeof(float) %zu",
                    t->name, nbytes, static_cast<size_t>(nelem) * sizeof(float));
            return false;
        }
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
        return true;
    }

    // Quantized / non-fp32 path: stage the raw bytes off the backend,
    // then walk the type's to_float to materialize fp32.
    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" type %s has no to_float", t->name,
                ggml_type_name(t->type));
        return false;
    }

    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    tt->to_float(raw.data(), out.data(), nelem);
    return true;
}

// Make the joint network's weights resident as fp32 ggml tensors on the
// model: enc / pred / out projections. Built once at load. out_w is
// dequantized from the model tensor src_out_w; the rest come from the
// host mirrors, freed here once uploaded. On failure frees partial state
// and returns false (w_ready stays false → hard decode error).
bool build_joint_weight(HostJoint & j, const ggml_tensor * src_out_w) {
    const int joint_h = j.joint_h;
    const int joint_n = j.joint_n;

    auto fail = [&]() -> bool {
        if (j.w_buf != nullptr) {
            safe_buffer_free(j.w_buf);
            j.w_buf = nullptr;
        }
        if (j.w_ctx != nullptr) {
            ggml_free(j.w_ctx);
            j.w_ctx = nullptr;
        }
        if (j.w_backend != nullptr) {
            safe_backend_free(j.w_backend);
            j.w_backend = nullptr;
        }
        j.g_enc_w  = nullptr;
        j.g_enc_b  = nullptr;
        j.g_pred_w = nullptr;
        j.g_pred_b = nullptr;
        j.gw_w     = nullptr;
        j.gw_b     = nullptr;
        j.w_ready  = false;
        return false;
    };

    // CPU backend used ONLY to allocate the resident weight buffer —
    // never graph_compute'd, so it carries no per-step state.
    j.w_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (j.w_backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml CPU backend init failed");
        return fail();
    }

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 8;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    j.w_ctx       = ggml_init(ip);
    if (j.w_ctx == nullptr) {
        return fail();
    }

    j.g_enc_w  = ggml_new_tensor_2d(j.w_ctx, GGML_TYPE_F32, j.d_enc, joint_h);
    j.g_enc_b  = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_h);
    j.g_pred_w = ggml_new_tensor_2d(j.w_ctx, GGML_TYPE_F32, j.pred_hidden, joint_h);
    j.g_pred_b = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_h);
    j.gw_w     = ggml_new_tensor_2d(j.w_ctx, GGML_TYPE_F32, joint_h, joint_n);
    j.gw_b     = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_n);

    j.w_buf = ggml_backend_alloc_ctx_tensors(j.w_ctx, j.w_backend);
    if (j.w_buf == nullptr) {
        return fail();
    }

    // out_w: dequantize the model tensor to fp32 once.
    {
        std::vector<float> tmp;
        if (!read_tensor_to_f32(src_out_w, tmp)) {
            return fail();
        }
        ggml_backend_tensor_set(j.gw_w, tmp.data(), 0, tmp.size() * sizeof(float));
    }
    // The rest come from the host mirrors.
    ggml_backend_tensor_set(j.g_enc_w, j.enc_w.data(), 0, j.enc_w.size() * sizeof(float));
    ggml_backend_tensor_set(j.g_enc_b, j.enc_b.data(), 0, j.enc_b.size() * sizeof(float));
    ggml_backend_tensor_set(j.g_pred_w, j.pred_w.data(), 0, j.pred_w.size() * sizeof(float));
    ggml_backend_tensor_set(j.g_pred_b, j.pred_b.data(), 0, j.pred_b.size() * sizeof(float));
    ggml_backend_tensor_set(j.gw_b, j.out_b.data(), 0, j.out_b.size() * sizeof(float));

    // Host mirrors are now resident in ggml — release them (load-time scratch).
    std::vector<float>().swap(j.enc_w);
    std::vector<float>().swap(j.enc_b);
    std::vector<float>().swap(j.pred_w);
    std::vector<float>().swap(j.pred_b);
    std::vector<float>().swap(j.out_b);

    j.w_ready = true;
    return true;
}

// Make the predictor LSTM weights resident as fp32 ggml tensors for the
// per-call PredGraph. Built once at load. ne is [pred_hidden, 4*pred_hidden]
// for Wx/Wh (row-major [4*H, H] host bytes as a mul_mat operand) and
// [4*pred_hidden] for the bias. On failure frees partial state and returns
// false (lstm_ready stays false → hard decode error).
bool build_pred_weights(HostPredictor & p) {
    const int H      = p.pred_hidden;
    const int four_H = 4 * H;
    const int L      = static_cast<int>(p.lstm.size());
    if (L == 0 || H == 0) {
        return false;
    }

    auto fail = [&]() -> bool {
        if (p.lstm_w_buf != nullptr) {
            safe_buffer_free(p.lstm_w_buf);
            p.lstm_w_buf = nullptr;
        }
        if (p.lstm_w_ctx != nullptr) {
            ggml_free(p.lstm_w_ctx);
            p.lstm_w_ctx = nullptr;
        }
        if (p.lstm_w_backend != nullptr) {
            safe_backend_free(p.lstm_w_backend);
            p.lstm_w_backend = nullptr;
        }
        for (auto & lh : p.lstm) {
            lh.g_Wx = nullptr;
            lh.g_Wh = nullptr;
            lh.g_b  = nullptr;
        }
        p.lstm_ready = false;
        return false;
    };

    // Alloc-only backend (never graph_compute'd), mirroring HostJoint::w_backend.
    p.lstm_w_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (p.lstm_w_backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml CPU backend init failed (pred)");
        return fail();
    }

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(L * 3 + 1);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    p.lstm_w_ctx  = ggml_init(ip);
    if (p.lstm_w_ctx == nullptr) {
        return fail();
    }

    for (int l = 0; l < L; ++l) {
        auto & lh = p.lstm[l];
        lh.g_Wx   = ggml_new_tensor_2d(p.lstm_w_ctx, GGML_TYPE_F32, H, four_H);
        lh.g_Wh   = ggml_new_tensor_2d(p.lstm_w_ctx, GGML_TYPE_F32, H, four_H);
        lh.g_b    = ggml_new_tensor_1d(p.lstm_w_ctx, GGML_TYPE_F32, four_H);
    }

    p.lstm_w_buf = ggml_backend_alloc_ctx_tensors(p.lstm_w_ctx, p.lstm_w_backend);
    if (p.lstm_w_buf == nullptr) {
        return fail();
    }

    for (int l = 0; l < L; ++l) {
        auto & lh = p.lstm[l];
        ggml_backend_tensor_set(lh.g_Wx, lh.Wx.data(), 0, lh.Wx.size() * sizeof(float));
        ggml_backend_tensor_set(lh.g_Wh, lh.Wh.data(), 0, lh.Wh.size() * sizeof(float));
        ggml_backend_tensor_set(lh.g_b, lh.b.data(), 0, lh.b.size() * sizeof(float));
        // Host mirrors now resident in ggml — release them.
        std::vector<float>().swap(lh.Wx);
        std::vector<float>().swap(lh.Wh);
        std::vector<float>().swap(lh.b);
    }

    p.lstm_ready = true;
    return true;
}

}  // namespace

// Per-call joint compute graph (the mutable half of the joint network):
//   pred_proj = pred_w @ pred_in + pred_b      [joint_h]
//   summed    = enc_proj + pred_proj           [joint_h]
//   activated = activation(summed)             [joint_h]
//   logits    = out_w @ activated + out_b      [joint_n]
// Built on a BORROWED backend (PredGraph's shared pool) so the joint runs
// on the same threadpool as the pred LSTM and enc_proj. Inputs are the
// predictor output and this frame's precomputed enc projection; output is
// the logits. Built fresh per decode call (reentrant); weights are the
// shared resident HostJoint tensors.
namespace {

struct JointGraph {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;  // BORROWED (PredGraph's); NOT freed here
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         pred_in = nullptr;  // [pred_hidden] fp32 input (decoder out)
    ggml_tensor *         enc_in  = nullptr;  // [joint_h] fp32 input (enc_proj frame)
    ggml_tensor *         logits  = nullptr;  // [joint_n] fp32 output
    bool                  ready   = false;

    JointGraph() = default;

    ~JointGraph() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        // backend is borrowed from PredGraph — do NOT free here.
    }

    JointGraph(const JointGraph &)             = delete;
    JointGraph & operator=(const JointGraph &) = delete;
};

// Build the full joint graph on the shared `backend` (PredGraph's pool).
// Returns false if the resident weights are absent or any ggml step
// fails (→ hard decode error).
bool build_joint_graph(JointGraph & g, const HostJoint & j, ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    if (!j.w_ready || j.gw_w == nullptr || j.gw_b == nullptr || j.g_pred_w == nullptr || j.g_pred_b == nullptr) {
        return false;
    }

    g.backend = backend;  // borrowed

    auto fail = [&]() -> bool {
        if (g.buf != nullptr) {
            safe_buffer_free(g.buf);
            g.buf = nullptr;
        }
        if (g.ctx != nullptr) {
            ggml_free(g.ctx);
            g.ctx = nullptr;
        }
        g.pred_in = nullptr;
        g.enc_in  = nullptr;
        g.logits  = nullptr;
        g.graph   = nullptr;
        g.ready   = false;
        return false;
    };

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return fail();
    }

    g.pred_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.pred_hidden);
    ggml_set_input(g.pred_in);
    g.enc_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.joint_h);
    ggml_set_input(g.enc_in);

    // pred_proj = pred_w @ pred_in + pred_b   [joint_h]
    ggml_tensor * pred_proj = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.g_pred_w, g.pred_in), j.g_pred_b);
    // summed = enc_proj + pred_proj           [joint_h]
    ggml_tensor * summed    = ggml_add(g.ctx, g.enc_in, pred_proj);
    // activation — loader allow-list guarantees exactly one of relu/sigmoid/tanh.
    ggml_tensor * activated;
    if (j.activation == "relu") {
        activated = ggml_relu(g.ctx, summed);
    } else if (j.activation == "sigmoid") {
        activated = ggml_sigmoid(g.ctx, summed);
    } else {  // "tanh"
        activated = ggml_tanh(g.ctx, summed);
    }
    // logits = out_w @ activated + out_b      [joint_n]
    ggml_tensor * mm = ggml_mul_mat(g.ctx, j.gw_w, activated);
    g.logits         = ggml_add(g.ctx, mm, j.gw_b);
    ggml_set_output(g.logits);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) {
        return fail();
    }

    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.logits);
    g.ready = true;
    return true;
}

// CPU-backend threadpool entry points, reached through the registry so
// the library stays DL-safe (under GGML_BACKEND_DL these symbols are not
// directly linkable). Resolved via ggml_backend_reg_get_proc_address.
typedef ggml_threadpool_t (*pfn_threadpool_new)(ggml_threadpool_params *);
typedef void (*pfn_threadpool_free)(ggml_threadpool_t);
typedef void (*pfn_set_threadpool)(ggml_backend_t, ggml_threadpool_t);

static void * cpu_backend_proc(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

// Free a CPU-backend threadpool via the registry. Resolving the free fn needs
// the backend alive, so callers must invoke this BEFORE freeing the backend.
static void free_cpu_threadpool(ggml_backend_t backend, ggml_threadpool_t tp) {
    if (tp == nullptr || backend == nullptr) {
        return;
    }
    if (auto fn = (pfn_threadpool_free) cpu_backend_proc(backend, "ggml_threadpool_free")) {
        fn(tp);
    }
}

// Per-call predictor LSTM graph (the mutable half of the predictor):
// a single per-step graph built fresh per decode call around the
// model-resident HostPredictor::lstm weights, recomputed in place each
// step. Inputs are the embedding x and per-layer previous (h, c); outputs
// are the new (h, c) per layer. Reentrant: every concurrent decode owns
// its own PredGraph (backend + threadpool + I/O tensors).
struct PredGraph {
    ggml_context *             ctx     = nullptr;
    ggml_backend_t             backend = nullptr;
    ggml_backend_buffer_t      buf     = nullptr;
    ggml_cgraph *              graph   = nullptr;
    ggml_threadpool_t          tp      = nullptr;
    ggml_tensor *              x       = nullptr;  // [H] input embedding
    std::vector<ggml_tensor *> ph;                 // [H] prev hidden per layer (input)
    std::vector<ggml_tensor *> pc;                 // [H] prev cell   per layer (input)
    std::vector<ggml_tensor *> nh;                 // [H] new  hidden per layer (output)
    std::vector<ggml_tensor *> nc;                 // [H] new  cell   per layer (output)
    int                        H     = 0;
    int                        L     = 0;
    bool                       ready = false;

    PredGraph() = default;

    ~PredGraph() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        free_cpu_threadpool(backend, tp);  // before safe_backend_free(backend)
        if (backend != nullptr) {
            safe_backend_free(backend);
        }
    }

    PredGraph(const PredGraph &)             = delete;
    PredGraph & operator=(const PredGraph &) = delete;
};

// Build the per-call LSTM graph around the resident p.lstm[*].g_Wx/g_Wh/g_b.
// Returns false if the resident weights are absent or any ggml step fails
// (→ hard decode error). n_threads is the resolved (>0) thread count.
bool build_pred_graph(PredGraph & g, const HostPredictor & p, int n_threads) {
    if (!p.lstm_ready) {
        return false;
    }
    const int H = p.pred_hidden;
    const int L = static_cast<int>(p.lstm.size());
    if (H == 0 || L == 0) {
        return false;
    }

    auto fail = [&]() -> bool {
        if (g.buf != nullptr) {
            safe_buffer_free(g.buf);
            g.buf = nullptr;
        }
        if (g.ctx != nullptr) {
            ggml_free(g.ctx);
            g.ctx = nullptr;
        }
        free_cpu_threadpool(g.backend, g.tp);
        g.tp = nullptr;  // before freeing backend
        if (g.backend != nullptr) {
            safe_backend_free(g.backend);
            g.backend = nullptr;
        }
        g.x     = nullptr;
        g.graph = nullptr;
        g.ready = false;
        g.ph.clear();
        g.pc.clear();
        g.nh.clear();
        g.nc.clear();
        return false;
    };

    g.H = H;
    g.L = L;

    g.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (g.backend == nullptr) {
        return fail();
    }
    if (ggml_backend_dev_t dev = ggml_backend_get_device(g.backend)) {
        auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
        if (set_n_threads != nullptr) {
            set_n_threads(g.backend, std::max(1, n_threads));
        }
    }
    // Persistent threadpool so the per-step graph_compute reuses workers instead
    // of spawning a transient pool each call. ggml's default hybrid-polling
    // (poll=50) keeps the workers hot between the sub-millisecond per-step
    // dispatches; poll=0 (park-immediately) was measured to regress pred to
    // 70-110 ms because parked workers are slow to reschedule.
    {
        auto tp_new = (pfn_threadpool_new) cpu_backend_proc(g.backend, "ggml_threadpool_new");
        auto tp_set = (pfn_set_threadpool) cpu_backend_proc(g.backend, "ggml_backend_cpu_set_threadpool");
        if (tp_new != nullptr && tp_set != nullptr) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(std::max(1, n_threads));
            g.tp                       = tp_new(&tpp);
            if (g.tp != nullptr) {
                tp_set(g.backend, g.tp);
            }
        }
        // If unresolved (non-CPU exotic backend), g.tp stays null and each
        // graph_compute spawns a transient pool — correct, just less tuned.
    }

    ggml_init_params ip{};
    // ~17 op nodes/layer + (1 + 2L) input tensors; generous headroom.
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(32 * L + 16) + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return fail();
    }

    g.x = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
    ggml_set_input(g.x);
    g.ph.assign(static_cast<size_t>(L), nullptr);
    g.pc.assign(static_cast<size_t>(L), nullptr);
    g.nh.assign(static_cast<size_t>(L), nullptr);
    g.nc.assign(static_cast<size_t>(L), nullptr);
    for (int l = 0; l < L; ++l) {
        g.ph[l] = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
        ggml_set_input(g.ph[l]);
        g.pc[l] = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
        ggml_set_input(g.pc[l]);
    }

    // gates = Wx@x + Wh@h_prev + b; split [i,f,g,o]; c' = f*c + i*g; h' = o*tanh(c').
    // Gate order [i, f, g, o] (PyTorch standard).
    ggml_tensor * in = g.x;
    for (int l = 0; l < L; ++l) {
        const auto &  lh = p.lstm[static_cast<size_t>(l)];
        ggml_tensor * gates =
            ggml_add(g.ctx, ggml_add(g.ctx, ggml_mul_mat(g.ctx, lh.g_Wx, in), ggml_mul_mat(g.ctx, lh.g_Wh, g.ph[l])),
                     lh.g_b);  // [4H]
        auto part = [&](int k) {
            return ggml_view_1d(g.ctx, gates, H,
                                static_cast<size_t>(k) * static_cast<size_t>(H) * ggml_element_size(gates));
        };
        ggml_tensor * i_ = ggml_sigmoid(g.ctx, part(0));
        ggml_tensor * f_ = ggml_sigmoid(g.ctx, part(1));
        ggml_tensor * gg = ggml_tanh(g.ctx, part(2));
        ggml_tensor * o_ = ggml_sigmoid(g.ctx, part(3));
        g.nc[l]          = ggml_add(g.ctx, ggml_mul(g.ctx, f_, g.pc[l]), ggml_mul(g.ctx, i_, gg));
        ggml_set_output(g.nc[l]);
        g.nh[l] = ggml_mul(g.ctx, o_, ggml_tanh(g.ctx, g.nc[l]));
        ggml_set_output(g.nh[l]);
        in = g.nh[l];
    }

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) {
        return fail();
    }

    g.graph = ggml_new_graph(g.ctx);
    for (int l = 0; l < L; ++l) {
        ggml_build_forward_expand(g.graph, g.nh[l]);
        ggml_build_forward_expand(g.graph, g.nc[l]);
    }
    g.ready = true;
    return true;
}

}  // namespace

HostJoint::~HostJoint() {
    if (w_buf != nullptr) {
        safe_buffer_free(w_buf);
    }
    if (w_ctx != nullptr) {
        ggml_free(w_ctx);
    }
    if (w_backend != nullptr) {
        safe_backend_free(w_backend);
    }
}

HostPredictor::~HostPredictor() {
    if (lstm_w_buf != nullptr) {
        safe_buffer_free(lstm_w_buf);
    }
    if (lstm_w_ctx != nullptr) {
        ggml_free(lstm_w_ctx);
    }
    if (lstm_w_backend != nullptr) {
        safe_backend_free(lstm_w_backend);
    }
}

// Resolve a decode thread count: n_threads <= 0 means "auto" →
// default_n_threads() (min(8, usable cpus)), matching the encoder.
static int resolve_decode_threads(int n_threads) {
    return n_threads > 0 ? n_threads : transcribe::default_n_threads();
}

transcribe_status build_host_decoder_weights(const ParakeetModel & model, HostDecoderWeights & out) {
    const ParakeetHParams & hp = model.hparams;
    const ParakeetWeights & w  = model.weights;

    // Head-kind dispatch. CTC skips predictor/joint entirely; RNNT and
    // TDT share the predictor + joint mirror code below.
    switch (hp.head_kind) {
        case HeadKind::TDT:
            out.head_kind = HostHeadKind::TDT;
            break;
        case HeadKind::RNNT:
            out.head_kind = HostHeadKind::RNNT;
            break;
        case HeadKind::CTC:
            out.head_kind = HostHeadKind::CTC;
            break;
    }

    if (hp.head_kind == HeadKind::CTC) {
        // CTC head mirror. Source weight ne [1, d_model, n_classes]
        // (PyTorch [n_classes, d_model, 1]); bytes are already row-major
        // [n_classes, d_model] as the host decoder consumes them. Bias is
        // [n_classes].
        out.ctc_head.n_classes = hp.head_ctc_n_classes;
        out.ctc_head.blank_id  = hp.head_ctc_n_classes - 1;  // NeMo convention
        out.ctc_head.d_enc     = hp.enc_d_model;

        if (!read_tensor_to_f32(w.ctc_head.weight, out.ctc_head.weight)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(w.ctc_head.bias, out.ctc_head.bias)) {
            return TRANSCRIBE_ERR_GGUF;
        }

        const size_t expected_w = static_cast<size_t>(out.ctc_head.n_classes) * static_cast<size_t>(out.ctc_head.d_enc);
        if (out.ctc_head.weight.size() != expected_w) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: head.ctc.weight size %zu "
                    "!= n_classes*d_enc %zu",
                    out.ctc_head.weight.size(), expected_w);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (static_cast<int>(out.ctc_head.bias.size()) != out.ctc_head.n_classes) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: head.ctc.bias size %zu "
                    "!= n_classes %d",
                    out.ctc_head.bias.size(), out.ctc_head.n_classes);
            return TRANSCRIBE_ERR_GGUF;
        }

        out.tdt_durations.clear();
        out.tdt_max_symbols = 0;
        out.blank_id        = out.ctc_head.blank_id;
        out.n_vocab         = out.ctc_head.n_classes - 1;
        return TRANSCRIBE_OK;
    }

    // ----- Predictor mirror (TDT, RNNT) -----
    out.predictor.pred_hidden = hp.pred_hidden;
    out.predictor.pred_vocab  = hp.pred_vocab;

    if (!read_tensor_to_f32(w.predictor.embed_w, out.predictor.embed_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    // Sanity-check the flattened size against the catalog shape.
    {
        const size_t expected = static_cast<size_t>(hp.pred_vocab) * static_cast<size_t>(hp.pred_hidden);
        if (out.predictor.embed_w.size() != expected) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: pred.embed.weight size %zu "
                    "!= pred_vocab*pred_hidden %zu",
                    out.predictor.embed_w.size(), expected);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    out.predictor.lstm.assign(hp.pred_n_layers, HostLstmLayer{});
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        const auto & lw = w.predictor.lstm[i];
        auto &       lh = out.predictor.lstm[i];
        if (!read_tensor_to_f32(lw.Wx, lh.Wx)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(lw.Wh, lh.Wh)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(lw.b, lh.b)) {
            return TRANSCRIBE_ERR_GGUF;
        }

        const size_t gates    = static_cast<size_t>(4 * hp.pred_hidden);
        const size_t mat_size = gates * static_cast<size_t>(hp.pred_hidden);
        if (lh.Wx.size() != mat_size || lh.Wh.size() != mat_size) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: pred.lstm.%d Wx/Wh wrong size", i);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (lh.b.size() != gates) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: pred.lstm.%d bias wrong size", i);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Make the predictor LSTM weights resident. Fatal: a failure means
    // the model cannot decode — fail fast at load.
    if (!build_pred_weights(out.predictor)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: predictor ggml weight build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // ----- Joint mirror -----
    out.joint.d_enc       = hp.enc_d_model;
    out.joint.pred_hidden = hp.pred_hidden;
    out.joint.joint_h     = hp.joint_hidden;
    out.joint.joint_n     = hp.joint_n_classes();
    out.joint.activation  = hp.joint_activation;

    // enc/pred/out_b are mirrored to host fp32; out_w is NOT mirrored
    // here (build_joint_weight dequantizes it from the model tensor).
    if (!read_tensor_to_f32(w.joint.enc_w, out.joint.enc_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.enc_b, out.joint.enc_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.pred_w, out.joint.pred_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.pred_b, out.joint.pred_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.out_b, out.joint.out_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // Make the joint weights resident as fp32 ggml tensors. Fatal: a
    // failure means the model cannot decode — fail fast at load.
    if (w.joint.out_w == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint out_w missing");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int ne0 = static_cast<int>(w.joint.out_w->ne[0]);
    const int ne1 = static_cast<int>(w.joint.out_w->ne[1]);
    if (ne0 != out.joint.joint_h || ne1 != out.joint.joint_n) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: out_w ne [%d,%d] != [joint_h=%d, joint_n=%d]", ne0, ne1,
                out.joint.joint_h, out.joint.joint_n);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!build_joint_weight(out.joint, w.joint.out_w)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint ggml weight build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // ----- TDT params -----
    out.tdt_durations            = hp.tdt_durations;
    out.tdt_max_symbols          = hp.tdt_max_symbols;
    out.tdt_global_symbol_budget = hp.tdt_global_symbol_budget;
    out.n_vocab                  = hp.pred_vocab - 1;  // raw SP vocab size
    out.blank_id                 = hp.pred_vocab - 1;  // blank lives at vocab_size

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// LSTM state
// ---------------------------------------------------------------------------

void LstmState::reset(int n_layers, int pred_hidden) {
    h.resize(static_cast<size_t>(n_layers));
    c.resize(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        h[static_cast<size_t>(layer)].assign(static_cast<size_t>(pred_hidden), 0.0f);
        c[static_cast<size_t>(layer)].assign(static_cast<size_t>(pred_hidden), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

namespace {

// One predictor step on the ggml graph: reads prev_state, writes the new
// step's (h, c) into new_state, returns a borrowed pointer into
// new_state.h.back(). Feeds prev state + embedding into the resident
// per-call graph, computes, reads new state back into the host LstmState.
// Caller guarantees g.ready and that new_state is sized to (L, H). Returns
// nullptr if the graph compute fails.
const float * predictor_step_ggml(const HostPredictor & predictor,
                                  PredGraph &           g,
                                  int                   last_token,
                                  const LstmState &     prev_state,
                                  LstmState &           new_state,
                                  std::vector<float> &  scratch_x) {
    const int H = predictor.pred_hidden;
    if (static_cast<int>(scratch_x.size()) < H) {
        scratch_x.resize(static_cast<size_t>(H));
    }

    // Embed lookup (or start-state zeros).
    if (last_token < 0) {
        std::fill_n(scratch_x.data(), H, 0.0f);
    } else {
        const size_t row_off = static_cast<size_t>(last_token) * static_cast<size_t>(H);
        std::memcpy(scratch_x.data(), predictor.embed_w.data() + row_off, static_cast<size_t>(H) * sizeof(float));
    }

    const size_t hb = static_cast<size_t>(H) * sizeof(float);
    ggml_backend_tensor_set(g.x, scratch_x.data(), 0, hb);
    for (int l = 0; l < g.L; ++l) {
        ggml_backend_tensor_set(g.ph[l], prev_state.h[static_cast<size_t>(l)].data(), 0, hb);
        ggml_backend_tensor_set(g.pc[l], prev_state.c[static_cast<size_t>(l)].data(), 0, hb);
    }

    if (const ggml_status gs = ggml_backend_graph_compute(g.backend, g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: predictor compute failed (%d)", static_cast<int>(gs));
        return nullptr;
    }

    for (int l = 0; l < g.L; ++l) {
        ggml_backend_tensor_get(g.nh[l], new_state.h[static_cast<size_t>(l)].data(), 0, hb);
        ggml_backend_tensor_get(g.nc[l], new_state.c[static_cast<size_t>(l)].data(), 0, hb);
    }
    return new_state.h.back().data();
}

// Run the joint network for one decode step.
//
// `enc_proj` is the precomputed encoder projection for this frame
// (enc_w @ enc_frame + enc_b, joint_h floats); the caller batches all
// T_enc projections via one GEMM (precompute_enc_proj_ggml) before the
// loop. The graph computes:
//   pred_proj = pred_w @ pred_state + pred_b     [joint_h]
//   summed    = enc_proj + pred_proj             [joint_h]
//   activated = activation(summed)               [joint_h]
//   logits    = out_w @ activated   + out_b      [joint_n]
// Activation is one of {relu, sigmoid, tanh} (loader allow-list). Returns
// false if the graph compute fails.
bool joint_step(const HostJoint &    j,
                const JointGraph &   g,
                const float *        enc_proj,
                const float *        pred_state,
                std::vector<float> & out_logits) {
    if (static_cast<int>(out_logits.size()) < j.joint_n) {
        out_logits.resize(static_cast<size_t>(j.joint_n));
    }

    // Full joint on the shared decoder pool, one graph, one dispatch.
    ggml_backend_tensor_set(g.pred_in, pred_state, 0, static_cast<size_t>(j.pred_hidden) * sizeof(float));
    ggml_backend_tensor_set(g.enc_in, enc_proj, 0, static_cast<size_t>(j.joint_h) * sizeof(float));
    if (const ggml_status gs = ggml_backend_graph_compute(g.backend, g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint compute failed (%d)", static_cast<int>(gs));
        return false;
    }
    ggml_backend_tensor_get(g.logits, out_logits.data(), 0, static_cast<size_t>(j.joint_n) * sizeof(float));

    // log_softmax over the full joint output matches NeMo's CPU-inference
    // representation of joint_after_projection. It is a uniform per-row
    // shift, so it leaves token/duration argmax AND token_confidence
    // (which re-softmaxes the token sub-range) invariant — the decode
    // output is bit-identical with or without it. It is only needed to
    // make the dec.joint.0 dump element-wise comparable to NeMo's
    // normalized reference, so skip it unless dumping.
    if (transcribe::debug::enabled()) {
        float max_v = out_logits[0];
        for (int i = 1; i < j.joint_n; ++i) {
            if (out_logits[i] > max_v) {
                max_v = out_logits[i];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < j.joint_n; ++i) {
            sum += std::exp(static_cast<double>(out_logits[i] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int i = 0; i < j.joint_n; ++i) {
            out_logits[i] -= log_sum;
        }
    }
    return true;
}

// Compute the per-utterance encoder projection out[T, joint_h] =
// enc_out[T, d_enc] @ enc_w^T + enc_b as a single GEMM on `backend`
// (the shared decoder pool). Weight + bias are the model-resident
// j.g_enc_w / j.g_enc_b; only the input and result are allocated here.
// At n = T the matmul is a real GEMM (tinyBLAS engages with
// GGML_LLAMAFILE=ON). Returns false on ggml failure (→ hard decode error).
bool precompute_enc_proj_ggml(const HostJoint &    j,
                              ggml_backend_t       backend,
                              const float *        enc_out,
                              int                  T,
                              int                  d_enc,
                              std::vector<float> & out) {
    if (backend == nullptr || j.g_enc_w == nullptr || j.g_enc_b == nullptr) {
        return false;
    }
    const int joint_h = j.joint_h;
    out.resize(static_cast<size_t>(T) * static_cast<size_t>(joint_h));

    ggml_init_params ip{};
    ip.mem_size        = ggml_tensor_overhead() * 6 + ggml_graph_overhead();
    ip.mem_buffer      = nullptr;
    ip.no_alloc        = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_enc, T);
    ggml_set_input(in);
    ggml_tensor * mm  = ggml_mul_mat(ctx, j.g_enc_w, in);  // [joint_h, T]
    ggml_tensor * res = ggml_add(ctx, mm, j.g_enc_b);      // + [joint_h] (broadcast over T)
    ggml_set_output(res);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(in, enc_out, 0, static_cast<size_t>(T) * static_cast<size_t>(d_enc) * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, res);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        safe_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(res, out.data(), 0, static_cast<size_t>(T) * static_cast<size_t>(joint_h) * sizeof(float));

    safe_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Argmax over a contiguous fp32 range. Returns the index of the
// largest value; ties go to the first occurrence (matches the numpy
// argmax convention). The decoder uses this for both the token and
// duration argmaxes.
int argmax_range(const float * data, int n) {
    int   best_i = 0;
    float best_v = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return best_i;
}

// Entropy-based confidence over a token-logit slice. Mirrors the
// reference ParakeetTDT.decode_greedy path:
//
//     token_probs = softmax(token_logits)
//     entropy     = -sum(p * log(p + 1e-10))
//     max_entropy = log(vocab_size + 1)
//     confidence  = 1 - entropy / max_entropy
//
// In our terms `vocab_size + 1 == pred_vocab == n_token_classes`.
// The result lives in [0, 1] modulo the +1e-10 epsilon (which
// matches the reference verbatim, including its slight
// negative-bias for nearly-uniform distributions).
float token_confidence(const float * token_logits, int n_token_classes, std::vector<float> & scratch_probs) {
    // Numerically stable softmax: subtract max before exp.
    float max_logit = token_logits[0];
    for (int i = 1; i < n_token_classes; ++i) {
        if (token_logits[i] > max_logit) {
            max_logit = token_logits[i];
        }
    }
    if (static_cast<int>(scratch_probs.size()) < n_token_classes) {
        scratch_probs.resize(static_cast<size_t>(n_token_classes));
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_token_classes; ++i) {
        const float e                         = std::exp(token_logits[i] - max_logit);
        scratch_probs[static_cast<size_t>(i)] = e;
        sum_exp += static_cast<double>(e);
    }
    const float inv_sum = static_cast<float>(1.0 / sum_exp);
    double      entropy = 0.0;
    for (int i = 0; i < n_token_classes; ++i) {
        const float p = scratch_probs[static_cast<size_t>(i)] * inv_sum;
        // +1e-10 matches the reference.
        entropy -= static_cast<double>(p) * std::log(static_cast<double>(p) + 1e-10);
    }
    const double max_entropy = std::log(static_cast<double>(n_token_classes));
    if (max_entropy <= 0.0) {
        return 1.0f;
    }
    return static_cast<float>(1.0 - entropy / max_entropy);
}

}  // namespace

// ---------------------------------------------------------------------------
// Greedy decode driver
// ---------------------------------------------------------------------------

transcribe_status decode_tdt_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    int                        n_threads,
                                    std::vector<TdtToken> &    out_tokens,
                                    DecodeCapture *            capture) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder: enc d_model mismatch (got %d, "
                "expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;  // == vocab_size + 1
    const int n_dur       = static_cast<int>(w.tdt_durations.size());
    const int blank_id    = w.blank_id;

    // All-ggml decode on ONE shared threadpool: PredGraph owns the
    // backend + pool; enc_proj and the joint graph borrow it. pg is
    // declared first so it is destroyed LAST (jg's buffer lives on pg's
    // backend). Built per decode call, reentrant. A build failure is a
    // hard decode error.
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Two LSTM states, both pre-sized: `state` is the committed
    // state we read from each iteration; `next_state` is where the
    // predictor writes the new step's outputs. On a non-blank
    // emission we swap; on a blank emission we just discard the
    // contents of `next_state` (overwritten on the next iteration).
    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (decode-state
    // independent; one sgemm before the loop). See precompute_enc_proj_ggml.
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t      t_enc_proj_start = ggml_time_us();
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    // Per-call scratch reused across every decode step.
    std::vector<float> scratch_x;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    int last_token  = -1;  // sentinel: no previous token (start state)
    int step        = 0;
    int new_symbols = 0;

    // Runaway-protection cap; legitimate transcriptions never approach it.
    const int max_iters = 16 * T_enc + 1024;
    int       iter      = 0;
    int64_t   t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    // On a blank emission (last_token, state) are preserved, so the
    // predictor would write the same next_state next iteration. Skip the
    // LSTM unroll and reuse the previous decoder_out; recompute on a
    // non-blank emission (state.swap + last_token change). Bit-exact with
    // the unconditional path.
    bool predictor_dirty = true;

    // See ParakeetHParams::tdt_global_symbol_budget.
    const bool global_budget   = w.tdt_global_symbol_budget;
    int64_t    steps_remaining = static_cast<int64_t>(w.tdt_max_symbols) * T_enc;

    while (step < T_enc && iter < max_iters && (!global_budget || steps_remaining > 0)) {
        ++iter;

        // ----- Predictor (one LSTM step) -----
        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        // ----- Joint (using precomputed encoder projection) -----
        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        const int64_t t2 = ggml_time_us();
        t_pred_us += t1 - t0;
        t_joint_us += t2 - t1;

        // ----- Argmax (token + duration) -----
        const float * token_logits    = logits.data();
        const float * duration_logits = logits.data() + n_token_cls;

        const int pred_token = argmax_range(token_logits, n_token_cls);
        const int decision   = argmax_range(duration_logits, n_dur);
        const int duration   = w.tdt_durations[static_cast<size_t>(decision)];

        // Optional dump of the first decode step (start state, encoder
        // frame 0): embed input, per-layer LSTM h, joint logits. Mirrors
        // the Python reference dumper's decode subcommand.
        if (iter == 1 && transcribe::debug::enabled()) {
            // Predictor scratch_x at start: zeros (vector of length H).
            const long long s_h = H;
            transcribe::debug::dump_host_f32("dec.embed.0", scratch_x.data(), s_h, &s_h, 1, "decoder.embed");

            for (int layer = 0; layer < n_layers; ++layer) {
                char name_h[64];
                char name_c[64];
                std::snprintf(name_h, sizeof(name_h), "dec.lstm.%d.h.0", layer);
                std::snprintf(name_c, sizeof(name_c), "dec.lstm.%d.c.0", layer);
                transcribe::debug::dump_host_f32(name_h, next_state.h[layer].data(), s_h, &s_h, 1, "decoder.lstm");
                transcribe::debug::dump_host_f32(name_c, next_state.c[layer].data(), s_h, &s_h, 1, "decoder.lstm");
            }

            const long long s_n = w.joint.joint_n;
            transcribe::debug::dump_host_f32("dec.joint.0", logits.data(), s_n, &s_n, 1, "decoder.joint");
        }

        // ----- TDT emit + state advance -----
        const bool is_blank = (pred_token == blank_id);
        if (!is_blank) {
            const int64_t tc0 = ggml_time_us();
            const float   p   = token_confidence(token_logits, n_token_cls, scratch_probs);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = duration;
            if (capture != nullptr && out_tokens.size() % DecodeCapture::k_checkpoint_every == 0) {
                capture->checkpoints.push_back({ static_cast<int>(out_tokens.size()), last_token, state });
            }
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);  // commit
            predictor_dirty = true;
        }

        if (global_budget) {
            step += (is_blank && duration == 0) ? 1 : duration;
            --steps_remaining;
            continue;
        }

        // Step / stuck advance. Matches the reference:
        //   step += duration
        //   new_symbols += 1
        //   if duration != 0: new_symbols = 0
        //   elif max_symbols and new_symbols >= max_symbols:
        //       step += 1; new_symbols = 0
        step += duration;
        new_symbols += 1;
        if (duration != 0) {
            new_symbols = 0;
        } else if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
            step += 1;
            new_symbols = 0;
        } else if (is_blank && w.tdt_max_symbols > 0) {
            // A blank with duration 0 leaves (step, last_token, state)
            // unchanged, so every following iteration is identical until
            // max_symbols forces the step to advance. Fast-forward those
            // repeated no-op loops without changing the observable result.
            const int skip = w.tdt_max_symbols - new_symbols;
            if (skip > 0 && iter + skip < max_iters) {
                iter += skip;
                step += 1;
                new_symbols = 0;
            }
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder: %d iters, %d tokens, T_enc=%d  "
            "enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
            "total=%.1f ms  per_iter=%.0f us",
            iter, static_cast<int>(out_tokens.size()), T_enc, t_enc_proj_us / 1000.0, t_pred_us / 1000.0,
            t_joint_us / 1000.0, t_conf_us / 1000.0, (t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0,
            static_cast<double>(t_pred_us + t_joint_us) / std::max(iter, 1));

    if (iter >= max_iters && !global_budget) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder: hit iteration cap (%d) — pathological "
                "logits or loop bug",
                max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (capture != nullptr) {
        capture->enc_proj             = std::move(enc_proj_all);
        capture->enc_proj_first_frame = 0;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode
// ---------------------------------------------------------------------------
//
// Same predictor + joint forward as TDT, but the joint emits exactly
// `vocab+1` logits (no duration extras). Step rule:
//
//   per iteration:
//     run predictor (one LSTM step)
//     run joint -> token_logits[vocab+1]
//     argmax -> pred_token
//     if pred_token == blank:
//         step += 1
//         new_symbols = 0
//     else:
//         emit token, swap predictor state, last_token = pred_token
//         new_symbols += 1
//         if max_symbols and new_symbols >= max_symbols:
//             step += 1; new_symbols = 0   # break out of stuck-on-frame loop
//
// Mirrors NeMo's `RNNTGreedyDecodeInfer.step_per_frame` (without
// duration). Matches the reference dump points (`dec.embed.0`,
// `dec.lstm.{layer}.{h,c}.0`, `dec.joint.0`) emitted on iter 1.

transcribe_status decode_rnnt_greedy(const HostDecoderWeights & w,
                                     const float *              enc_out,
                                     int                        T_enc,
                                     int                        d_enc,
                                     int                        n_threads,
                                     std::vector<TdtToken> &    out_tokens,
                                     DecodeCapture *            capture) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt): enc d_model mismatch (got %d, "
                "expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;  // == vocab + 1
    const int blank_id    = w.blank_id;

    // Per-call decode graphs (see decode_tdt_greedy for the lifecycle).
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (same as TDT).
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t      t_enc_proj_start = ggml_time_us();
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    std::vector<float> scratch_x;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    int last_token  = -1;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc + 1024;
    int       iter      = 0;
    int64_t   t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    // Predictor-step cache (see decode_tdt_greedy). Bit-exact with the
    // unconditional path; elides 60-80% of predictor work on a typical
    // RNN-T iters/token ratio.
    bool predictor_dirty = true;

    while (step < T_enc && iter < max_iters) {
        ++iter;

        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        const int64_t t2 = ggml_time_us();
        t_pred_us += t1 - t0;
        t_joint_us += t2 - t1;

        // RNNT joint output is just `n_token_cls` floats (no duration extras).
        const float * token_logits = logits.data();
        const int     pred_token   = argmax_range(token_logits, n_token_cls);

        if (iter == 1 && transcribe::debug::enabled()) {
            const long long s_h = H;
            transcribe::debug::dump_host_f32("dec.embed.0", scratch_x.data(), s_h, &s_h, 1, "decoder.embed");
            for (int layer = 0; layer < n_layers; ++layer) {
                char name_h[64];
                char name_c[64];
                std::snprintf(name_h, sizeof(name_h), "dec.lstm.%d.h.0", layer);
                std::snprintf(name_c, sizeof(name_c), "dec.lstm.%d.c.0", layer);
                transcribe::debug::dump_host_f32(name_h, next_state.h[layer].data(), s_h, &s_h, 1, "decoder.lstm");
                transcribe::debug::dump_host_f32(name_c, next_state.c[layer].data(), s_h, &s_h, 1, "decoder.lstm");
            }
            const long long s_n = w.joint.joint_n;
            transcribe::debug::dump_host_f32("dec.joint.0", logits.data(), s_n, &s_n, 1, "decoder.joint");
        }

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step += 1;
            new_symbols = 0;
        } else {
            const int64_t tc0 = ggml_time_us();
            const float   p   = token_confidence(token_logits, n_token_cls, scratch_probs);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = 1;
            if (capture != nullptr && out_tokens.size() % DecodeCapture::k_checkpoint_every == 0) {
                capture->checkpoints.push_back({ static_cast<int>(out_tokens.size()), last_token, state });
            }
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder (rnnt): %d iters, %d tokens, T_enc=%d  "
            "enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
            "total=%.1f ms",
            iter, static_cast<int>(out_tokens.size()), T_enc, t_enc_proj_us / 1000.0, t_pred_us / 1000.0,
            t_joint_us / 1000.0, t_conf_us / 1000.0, (t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0);

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt): hit iteration cap (%d) — "
                "pathological logits or loop bug",
                max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (capture != nullptr) {
        capture->enc_proj             = std::move(enc_proj_all);
        capture->enc_proj_first_frame = 0;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode — streaming
// ---------------------------------------------------------------------------
//
// Same algorithm as decode_rnnt_greedy but with externally-managed
// LSTM state (state_io) and previous token (last_token_io). The
// chunk's encoder frames are decoded in stream-wide coordinates
// (step_at_emit = frame_offset + local_step). No timing log.
transcribe_status decode_rnnt_greedy_streaming(const HostDecoderWeights & w,
                                               const float *              enc_out,
                                               int                        T_enc_new,
                                               int                        d_enc,
                                               LstmState &                state_io,
                                               int &                      last_token_io,
                                               int                        frame_offset,
                                               int                        n_threads,
                                               std::vector<TdtToken> &    out_tokens,
                                               DecodeCapture *            capture) {
    if (enc_out == nullptr || T_enc_new <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Pure RNN-T greedy only: a TDT head's durations would be silently
    // ignored and a CTC head carries no predictor state. Fail loud rather
    // than mis-decode a hypothetical future TDT/CTC streaming model.
    if (w.head_kind != HostHeadKind::RNNT) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt-stream): streaming decode "
                "requires an RNN-T head; this model's head is not "
                "RNN-T");
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt-stream): enc d_model mismatch "
                "(got %d, expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;
    const int blank_id    = w.blank_id;

    // Per-call decode graphs (see decode_tdt_greedy for the lifecycle).
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Validate state_io shape; reset if degenerate.
    if (static_cast<int>(state_io.h.size()) != n_layers || static_cast<int>(state_io.c.size()) != n_layers) {
        state_io.reset(n_layers, H);
        last_token_io = -1;
    }
    for (int l = 0; l < n_layers; ++l) {
        if (static_cast<int>(state_io.h[l].size()) != H || static_cast<int>(state_io.c[l].size()) != H) {
            state_io.reset(n_layers, H);
            last_token_io = -1;
            break;
        }
    }

    // Precompute encoder projections for this chunk only.
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc_new, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (rnnt-stream): enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Working state. On return, `state` (committed after the last
    // non-blank emission, or unchanged for an all-blank chunk) and
    // last_token are copied back into state_io / last_token_io.
    LstmState state = state_io;
    LstmState next_state;
    next_state.reset(n_layers, H);

    std::vector<float> scratch_x;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    int last_token  = last_token_io;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc_new + 1024;
    int       iter      = 0;

    bool predictor_dirty = true;

    while (step < T_enc_new && iter < max_iters) {
        ++iter;

        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }

        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
            return TRANSCRIBE_ERR_BACKEND;
        }

        const float * token_logits = logits.data();
        const int     pred_token   = argmax_range(token_logits, n_token_cls);

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step += 1;
            new_symbols = 0;
        } else {
            const float p = token_confidence(token_logits, n_token_cls, scratch_probs);
            TdtToken    tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = frame_offset + step;
            tok.duration_frames = 1;
            if (capture != nullptr && out_tokens.size() % DecodeCapture::k_checkpoint_every == 0) {
                capture->checkpoints.push_back({ static_cast<int>(out_tokens.size()), last_token, state });
            }
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }
    }

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (rnnt-stream): hit iteration cap (%d)", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Commit final state. After the loop, `state` is the post-last-
    // non-blank state (the predictor's "current" since the last emit).
    state_io      = std::move(state);
    last_token_io = last_token;
    if (capture != nullptr) {
        capture->enc_proj.insert(capture->enc_proj.end(), enc_proj_all.begin(),
                                 enc_proj_all.begin() + static_cast<std::ptrdiff_t>(T_enc_new) * joint_h);
    }
    return TRANSCRIBE_OK;
}

// CTC greedy decode.
//
// Per-frame: logits[t] = W @ enc[t] + b -> log_softmax -> argmax.
// Collapse: drop adjacent duplicates ("aaab" -> "ab"), then drop blanks
// (standard CTC greedy, cf. NeMo GreedyCTCInfer).
transcribe_status decode_ctc_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    int                        n_threads,
                                    std::vector<TdtToken> &    out_tokens) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.ctc_head.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (ctc): enc d_model mismatch "
                "(got %d, expected %d)",
                d_enc, w.ctc_head.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int n_classes = w.ctc_head.n_classes;
    const int blank_id  = w.ctc_head.blank_id;

    // Compute T_enc frames of logits: [T_enc, n_classes] = enc_out[T_enc, d_enc] @ W^T + b.
    std::vector<float> logits_all(static_cast<size_t>(T_enc) * static_cast<size_t>(n_classes));

    const int64_t t_proj_start = ggml_time_us();
#if TRANSCRIBE_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, n_classes, d_enc, 1.0f, enc_out, d_enc,
                w.ctc_head.weight.data(), d_enc, 0.0f, logits_all.data(), n_classes);
#else
    // No BLAS: project all T frames in parallel over parallel_for_all
    // (rows within a frame are a serial, auto-vectorized dot). Bias folded
    // in below.
    {
        const int     nt = resolve_decode_threads(n_threads);
        const float * Wc = w.ctc_head.weight.data();
        transcribe::parallel_for_all(T_enc, nt, [&](int t) {
            const float * frame = enc_out + static_cast<size_t>(t) * static_cast<size_t>(d_enc);
            float *       row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
            for (int r = 0; r < n_classes; ++r) {
                const float * wr  = Wc + static_cast<size_t>(r) * static_cast<size_t>(d_enc);
                float         acc = 0.0f;
                for (int c = 0; c < d_enc; ++c) {
                    acc += wr[c] * frame[c];
                }
                row[r] = acc;
            }
            return true;
        });
    }
#endif
    for (int t = 0; t < T_enc; ++t) {
        float * row = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        for (int c = 0; c < n_classes; ++c) {
            row[c] += w.ctc_head.bias[static_cast<size_t>(c)];
        }
    }
    const int64_t t_proj_us = ggml_time_us() - t_proj_start;

    // Per-frame log-softmax in place. The reference dumps log-probs,
    // not raw logits.
    for (int t = 0; t < T_enc; ++t) {
        float * row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        float   max_v = row[0];
        for (int c = 1; c < n_classes; ++c) {
            if (row[c] > max_v) {
                max_v = row[c];
            }
        }
        double sum = 0.0;
        for (int c = 0; c < n_classes; ++c) {
            sum += std::exp(static_cast<double>(row[c] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int c = 0; c < n_classes; ++c) {
            row[c] -= log_sum;
        }
    }

    // Optional dumps. Frame 0 is shape [n_classes]; full is
    // [T_enc, n_classes] in row-major (== ggml ne fast-to-slow
    // [n_classes, T_enc]).
    if (transcribe::debug::enabled()) {
        const long long s_n = n_classes;
        transcribe::debug::dump_host_f32("dec.ctc.logprobs.0", logits_all.data(), s_n, &s_n, 1,
                                         "decoder.ctc.logprobs.0");
        const long long shape_full[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32("dec.ctc.logprobs", logits_all.data(),
                                         static_cast<long long>(logits_all.size()), shape_full, 2,
                                         "decoder.ctc.logprobs");
    }

    // Greedy collapse: drop runs of identical labels, then drop blanks.
    // Per-emit `step_at_emit` is the encoder frame at which the label
    // first changed; duration_frames = 1.
    int prev_label = -1;
    for (int t = 0; t < T_enc; ++t) {
        const float * row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        const int     label = argmax_range(row, n_classes);
        if (label == prev_label) {
            continue;  // run-length collapse
        }
        prev_label = label;
        if (label == blank_id) {
            continue;
        }
        TdtToken tok;
        tok.id              = label;
        // CTC greedy confidence = exp(top log-prob); this is the
        // model's own probability for the chosen label, in [0, 1].
        tok.p               = std::exp(row[label]);
        tok.step_at_emit    = t;
        tok.duration_frames = 1;
        out_tokens.push_back(tok);
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "decoder (ctc): T_enc=%d  proj=%.1f ms  emitted=%d", T_enc, t_proj_us / 1000.0,
            static_cast<int>(out_tokens.size()));

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Vocabulary rescoring (see vocabulary.h)
// ---------------------------------------------------------------------------

namespace {

constexpr double k_neg_inf = -1e30;

double log_add(double a, double b) {
    if (a <= k_neg_inf) {
        return b;
    }
    if (b <= k_neg_inf) {
        return a;
    }
    const double hi = std::max(a, b);
    return hi + std::log1p(std::exp(std::min(a, b) - hi));
}

// The joint over a window of W encoder frames for one predictor output:
// logits[joint_n, W] = out_w @ act(enc_proj[:, t] + pred_proj) + out_b.
// Built per window on the decoder pool; enc_in is set once, pred_in per
// predictor output.
struct WindowJoint {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;  // BORROWED
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         enc_in  = nullptr;
    ggml_tensor *         pred_in = nullptr;
    ggml_tensor *         logits  = nullptr;
    int                   W       = 0;

    WindowJoint() = default;

    ~WindowJoint() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    WindowJoint(const WindowJoint &)             = delete;
    WindowJoint & operator=(const WindowJoint &) = delete;
};

bool build_window_joint(WindowJoint & g, const HostJoint & j, ggml_backend_t backend, int W) {
    g.backend = backend;
    g.W       = W;
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }
    g.enc_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, j.joint_h, W);
    ggml_set_input(g.enc_in);
    g.pred_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.pred_hidden);
    ggml_set_input(g.pred_in);
    ggml_tensor * pred_proj = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.g_pred_w, g.pred_in), j.g_pred_b);
    ggml_tensor * summed    = ggml_add(g.ctx, g.enc_in, pred_proj);  // [joint_h] broadcast over W
    ggml_tensor * activated;
    if (j.activation == "relu") {
        activated = ggml_relu(g.ctx, summed);
    } else if (j.activation == "sigmoid") {
        activated = ggml_sigmoid(g.ctx, summed);
    } else {
        activated = ggml_tanh(g.ctx, summed);
    }
    g.logits = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.gw_w, activated), j.gw_b);  // [joint_n, W]
    ggml_set_output(g.logits);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (g.buf == nullptr) {
        return false;
    }
    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.logits);
    return true;
}

double log_sum_exp(const float * v, int n) {
    float hi = v[0];
    for (int i = 1; i < n; ++i) {
        hi = std::max(hi, v[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(v[i] - hi));
    }
    return static_cast<double>(hi) + std::log(sum);
}

struct PredictorSnapshot {
    LstmState state;
    int       last_token = -1;
};

struct WindowScorer {
    const HostDecoderWeights & w;
    PredGraph &                pg;
    WindowJoint &              joint;
    std::vector<float>         scratch_x;
    std::vector<float>         logits;
    int64_t                    joint_columns = 0;

    // log P(ids | frames [s, s + W)) under the transducer, starting from
    // `start`. RNN-T: every alignment that has emitted all of `ids` when it
    // reaches the window's last frame, then emits next_id there (or blank
    // when next_id < 0). TDT: every alignment that has emitted all of `ids`
    // by the last frame, with token and duration probabilities; arrivals
    // that jump past the last frame count as reaching it.
    //
    // Every alignment emits each id once at some frame and pays nothing
    // positive elsewhere, so the sum of each id's best frame bounds the
    // result. Once that bound falls below `floor`, scoring stops and `out`
    // is the bound (a term that cannot win costs only the rows it lost on).
    bool score(const PredictorSnapshot &    start,
               const std::vector<int32_t> & ids,
               int                          next_id,
               double                       floor,
               double &                     out) {
        const int  W           = joint.W;
        const int  U           = static_cast<int>(ids.size());
        const int  n_token_cls = w.predictor.pred_vocab;
        const int  joint_n     = w.joint.joint_n;
        const bool tdt         = w.head_kind == HostHeadKind::TDT;
        const int  n_dur       = tdt ? static_cast<int>(w.tdt_durations.size()) : 0;
        const int  blank       = w.blank_id;

        // lp_blank[u][t], lp_tok[u][t], lp_dur[(u*W + t)*n_dur + k]
        std::vector<double> lp_blank(static_cast<size_t>((U + 1) * W), k_neg_inf);
        std::vector<double> lp_tok(static_cast<size_t>((U + 1) * W), k_neg_inf);
        std::vector<double> lp_dur(static_cast<size_t>((U + 1) * W * std::max(n_dur, 1)), k_neg_inf);
        double              lp_next = k_neg_inf;

        LstmState state = start.state;
        LstmState next_state;
        next_state.reset(static_cast<int>(w.predictor.lstm.size()), w.predictor.pred_hidden);
        int token = start.last_token;
        logits.resize(static_cast<size_t>(joint_n) * static_cast<size_t>(W));
        double bound = 0.0;
        for (int u = 0; u <= U; ++u) {
            const float * g = predictor_step_ggml(w.predictor, pg, token, state, next_state, scratch_x);
            if (g == nullptr) {
                return false;
            }
            ggml_backend_tensor_set(joint.pred_in, g, 0, static_cast<size_t>(w.joint.pred_hidden) * sizeof(float));
            if (ggml_backend_graph_compute(joint.backend, joint.graph) != GGML_STATUS_SUCCESS) {
                return false;
            }
            ggml_backend_tensor_get(joint.logits, logits.data(), 0, logits.size() * sizeof(float));
            joint_columns += W;
            for (int t = 0; t < W; ++t) {
                const float * col = logits.data() + static_cast<size_t>(t) * static_cast<size_t>(joint_n);
                const double  lse = log_sum_exp(col, n_token_cls);
                const size_t  ut  = static_cast<size_t>(u * W + t);
                lp_blank[ut]      = col[blank] - lse;
                if (u < U) {
                    lp_tok[ut] = col[ids[static_cast<size_t>(u)]] - lse;
                }
                if (tdt) {
                    const double dlse = log_sum_exp(col + n_token_cls, n_dur);
                    for (int k = 0; k < n_dur; ++k) {
                        lp_dur[ut * static_cast<size_t>(n_dur) + static_cast<size_t>(k)] = col[n_token_cls + k] - dlse;
                    }
                } else if (u == U && t == W - 1) {
                    lp_next = next_id >= 0 ? col[next_id] - lse : lp_blank[ut];
                }
            }
            if (u < U) {
                double best = k_neg_inf;
                for (int t = 0; t < W; ++t) {
                    best = std::max(best, lp_tok[static_cast<size_t>(u * W + t)]);
                }
                bound += best;
                if (bound < floor) {
                    out = bound;
                    return true;
                }
            }
            std::swap(state, next_state);
            if (u < U) {
                token = ids[static_cast<size_t>(u)];
            }
        }

        if (!tdt) {
            // alpha[t][u]: emitted ids[0..u) and standing on frame t.
            std::vector<double> alpha(static_cast<size_t>((U + 1) * W), k_neg_inf);
            for (int t = 0; t < W; ++t) {
                for (int u = 0; u <= U; ++u) {
                    double a = (t == 0 && u == 0) ? 0.0 : k_neg_inf;
                    if (t > 0) {
                        a = log_add(a, alpha[static_cast<size_t>((t - 1) * (U + 1) + u)] +
                                           lp_blank[static_cast<size_t>(u * W + t - 1)]);
                    }
                    if (u > 0) {
                        a = log_add(a, alpha[static_cast<size_t>(t * (U + 1) + u - 1)] +
                                           lp_tok[static_cast<size_t>((u - 1) * W + t)]);
                    }
                    alpha[static_cast<size_t>(t * (U + 1) + u)] = a;
                }
            }
            out = alpha[static_cast<size_t>((W - 1) * (U + 1) + U)] + lp_next;
            return true;
        }

        // TDT: expand from every frame; a jump to or past the last frame
        // with all ids emitted ends the alignment (blank jumps from the
        // last frame itself are not taken, so nothing is counted twice).
        int max_dur = 0;
        for (const int32_t d : w.tdt_durations) {
            max_dur = std::max(max_dur, static_cast<int>(d));
        }
        const int           T2 = W + max_dur;
        std::vector<double> alpha(static_cast<size_t>((U + 1) * T2), k_neg_inf);
        alpha[0]   = 0.0;
        double end = k_neg_inf;
        for (int t = 0; t < W; ++t) {
            for (int u = 0; u <= U; ++u) {
                const double a = alpha[static_cast<size_t>(t * (U + 1) + u)];
                if (a <= k_neg_inf) {
                    continue;
                }
                const size_t ut = static_cast<size_t>(u * W + t);
                for (int k = 0; k < n_dur; ++k) {
                    const int    d  = w.tdt_durations[static_cast<size_t>(k)];
                    const double ld = lp_dur[ut * static_cast<size_t>(n_dur) + static_cast<size_t>(k)];
                    if (u < U) {
                        auto & dst = alpha[static_cast<size_t>((t + d) * (U + 1) + u + 1)];
                        dst        = log_add(dst, a + lp_tok[ut] + ld);
                    }
                    if (d > 0 && !(u == U && t == W - 1)) {
                        auto & dst = alpha[static_cast<size_t>((t + d) * (U + 1) + u)];
                        dst        = log_add(dst, a + lp_blank[ut] + ld);
                    }
                }
            }
        }
        for (int t = W - 1; t < T2; ++t) {
            end = log_add(end, alpha[static_cast<size_t>(t * (U + 1) + U)]);
        }
        out = end;
        return true;
    }
};

struct ScoredCandidate {
    VocabularyCandidate c;
    double              gain = 0.0;
};

}  // namespace

transcribe_status rescore_vocabulary(const HostDecoderWeights &    w,
                                     const transcribe::Tokenizer & tok,
                                     const VocabularyBias &        bias,
                                     const DecodeCapture &         capture,
                                     int                           n_threads,
                                     std::vector<TdtToken> &       tokens) {
    const int64_t joint_h  = w.joint.joint_h;
    const int     n_frames = joint_h > 0 ? static_cast<int>(capture.enc_proj.size() / static_cast<size_t>(joint_h)) : 0;
    if (!bias.active() || tokens.empty() || n_frames <= 0 || capture.checkpoints.empty() ||
        w.head_kind == HostHeadKind::CTC) {
        return TRANSCRIBE_OK;
    }
    const float *                          enc_proj             = capture.enc_proj.data();
    const int64_t                          enc_proj_first_frame = capture.enc_proj_first_frame;
    const int64_t                          t_start              = ggml_time_us();
    const std::vector<VocabularyWord>      words                = vocabulary_words(tok, tokens);
    const std::vector<VocabularyCandidate> candidates           = find_vocabulary_candidates(words, bias);
    if (candidates.empty()) {
        return TRANSCRIBE_OK;
    }

    PredGraph pg;
    build_pred_graph(pg, w.predictor, resolve_decode_threads(n_threads));
    if (!pg.ready) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Predictor state before each candidate's first token: the decoder's
    // nearest checkpoint at or before it, stepped through the tokens in
    // between (the same lazy state machine as the decoders). A candidate
    // before the first retained checkpoint is not rescored.
    std::vector<int> needed;
    for (const VocabularyCandidate & c : candidates) {
        if (c.first >= capture.checkpoints.front().token) {
            needed.push_back(c.first);
        }
    }
    std::sort(needed.begin(), needed.end());
    needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
    std::vector<PredictorSnapshot> snapshots(needed.size());
    {
        const int          n_layers = static_cast<int>(w.predictor.lstm.size());
        const int          H        = w.predictor.pred_hidden;
        LstmState          next_state;
        std::vector<float> scratch_x;
        next_state.reset(n_layers, H);
        for (size_t k = 0; k < needed.size(); ++k) {
            const auto cp    = std::upper_bound(capture.checkpoints.begin(), capture.checkpoints.end(), needed[k],
                                                [](int t, const PredictorCheckpoint & c) { return t < c.token; }) -
                               1;
            LstmState  state = cp->state;
            int        last  = cp->last_token;
            for (int i = cp->token; i < needed[k]; ++i) {
                if (predictor_step_ggml(w.predictor, pg, last, state, next_state, scratch_x) == nullptr) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                std::swap(state, next_state);
                last = tokens[static_cast<size_t>(i)].id;
            }
            snapshots[k].state      = std::move(state);
            snapshots[k].last_token = last;
        }
    }

    const int64_t                t_replayed = ggml_time_us();
    const int64_t                last_frame = enc_proj_first_frame + n_frames - 1;
    std::vector<ScoredCandidate> accepted;
    int64_t                      joint_columns = 0;
    int                          scored        = 0;
    for (const VocabularyCandidate & c : candidates) {
        const TdtToken & first = tokens[static_cast<size_t>(c.first)];
        const TdtToken & lastk = tokens[static_cast<size_t>(c.end - 1)];
        // Window: a few frames either side of the span, clipped to the
        // neighbouring tokens' frames. When the next token falls inside the
        // tail, both paths must end by emitting it there; otherwise by a
        // blank at the window's last frame.
        int64_t          s     = std::max<int64_t>(first.step_at_emit - k_vocab_window_lead, 0);
        if (c.first > 0) {
            s = std::max<int64_t>(s, tokens[static_cast<size_t>(c.first - 1)].step_at_emit);
        }
        int     next_id = -1;
        int64_t e       = lastk.step_at_emit + std::max(lastk.duration_frames, 1) + k_vocab_window_tail;
        if (c.end < static_cast<int>(tokens.size())) {
            const TdtToken & nx = tokens[static_cast<size_t>(c.end)];
            if (nx.step_at_emit <= e) {
                e       = nx.step_at_emit;
                next_id = nx.id;
            }
        }
        e = std::min(e, last_frame);
        if (s < enc_proj_first_frame || e < s || e - s + 1 > k_vocab_max_window_frames) {
            continue;
        }
        const int W = static_cast<int>(e - s + 1);

        WindowJoint joint;
        if (!build_window_joint(joint, w.joint, pg.backend, W)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        ggml_backend_tensor_set(
            joint.enc_in,
            enc_proj + static_cast<size_t>(s - enc_proj_first_frame) * static_cast<size_t>(w.joint.joint_h), 0,
            static_cast<size_t>(W) * static_cast<size_t>(w.joint.joint_h) * sizeof(float));

        const auto snap_at = std::lower_bound(needed.begin(), needed.end(), c.first);
        if (snap_at == needed.end() || *snap_at != c.first) {
            continue;
        }
        const size_t         snap = static_cast<size_t>(snap_at - needed.begin());
        std::vector<int32_t> heard;
        for (int i = c.first; i < c.end; ++i) {
            heard.push_back(tokens[static_cast<size_t>(i)].id);
        }
        WindowScorer scorer{ w, pg, joint, {}, {}, 0 };
        double       heard_lp = 0.0;
        double       term_lp  = 0.0;
        const auto & term     = bias.terms[static_cast<size_t>(c.term)];
        const double boost    = static_cast<double>(bias.boost) * static_cast<double>(term.ids.size());
        if (!scorer.score(snapshots[snap], heard, next_id, k_neg_inf, heard_lp) ||
            !scorer.score(snapshots[snap], term.ids, next_id, heard_lp - boost, term_lp)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        joint_columns += scorer.joint_columns;
        ++scored;
        const double gain = term_lp + boost - heard_lp;
        std::string  span;
        for (const VocabularyWord & word : words) {
            if (word.first >= c.first && word.first < c.end) {
                span += span.empty() ? word.text : " " + word.text;
            }
        }
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "parakeet vocabulary: \"%s\" for \"%s\" frames %lld-%lld heard %.2f term %.2f boost %.2f -> %s",
                term.text.c_str(), span.c_str(), static_cast<long long>(s), static_cast<long long>(e), heard_lp,
                term_lp, boost, gain >= 0.0 ? "replace" : "keep");
        if (gain >= 0.0) {
            accepted.push_back({ c, gain });
        }
    }

    // Best gain first; overlapping spans give way.
    std::stable_sort(accepted.begin(), accepted.end(),
                     [](const ScoredCandidate & a, const ScoredCandidate & b) { return a.gain > b.gain; });
    std::vector<ScoredCandidate> chosen;
    for (const ScoredCandidate & a : accepted) {
        bool overlaps = false;
        for (const ScoredCandidate & b : chosen) {
            overlaps = overlaps || (a.c.first < b.c.end && b.c.first < a.c.end);
        }
        if (!overlaps) {
            chosen.push_back(a);
        }
    }
    std::sort(chosen.begin(), chosen.end(),
              [](const ScoredCandidate & a, const ScoredCandidate & b) { return a.c.first < b.c.first; });

    if (!chosen.empty()) {
        std::vector<TdtToken> out;
        out.reserve(tokens.size() + 16);
        int i = 0;
        for (const ScoredCandidate & ch : chosen) {
            for (; i < ch.c.first; ++i) {
                out.push_back(tokens[static_cast<size_t>(i)]);
            }
            const TdtToken & first = tokens[static_cast<size_t>(ch.c.first)];
            const TdtToken & lastk = tokens[static_cast<size_t>(ch.c.end - 1)];
            float            p     = 1.0f;
            for (int k = ch.c.first; k < ch.c.end; ++k) {
                p = std::min(p, tokens[static_cast<size_t>(k)].p);
            }
            const auto & ids  = bias.terms[static_cast<size_t>(ch.c.term)].ids;
            const int    n    = static_cast<int>(ids.size());
            const int    span = lastk.step_at_emit - first.step_at_emit;
            for (int k = 0; k < n; ++k) {
                TdtToken t;
                t.id              = ids[static_cast<size_t>(k)];
                t.p               = p;
                t.step_at_emit    = first.step_at_emit + (n > 1 ? span * k / (n - 1) : 0);
                t.duration_frames = (k == n - 1) ? lastk.duration_frames : 1;
                out.push_back(t);
            }
            i = ch.c.end;
        }
        for (; i < static_cast<int>(tokens.size()); ++i) {
            out.push_back(tokens[static_cast<size_t>(i)]);
        }
        tokens.swap(out);
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
            "parakeet vocabulary: %d candidates, %d scored, %d replaced, %lld joint columns, %.1f ms (replay %.1f ms)",
            static_cast<int>(candidates.size()), scored, static_cast<int>(chosen.size()),
            static_cast<long long>(joint_columns), (ggml_time_us() - t_start) / 1000.0,
            (t_replayed - t_start) / 1000.0);
    return TRANSCRIBE_OK;
}


// ---------------------------------------------------------------------------
// RNN-T beam search
// ---------------------------------------------------------------------------

namespace {

// One LSTM step for N hypotheses at once: PredGraph's graph with a batch
// dimension, on PredGraph's backend (borrowed).
struct BatchPredGraph {
    ggml_context *             ctx   = nullptr;
    ggml_backend_buffer_t      buf   = nullptr;
    ggml_cgraph *              graph = nullptr;
    ggml_tensor *              x     = nullptr;  // [H, N]
    std::vector<ggml_tensor *> ph, pc, nh, nc;   // [H, N] per layer
    ggml_tensor *              proj = nullptr;   // [joint_h, N] pred_w @ h + pred_b
    int                        N    = 0;

    BatchPredGraph() = default;

    ~BatchPredGraph() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    BatchPredGraph(const BatchPredGraph &)             = delete;
    BatchPredGraph & operator=(const BatchPredGraph &) = delete;
};

bool build_batch_pred_graph(BatchPredGraph &      g,
                            const HostPredictor & p,
                            const HostJoint &     j,
                            ggml_backend_t        backend,
                            int                   N) {
    const int H = p.pred_hidden;
    const int L = static_cast<int>(p.lstm.size());
    g.N         = N;
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(48 * L + 16) + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }
    g.x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, H, N);
    ggml_set_input(g.x);
    g.ph.assign(static_cast<size_t>(L), nullptr);
    g.pc.assign(static_cast<size_t>(L), nullptr);
    g.nh.assign(static_cast<size_t>(L), nullptr);
    g.nc.assign(static_cast<size_t>(L), nullptr);
    for (int l = 0; l < L; ++l) {
        g.ph[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, H, N);
        ggml_set_input(g.ph[l]);
        g.pc[l] = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, H, N);
        ggml_set_input(g.pc[l]);
    }
    ggml_tensor * in = g.x;
    for (int l = 0; l < L; ++l) {
        const auto &  lh = p.lstm[static_cast<size_t>(l)];
        ggml_tensor * gates =
            ggml_add(g.ctx, ggml_add(g.ctx, ggml_mul_mat(g.ctx, lh.g_Wx, in), ggml_mul_mat(g.ctx, lh.g_Wh, g.ph[l])),
                     lh.g_b);  // [4H, N]
        auto part = [&](int k) {
            return ggml_cont(g.ctx, ggml_view_2d(g.ctx, gates, H, N, gates->nb[1],
                                                 static_cast<size_t>(k) * static_cast<size_t>(H) *
                                                     ggml_element_size(gates)));
        };
        ggml_tensor * i_ = ggml_sigmoid(g.ctx, part(0));
        ggml_tensor * f_ = ggml_sigmoid(g.ctx, part(1));
        ggml_tensor * gg = ggml_tanh(g.ctx, part(2));
        ggml_tensor * o_ = ggml_sigmoid(g.ctx, part(3));
        g.nc[l]          = ggml_add(g.ctx, ggml_mul(g.ctx, f_, g.pc[l]), ggml_mul(g.ctx, i_, gg));
        ggml_set_output(g.nc[l]);
        g.nh[l] = ggml_mul(g.ctx, o_, ggml_tanh(g.ctx, g.nc[l]));
        ggml_set_output(g.nh[l]);
        in = g.nh[l];
    }
    g.proj = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.g_pred_w, in), j.g_pred_b);
    ggml_set_output(g.proj);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (g.buf == nullptr) {
        return false;
    }
    g.graph = ggml_new_graph(g.ctx);
    for (int l = 0; l < L; ++l) {
        ggml_build_forward_expand(g.graph, g.nh[l]);
        ggml_build_forward_expand(g.graph, g.nc[l]);
    }
    ggml_build_forward_expand(g.graph, g.proj);
    return true;
}

// The joint for N hypotheses at one encoder frame, from each one's
// projected predictor output (computed once, with its predictor step):
// logits[joint_n, N] = out_w @ act(enc_proj + proj[:, n]) + out_b.
struct BatchJoint {
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         proj_in = nullptr;  // [joint_h, N]
    ggml_tensor *         enc_in  = nullptr;  // [joint_h]
    ggml_tensor *         logits  = nullptr;  // [joint_n, N]
    int                   N       = 0;

    BatchJoint() = default;

    ~BatchJoint() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    BatchJoint(const BatchJoint &)             = delete;
    BatchJoint & operator=(const BatchJoint &) = delete;
};

bool build_batch_joint(BatchJoint & g, const HostJoint & j, ggml_backend_t backend, int N) {
    g.N = N;
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }
    g.proj_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, j.joint_h, N);
    ggml_set_input(g.proj_in);
    g.enc_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.joint_h);
    ggml_set_input(g.enc_in);
    ggml_tensor * summed = ggml_add(g.ctx, g.proj_in, g.enc_in);
    ggml_tensor * activated;
    if (j.activation == "relu") {
        activated = ggml_relu(g.ctx, summed);
    } else if (j.activation == "sigmoid") {
        activated = ggml_sigmoid(g.ctx, summed);
    } else {
        activated = ggml_tanh(g.ctx, summed);
    }
    g.logits = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.gw_w, activated), j.gw_b);
    ggml_set_output(g.logits);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (g.buf == nullptr) {
        return false;
    }
    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.logits);
    return true;
}

// One emitted token of some hypothesis. Hypotheses share their prefixes
// through `parent`. A node at token index i % k_checkpoint_every == 0
// holds the predictor as it stood before that token, for rescoring; the
// nodes no surviving hypothesis reaches drop theirs (see gc below).
struct BeamNode {
    int                                  parent = -1;
    int                                  id     = 0;
    float                                p      = 0.0f;
    int                                  step   = 0;
    std::unique_ptr<PredictorCheckpoint> checkpoint;
};

struct BeamHyp {
    double     score = 0.0;  // log P(tokens, alignment so far) + boost earned
    BoostState boost;
    uint64_t   key        = 0;  // hash of the token sequence
    int        tail       = -1;
    int        len        = 0;
    int        last_token = -1;
    LstmState  prev;   // the predictor before consuming last_token
    LstmState  state;  // and after; state.h.back() is the decoder output
    std::shared_ptr<const std::vector<float>> proj;  // pred_w @ state.h.back() + pred_b
};

// A predictor step's result, kept by token-sequence key: in a long pause the
// same expansion is proposed frame after frame, and the LSTM is the costliest
// part of a step.
struct PredResult {
    LstmState                                 state;
    std::shared_ptr<const std::vector<float>> proj;
};

struct BeamExpansion {
    int        parent = 0;
    int        token  = 0;
    double     score  = 0.0;
    BoostState boost;
};

uint64_t extend_key(uint64_t key, int token) {
    uint64_t z = key + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(token + 2);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Add `h` to `set`, summing the probability of a hypothesis with the same
// tokens reached by another alignment.
void merge_hyp(std::vector<BeamHyp> & set, BeamHyp && h) {
    for (BeamHyp & b : set) {
        if (b.key == h.key && b.len == h.len) {
            const double sum = log_add(b.score, h.score);
            if (h.score > b.score) {
                b = std::move(h);
            }
            b.score = sum;
            return;
        }
    }
    set.push_back(std::move(h));
}

}  // namespace

transcribe_status decode_rnnt_beam(const HostDecoderWeights & w,
                                   const float *              enc_out,
                                   int                        T_enc,
                                   int                        d_enc,
                                   int                        n_threads,
                                   const BeamOptions &        options,
                                   const VocabularyTree *     tree,
                                   std::vector<TdtToken> &    out_tokens,
                                   DecodeCapture *            capture) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc || w.head_kind != HostHeadKind::RNNT) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (tree != nullptr && tree->empty()) {
        tree = nullptr;
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;
    const int joint_n     = w.joint.joint_n;
    const int joint_h     = w.joint.joint_h;
    const int blank_id    = w.blank_id;
    const int beam        = std::clamp(options.beam, 1, 16);
    const int max_sym     = w.tdt_max_symbols > 0 ? w.tdt_max_symbols : 10;

    PredGraph pg;
    build_pred_graph(pg, w.predictor, nt);
    if (!pg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (beam): ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    std::vector<std::unique_ptr<BatchPredGraph>> preds(static_cast<size_t>(beam) + 1);
    std::vector<std::unique_ptr<BatchJoint>>     joints(static_cast<size_t>(beam) + 1);

    const int64_t      t_start = ggml_time_us();
    std::vector<float> enc_proj_all;
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (beam): enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    const size_t       hb = static_cast<size_t>(H) * sizeof(float);
    std::vector<float> x_host;
    int64_t            t_pred_us = 0, t_joint_us = 0;
    int                n_pred = 0, n_joint = 0;

    // Predictor step for each (token, state) pair: out[k] = step(tokens[k], *prev[k]).
    auto predict = [&](const std::vector<int> & tokens, const std::vector<const LstmState *> & prev,
                       std::vector<PredResult> & out) -> bool {
        const int n  = static_cast<int>(tokens.size());
        auto &    gp = preds[static_cast<size_t>(n)];
        if (gp == nullptr) {
            gp = std::make_unique<BatchPredGraph>();
            if (!build_batch_pred_graph(*gp, w.predictor, w.joint, pg.backend, n)) {
                return false;
            }
        }
        const int64_t t0 = ggml_time_us();
        x_host.assign(static_cast<size_t>(n) * static_cast<size_t>(H), 0.0f);
        for (int k = 0; k < n; ++k) {
            if (tokens[static_cast<size_t>(k)] >= 0) {
                std::memcpy(x_host.data() + static_cast<size_t>(k) * static_cast<size_t>(H),
                            w.predictor.embed_w.data() +
                                static_cast<size_t>(tokens[static_cast<size_t>(k)]) * static_cast<size_t>(H),
                            hb);
            }
        }
        ggml_backend_tensor_set(gp->x, x_host.data(), 0, hb * static_cast<size_t>(n));
        for (int k = 0; k < n; ++k) {
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_set(gp->ph[l], prev[static_cast<size_t>(k)]->h[static_cast<size_t>(l)].data(),
                                        hb * static_cast<size_t>(k), hb);
                ggml_backend_tensor_set(gp->pc[l], prev[static_cast<size_t>(k)]->c[static_cast<size_t>(l)].data(),
                                        hb * static_cast<size_t>(k), hb);
            }
        }
        if (ggml_backend_graph_compute(pg.backend, gp->graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        out.resize(static_cast<size_t>(n));
        const size_t jb = static_cast<size_t>(joint_h) * sizeof(float);
        for (int k = 0; k < n; ++k) {
            PredResult & r = out[static_cast<size_t>(k)];
            r.state.reset(n_layers, H);
            for (int l = 0; l < n_layers; ++l) {
                ggml_backend_tensor_get(gp->nh[l], r.state.h[static_cast<size_t>(l)].data(),
                                        hb * static_cast<size_t>(k), hb);
                ggml_backend_tensor_get(gp->nc[l], r.state.c[static_cast<size_t>(l)].data(),
                                        hb * static_cast<size_t>(k), hb);
            }
            auto proj = std::make_shared<std::vector<float>>(static_cast<size_t>(joint_h));
            ggml_backend_tensor_get(gp->proj, proj->data(), jb * static_cast<size_t>(k), jb);
            r.proj = std::move(proj);
        }
        t_pred_us += ggml_time_us() - t0;
        ++n_pred;
        return true;
    };

    // Joint logits for every hypothesis of `set` at frame t, [joint_n] each.
    std::vector<float> logits;
    auto               joint = [&](const std::vector<BeamHyp> & set, int t) -> bool {
        const int n  = static_cast<int>(set.size());
        auto &    gj = joints[static_cast<size_t>(n)];
        if (gj == nullptr) {
            gj = std::make_unique<BatchJoint>();
            if (!build_batch_joint(*gj, w.joint, pg.backend, n)) {
                return false;
            }
        }
        const int64_t t0 = ggml_time_us();
        const size_t  pb = static_cast<size_t>(joint_h) * sizeof(float);
        for (int k = 0; k < n; ++k) {
            ggml_backend_tensor_set(gj->proj_in, set[static_cast<size_t>(k)].proj->data(), pb * static_cast<size_t>(k),
                                    pb);
        }
        ggml_backend_tensor_set(gj->enc_in, enc_proj_all.data() + static_cast<size_t>(t) * static_cast<size_t>(joint_h),
                                0, static_cast<size_t>(joint_h) * sizeof(float));
        if (ggml_backend_graph_compute(pg.backend, gj->graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        logits.resize(static_cast<size_t>(joint_n) * static_cast<size_t>(n));
        ggml_backend_tensor_get(gj->logits, logits.data(), 0, logits.size() * sizeof(float));
        t_joint_us += ggml_time_us() - t0;
        ++n_joint;
        return true;
    };

    std::vector<BeamNode> nodes;
    std::vector<BeamHyp>  B(1);
    {
        BeamHyp & start = B[0];
        start.prev.reset(n_layers, H);
        std::vector<PredResult> out;
        if (!predict({ -1 }, { &start.prev }, out)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        start.state = std::move(out[0].state);
        start.proj  = std::move(out[0].proj);
    }

    // Root children: every term's first piece, offered to every hypothesis.
    std::vector<int> root_tokens;
    if (tree != nullptr) {
        for (const auto & [id, next] : tree->nodes[0].children) {
            root_tokens.push_back(id);
        }
    }

    std::vector<BeamHyp>       A;
    std::vector<BeamExpansion> expansions;
    std::vector<double>        best_by_parent;
    std::vector<int>           cand;
    std::vector<float>         scratch_probs;
    std::vector<uint8_t>       reached;
    std::vector<PredResult>    new_states;
    std::unordered_map<uint64_t, PredResult> pred_cache;
    constexpr size_t           k_pred_cache_max = 512;
    int                        n_cached         = 0;
    const int                  k_cp  = DecodeCapture::k_checkpoint_every;
    const bool                 trace = std::getenv("TRANSCRIBE_PARAKEET_BEAM_TRACE") != nullptr;

    for (int t = 0; t < T_enc; ++t) {
        A.swap(B);
        B.clear();
        for (int s = 0; s <= max_sym && !A.empty(); ++s) {
            if (!joint(A, t)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            const int n = static_cast<int>(A.size());
            expansions.clear();
            best_by_parent.assign(static_cast<size_t>(n), k_neg_inf);
            for (int i = 0; i < n; ++i) {
                const float *   col = logits.data() + static_cast<size_t>(i) * static_cast<size_t>(joint_n);
                const double    lse = log_sum_exp(col, n_token_cls);
                const BeamHyp & a   = A[static_cast<size_t>(i)];
                best_by_parent[static_cast<size_t>(i)] = a.score + (col[blank_id] - lse);
                {
                    BeamHyp b = a;
                    b.score += col[blank_id] - lse;
                    merge_hyp(B, std::move(b));
                }
                if (s == max_sym) {
                    continue;
                }
                // The `beam` likeliest tokens, plus any that continue or start a term.
                cand.clear();
                for (int id = 0; id < n_token_cls; ++id) {
                    if (id == blank_id) {
                        continue;
                    }
                    if (static_cast<int>(cand.size()) < beam) {
                        cand.push_back(id);
                        std::push_heap(cand.begin(), cand.end(), [&](int x, int y) { return col[x] > col[y]; });
                    } else if (col[id] > col[cand.front()]) {
                        std::pop_heap(cand.begin(), cand.end(), [&](int x, int y) { return col[x] > col[y]; });
                        cand.back() = id;
                        std::push_heap(cand.begin(), cand.end(), [&](int x, int y) { return col[x] > col[y]; });
                    }
                }
                if (tree != nullptr) {
                    auto offer = [&](int id) {
                        if (id != blank_id && id < n_token_cls && std::find(cand.begin(), cand.end(), id) == cand.end()) {
                            cand.push_back(id);
                        }
                    };
                    if (a.boost.node > 0) {
                        for (const auto & [id, next] : tree->nodes[static_cast<size_t>(a.boost.node)].children) {
                            offer(id);
                        }
                    }
                    for (const int id : root_tokens) {
                        offer(id);
                    }
                }
                for (const int id : cand) {
                    BeamExpansion e;
                    e.parent = i;
                    e.token  = id;
                    e.boost  = a.boost;
                    const float bonus = tree != nullptr ? boost_step(*tree, options.boost, e.boost, id) : 0.0f;
                    e.score           = a.score + (col[id] - lse) + bonus;
                    best_by_parent[static_cast<size_t>(i)] =
                        std::max(best_by_parent[static_cast<size_t>(i)], e.score);
                    expansions.push_back(e);
                }
            }

            std::sort(B.begin(), B.end(), [](const BeamHyp & x, const BeamHyp & y) { return x.score > y.score; });
            if (static_cast<int>(B.size()) > beam) {
                B.resize(static_cast<size_t>(beam));
            }
            const double floor = static_cast<int>(B.size()) >= beam ? B.back().score : k_neg_inf;
            expansions.erase(std::remove_if(expansions.begin(), expansions.end(),
                                            [&](const BeamExpansion & e) {
                                                return e.score <= floor ||
                                                       e.score < best_by_parent[static_cast<size_t>(e.parent)] -
                                                                     options.gamma;
                                            }),
                             expansions.end());
            if (expansions.empty()) {
                break;
            }
            std::sort(expansions.begin(), expansions.end(),
                      [](const BeamExpansion & x, const BeamExpansion & y) { return x.score > y.score; });
            if (static_cast<int>(expansions.size()) > beam) {
                expansions.resize(static_cast<size_t>(beam));
            }

            std::vector<int>               tokens;
            std::vector<const LstmState *> prev;
            std::vector<uint64_t>          keys;
            std::vector<int>               batch_of(expansions.size(), -1);
            if (pred_cache.size() + expansions.size() > k_pred_cache_max) {
                pred_cache.clear();
            }
            for (size_t k = 0; k < expansions.size(); ++k) {
                const BeamExpansion & e   = expansions[k];
                const uint64_t        key = extend_key(A[static_cast<size_t>(e.parent)].key, e.token);
                keys.push_back(key);
                if (pred_cache.count(key) != 0) {
                    ++n_cached;
                    continue;
                }
                batch_of[k] = static_cast<int>(tokens.size());
                tokens.push_back(e.token);
                prev.push_back(&A[static_cast<size_t>(e.parent)].state);
            }
            if (!tokens.empty() && !predict(tokens, prev, new_states)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            for (size_t k = 0; k < expansions.size(); ++k) {
                if (batch_of[k] >= 0) {
                    pred_cache[keys[k]] = new_states[static_cast<size_t>(batch_of[k])];
                }
            }
            std::vector<BeamHyp> next;
            next.reserve(expansions.size());
            for (size_t k = 0; k < expansions.size(); ++k) {
                const BeamExpansion & e      = expansions[k];
                const BeamHyp &       parent = A[static_cast<size_t>(e.parent)];
                const float *         col = logits.data() + static_cast<size_t>(e.parent) * static_cast<size_t>(joint_n);
                BeamNode              node;
                node.parent = parent.tail;
                node.id     = e.token;
                node.p      = token_confidence(col, n_token_cls, scratch_probs);
                node.step   = t;
                if (capture != nullptr && parent.len % k_cp == 0) {
                    node.checkpoint = std::make_unique<PredictorCheckpoint>(
                        PredictorCheckpoint{ parent.len, parent.last_token, parent.prev });
                }
                nodes.push_back(std::move(node));
                BeamHyp h;
                h.score      = e.score;
                h.boost      = e.boost;
                const PredResult & r = pred_cache.count(keys[k]) != 0 ? pred_cache.at(keys[k])
                                                                      : new_states[static_cast<size_t>(batch_of[k])];
                h.key        = keys[k];
                h.tail       = static_cast<int>(nodes.size()) - 1;
                h.len        = parent.len + 1;
                h.last_token = e.token;
                h.prev       = parent.state;
                h.state      = r.state;
                h.proj       = r.proj;
                next.push_back(std::move(h));
            }
            A.swap(next);
        }
        A.clear();
        if (trace) {
            std::string line;
            for (const BeamHyp & h : B) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), " [len %d %.2f]", h.len, h.score);
                line += buf;
            }
            log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "beam t=%d:%s", t, line.c_str());
        }

        // Drop the checkpoints no surviving hypothesis can reach.
        if (capture != nullptr && (t % 64 == 63)) {
            reached.assign(nodes.size(), 0);
            for (const BeamHyp & h : B) {
                for (int i = h.tail; i >= 0 && reached[static_cast<size_t>(i)] == 0; i = nodes[static_cast<size_t>(i)].parent) {
                    reached[static_cast<size_t>(i)] = 1;
                }
            }
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (reached[i] == 0) {
                    nodes[i].checkpoint.reset();
                }
            }
        }
    }

    if (B.empty()) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    // The final pick is by score per token (NeMo's score_norm): over a long
    // silence a finished sentence keeps paying a little for every frame it
    // might have said more, while the empty hypothesis pays nothing, so
    // raw scores end up preferring "nothing was said".
    const BeamHyp * best       = nullptr;
    double          best_score = k_neg_inf;
    for (const BeamHyp & h : B) {
        const double final_score =
            (h.score + (tree != nullptr ? boost_finish(*tree, h.boost) : 0.0)) / static_cast<double>(h.len + 1);
        if (best == nullptr || final_score > best_score) {
            best       = &h;
            best_score = final_score;
        }
    }
    std::vector<int> path;
    for (int i = best->tail; i >= 0; i = nodes[static_cast<size_t>(i)].parent) {
        path.push_back(i);
    }
    std::reverse(path.begin(), path.end());
    const size_t first_out = out_tokens.size();
    for (const int i : path) {
        BeamNode & node = nodes[static_cast<size_t>(i)];
        if (capture != nullptr && node.checkpoint != nullptr) {
            PredictorCheckpoint cp = std::move(*node.checkpoint);
            cp.token += static_cast<int>(first_out);
            capture->checkpoints.push_back(std::move(cp));
        }
        TdtToken tok;
        tok.id              = node.id;
        tok.p               = node.p;
        tok.step_at_emit    = node.step;
        tok.duration_frames = 1;
        out_tokens.push_back(tok);
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder (rnnt beam %d): %d tokens, T_enc=%d  %d joint (%.1f ms)  %d pred (%.1f ms)  %d cached  "
            "%zu nodes  total=%.1f ms",
            beam, static_cast<int>(path.size()), T_enc, n_joint, t_joint_us / 1000.0, n_pred, t_pred_us / 1000.0,
            n_cached, nodes.size(), (ggml_time_us() - t_start) / 1000.0);

    if (capture != nullptr) {
        capture->enc_proj             = std::move(enc_proj_all);
        capture->enc_proj_first_frame = 0;
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::parakeet
