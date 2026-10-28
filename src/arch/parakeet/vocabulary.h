// arch/parakeet/vocabulary.h - Custom vocabulary rescoring for the
// transducer heads (RNN-T and TDT).
//
// Parakeet takes transcribe_run_params::vocabulary (and the optional
// transcribe_parakeet_vocabulary_ext) as a post-decode correction, not a
// decode-time bias. Biasing the greedy argmax toward a term's first piece
// either never fires (the margin gate holds against a confident "a") or
// fires on ordinary speech; both were measured. Instead, after the greedy
// decode:
//
//   1. find_vocabulary_candidates picks word spans of the transcript whose
//      letters or consonant skeleton are close to a term, or that equal one
//      of its sounds-like spellings ("a bat war" for "Abattoir");
//   2. rescore_vocabulary (decoder.cpp) scores the term and the span by the
//      transducer's own forward algorithm over the same encoder frames, from
//      the same predictor state;
//   3. the term replaces the span when its log-probability plus `boost`
//      per term piece is at least the span's.
//
// The audio decides; the boost only says how much worse than the heard
// words a term may score and still win. It is per piece because the
// predictor's own language model charges a rare word for every piece it
// spells ("▁A b att oir" against "▁avatar"). Bounded: at most k_vocab_max_terms
// terms of at most k_vocab_max_term_tokens pieces, k_vocab_max_candidates
// spans per decode, and windows of at most k_vocab_max_window_frames.

#pragma once

#include "decoder.h"
#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe {
class Tokenizer;
}

namespace transcribe::parakeet {

constexpr int   k_vocab_max_terms         = 64;
constexpr int   k_vocab_max_spellings     = 256;
constexpr int   k_vocab_max_term_bytes    = 128;
constexpr int   k_vocab_max_term_tokens   = 12;
constexpr int   k_vocab_max_span_words    = 4;
constexpr int   k_vocab_max_candidates    = 16;
constexpr int   k_vocab_max_window_frames = 96;
constexpr int   k_vocab_window_lead       = 4;  // frames before the span's first token
constexpr int   k_vocab_window_tail       = 6;  // frames after its last
// Shortest term (in letters) matched by sound alone; shorter terms are
// matched only through their sounds-like spellings.
constexpr int   k_vocab_min_fuzzy_letters = 5;
constexpr float k_vocab_default_boost = 2.5f;  // nats per term piece; see Hex docs/research/vocabulary-boost-results.md

struct VocabularyTerm {
    std::string          text;
    std::vector<int32_t> ids;       // SentencePiece pieces of " " + text
    std::string          letters;   // vocabulary_letters(text)
    std::string          skeleton;  // vocabulary_skeleton(letters)
};

struct VocabularySpelling {
    std::string letters;
    int         term = 0;
};

struct VocabularyBias {
    float                           boost = 0.0f;
    std::vector<VocabularyTerm>     terms;
    std::vector<VocabularySpelling> spellings;

    bool active() const { return boost > 0.0f && !terms.empty(); }
};

// Compile params->vocabulary (+ the PKVB extension, if any) against the
// model's tokenizer. Terms that do not encode, encode to an unknown piece,
// or exceed the bounds are skipped with a warning. An inactive result
// (no terms, or boost 0) means no rescoring. Fails only on a malformed
// extension.
transcribe_status compile_vocabulary(const Tokenizer & tok, const transcribe_run_params * params, VocabularyBias & out);

// One word of the transcript: raw token range and the range without its
// trailing punctuation pieces. `text` is the core's decoded text.
struct VocabularyWord {
    int         first    = 0;
    int         core_end = 0;
    int         end      = 0;
    std::string text;
};

std::vector<VocabularyWord> vocabulary_words(const Tokenizer & tok, const std::vector<TdtToken> & tokens);

struct VocabularyCandidate {
    int   first = 0;  // raw token index of the span's first token
    int   end   = 0;  // one past the span's last core token
    int   term  = 0;
    float rank  = 0;  // lower is closer; 0 for a sounds-like spelling
};

// Spans of `words` worth rescoring, closest first, at most
// k_vocab_max_candidates. Pure (no model access).
std::vector<VocabularyCandidate> find_vocabulary_candidates(const std::vector<VocabularyWord> & words,
                                                            const VocabularyBias &              bias);

// Lowercase ASCII letters and digits of `text`; other ASCII is dropped and
// non-ASCII bytes are kept.
std::string vocabulary_letters(const std::string & text);

// A rough consonant skeleton of `letters` for sound-alike matching: an
// initial vowel becomes '*', later vowels and h/w/y drop, voiced
// consonants map to voiceless ones (b/v→p, d→t, g→k, z→s), c/q→k,
// x→ks, ph→f, and repeats collapse. "abattoir", "a bat war" and
// "avatar" all give "*ptr".
std::string vocabulary_skeleton(const std::string & letters);

int vocabulary_edit_distance(const std::string & a, const std::string & b);

}  // namespace transcribe::parakeet
