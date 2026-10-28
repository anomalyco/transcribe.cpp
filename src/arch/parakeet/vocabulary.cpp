// arch/parakeet/vocabulary.cpp - term compilation and candidate spans for
// vocabulary rescoring. See vocabulary.h; the acoustic scoring lives in
// decoder.cpp beside the predictor and joint graphs it reuses.

#include "vocabulary.h"

#include "transcribe-log.h"
#include "transcribe-tokenizer.h"
#include "transcribe/parakeet.h"

#include <algorithm>
#include <cstring>

namespace transcribe::parakeet {

namespace {

constexpr const char k_sp_marker[]   = "\xE2\x96\x81";
constexpr size_t     k_sp_marker_len = 3;

bool has_alnum(const std::string & text) {
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        if (c >= 0x80 || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            return true;
        }
    }
    return false;
}

bool is_special_piece(const Tokenizer & tok, int id) {
    if (tok.is_strippable_special(id)) {
        return true;
    }
    const std::string & piece = tok.token(id);
    return piece.size() >= 2 && piece.front() == '<' && piece.back() == '>';
}

std::string trim(const std::string & text) {
    size_t a = 0;
    size_t b = text.size();
    while (a < b && (text[a] == ' ' || text[a] == '\t')) {
        ++a;
    }
    while (b > a && (text[b - 1] == ' ' || text[b - 1] == '\t')) {
        --b;
    }
    return text.substr(a, b - a);
}

bool add_term(const Tokenizer & tok, const std::string & raw, VocabularyBias & out) {
    const std::string text = trim(raw);
    if (text.empty()) {
        return false;
    }
    if (text.size() > static_cast<size_t>(k_vocab_max_term_bytes)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet vocabulary: skipping a term longer than %d bytes",
                k_vocab_max_term_bytes);
        return false;
    }
    VocabularyTerm term;
    term.text = text;
    if (tok.encode_sentencepiece_bpe(text, term.ids, 0, -1, true) != TRANSCRIBE_OK || term.ids.empty()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet vocabulary: could not encode \"%s\"; skipping it", text.c_str());
        return false;
    }
    for (const int32_t id : term.ids) {
        if (id < 0 || is_special_piece(tok, id)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet vocabulary: \"%s\" has characters the model cannot write",
                    text.c_str());
            return false;
        }
    }
    if (static_cast<int>(term.ids.size()) > k_vocab_max_term_tokens) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet vocabulary: \"%s\" is more than %d pieces; skipping it",
                text.c_str(), k_vocab_max_term_tokens);
        return false;
    }
    term.letters  = vocabulary_letters(text);
    term.skeleton = vocabulary_skeleton(term.letters);
    if (term.letters.empty()) {
        return false;
    }
    out.terms.push_back(std::move(term));
    return true;
}

}  // namespace

std::string vocabulary_letters(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        if (c >= 'A' && c <= 'Z') {
            out.push_back(static_cast<char>(c - 'A' + 'a'));
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c >= 0x80) {
            out.push_back(ch);
        }
    }
    return out;
}

std::string vocabulary_skeleton(const std::string & letters) {
    auto is_vowel = [](char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    };
    std::string out;
    out.reserve(letters.size());
    auto push = [&](char c) {
        if (out.empty() || out.back() != c) {
            out.push_back(c);
        }
    };
    for (size_t i = 0; i < letters.size(); ++i) {
        const char c = letters[i];
        if (i == 0 && (is_vowel(c) || c == 'y')) {
            out.push_back('*');
            continue;
        }
        switch (c) {
            case 'a':
            case 'e':
            case 'i':
            case 'o':
            case 'u':
            case 'y':
            case 'w':
                break;
            case 'h':
                if (i == 0) {
                    push('h');
                }
                break;
            case 'b':
            case 'v':
                push('p');
                break;
            case 'd':
                push('t');
                break;
            case 'g':
            case 'c':
            case 'q':
                push('k');
                break;
            case 'z':
                push('s');
                break;
            case 'x':
                push('k');
                push('s');
                break;
            case 'p':
                if (i + 1 < letters.size() && letters[i + 1] == 'h') {
                    push('f');
                    ++i;
                } else {
                    push('p');
                }
                break;
            default:
                push(c);
                break;
        }
    }
    return out;
}

int vocabulary_edit_distance(const std::string & a, const std::string & b) {
    std::vector<int> prev(b.size() + 1);
    std::vector<int> cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) {
        prev[j] = static_cast<int>(j);
    }
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= b.size(); ++j) {
            const int sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            cur[j]        = std::min({ prev[j] + 1, cur[j - 1] + 1, sub });
        }
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

transcribe_status compile_vocabulary(const Tokenizer &             tok,
                                     const transcribe_run_params * params,
                                     VocabularyBias &              out) {
    out = VocabularyBias{};
    if (params == nullptr) {
        return TRANSCRIBE_OK;
    }
    const transcribe_parakeet_vocabulary_ext * ext = nullptr;
    if (params->family != nullptr) {
        if (const transcribe_status st = transcribe_ext_check(params->family, TRANSCRIBE_EXT_KIND_PARAKEET_VOCABULARY,
                                                              sizeof(struct transcribe_parakeet_vocabulary_ext));
            st != TRANSCRIBE_OK) {
            return st;
        }
        ext = reinterpret_cast<const transcribe_parakeet_vocabulary_ext *>(params->family);
        if (ext->n_sounds_like < 0 ||
            (ext->n_sounds_like > 0 && (ext->sounds_like == nullptr || ext->sounds_like_term == nullptr))) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    out.boost = (ext != nullptr && ext->boost >= 0.0f) ? ext->boost : k_vocab_default_boost;
    if (params->vocabulary == nullptr || params->n_vocabulary <= 0 || out.boost <= 0.0f) {
        return TRANSCRIBE_OK;
    }

    // Original index -> compiled index, for the spellings.
    std::vector<int> compiled(static_cast<size_t>(params->n_vocabulary), -1);
    for (int i = 0; i < params->n_vocabulary; ++i) {
        if (static_cast<int>(out.terms.size()) >= k_vocab_max_terms) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet vocabulary: using the first %d terms", k_vocab_max_terms);
            break;
        }
        const char * term = params->vocabulary[i];
        if (term != nullptr && add_term(tok, term, out)) {
            compiled[static_cast<size_t>(i)] = static_cast<int>(out.terms.size()) - 1;
        }
    }
    if (ext != nullptr) {
        for (int i = 0; i < ext->n_sounds_like; ++i) {
            if (static_cast<int>(out.spellings.size()) >= k_vocab_max_spellings) {
                break;
            }
            const int32_t owner = ext->sounds_like_term[i];
            if (owner < 0 || owner >= params->n_vocabulary || compiled[static_cast<size_t>(owner)] < 0 ||
                ext->sounds_like[i] == nullptr) {
                continue;
            }
            const std::string raw = ext->sounds_like[i];
            if (raw.size() > static_cast<size_t>(k_vocab_max_term_bytes)) {
                continue;
            }
            VocabularySpelling spelling;
            spelling.letters = vocabulary_letters(raw);
            spelling.term    = compiled[static_cast<size_t>(owner)];
            if (!spelling.letters.empty()) {
                out.spellings.push_back(std::move(spelling));
            }
        }
    }
    return TRANSCRIBE_OK;
}

std::vector<VocabularyWord> vocabulary_words(const Tokenizer & tok, const std::vector<TdtToken> & tokens) {
    std::vector<VocabularyWord> words;
    VocabularyWord              cur;
    bool                        open  = false;
    auto                        close = [&](int end) {
        if (!open) {
            return;
        }
        cur.end = end;
        if (cur.core_end > cur.first) {
            std::vector<int> ids;
            for (int i = cur.first; i < cur.core_end; ++i) {
                ids.push_back(tokens[static_cast<size_t>(i)].id);
            }
            cur.text = trim(tok.decode(ids.data(), static_cast<int>(ids.size())));
        }
        words.push_back(cur);
        open = false;
    };
    for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
        const int id = tokens[static_cast<size_t>(i)].id;
        if (is_special_piece(tok, id)) {
            close(i);
            continue;
        }
        const std::string & piece = tok.token(id);
        const bool          starts =
            !open || (piece.size() >= k_sp_marker_len && std::memcmp(piece.data(), k_sp_marker, k_sp_marker_len) == 0);
        if (starts) {
            close(i);
            cur          = VocabularyWord{};
            cur.first    = i;
            cur.core_end = i;
            open         = true;
        }
        // Trailing punctuation stays outside the core; a letter after
        // punctuation inside one word ("don't") pulls the core forward.
        if (has_alnum(tok.decode(&id, 1))) {
            cur.core_end = i + 1;
        }
    }
    close(static_cast<int>(tokens.size()));
    return words;
}

std::vector<VocabularyCandidate> find_vocabulary_candidates(const std::vector<VocabularyWord> & words,
                                                            const VocabularyBias &              bias) {
    std::vector<VocabularyCandidate> out;
    if (!bias.active()) {
        return out;
    }
    const int n_words = static_cast<int>(words.size());

    // Whether words [a, a + n) form a span: every word has a core and only
    // the last may carry trailing punctuation.
    auto is_span = [&](int a, int n) {
        for (int k = a; k < a + n; ++k) {
            const VocabularyWord & w = words[static_cast<size_t>(k)];
            if (w.core_end <= w.first || (k < a + n - 1 && w.core_end != w.end)) {
                return false;
            }
        }
        return true;
    };
    auto span_letters = [&](int a, int n) {
        std::string letters;
        for (int k = a; k < a + n; ++k) {
            letters += vocabulary_letters(words[static_cast<size_t>(k)].text);
        }
        return letters;
    };
    // How close the span's letters are to term t (lower is closer), or -1.
    auto rank_of = [&](const std::string & letters, int t) -> float {
        const VocabularyTerm & term = bias.terms[static_cast<size_t>(t)];
        for (const VocabularySpelling & s : bias.spellings) {
            if (s.term == t && s.letters == letters) {
                return 0.0f;
            }
        }
        if (letters == term.letters) {
            return 0.01f;
        }
        const int L = static_cast<int>(term.letters.size());
        const int S = static_cast<int>(letters.size());
        if (L < k_vocab_min_fuzzy_letters || S * 2 < L || S > L * 2) {
            return -1.0f;
        }
        const int d_letters = vocabulary_edit_distance(letters, term.letters);
        const int d_skel    = vocabulary_edit_distance(vocabulary_skeleton(letters), term.skeleton);
        const int max_skel  = term.skeleton.size() >= 4 ? 1 : 0;
        if (d_letters > std::max(1, L / 3) && d_skel > max_skel) {
            return -1.0f;
        }
        return static_cast<float>(d_letters) / static_cast<float>(L) + 0.5f * static_cast<float>(d_skel);
    };

    for (int a = 0; a < n_words; ++a) {
        for (int n = 1; n <= k_vocab_max_span_words && a + n <= n_words && is_span(a, n); ++n) {
            const std::string letters = span_letters(a, n);
            std::string       text;
            for (int k = a; k < a + n; ++k) {
                text += (k > a ? " " : "") + words[static_cast<size_t>(k)].text;
            }
            for (int t = 0; t < static_cast<int>(bias.terms.size()); ++t) {
                const VocabularyTerm & term = bias.terms[static_cast<size_t>(t)];
                // Already written as the term, or differing only in case
                // ("effect" vs "Effect"), which is not an acoustic question;
                // a split or join is ("open code").
                if (text == term.text || (n == 1 && letters == term.letters)) {
                    continue;
                }
                // An inflection of the term is the term, already written
                // ("Aiden's", "Effects").
                if (letters.size() > term.letters.size() &&
                    letters.compare(0, term.letters.size(), term.letters) == 0) {
                    const std::string rest = letters.substr(term.letters.size());
                    if (rest == "s" || rest == "es" || rest == "ed" || rest == "d" || rest == "ing" || rest == "er" ||
                        rest == "ers") {
                        continue;
                    }
                }
                const float rank = rank_of(letters, t);
                if (rank < 0.0f) {
                    continue;
                }
                // A longer span loses to the shorter one inside it that is at
                // least as close ("an effect" to "effect", "use open code" to
                // "open code"), so a term never swallows a neighbouring word.
                if (n > 1) {
                    const float inner_tail = rank_of(span_letters(a + 1, n - 1), t);
                    const float inner_head = rank_of(span_letters(a, n - 1), t);
                    if ((inner_tail >= 0.0f && inner_tail <= rank) || (inner_head >= 0.0f && inner_head <= rank)) {
                        continue;
                    }
                }
                VocabularyCandidate c;
                c.first = words[static_cast<size_t>(a)].first;
                c.end   = words[static_cast<size_t>(a + n - 1)].core_end;
                c.term  = t;
                c.rank  = rank;
                out.push_back(c);
            }
        }
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const VocabularyCandidate & x, const VocabularyCandidate & y) { return x.rank < y.rank; });
    if (static_cast<int>(out.size()) > k_vocab_max_candidates) {
        out.resize(static_cast<size_t>(k_vocab_max_candidates));
    }
    return out;
}

}  // namespace transcribe::parakeet
