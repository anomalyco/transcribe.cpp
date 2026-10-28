// Unit test for parakeet vocabulary matching: the consonant skeleton, edit
// distance, and which transcript spans become rescoring candidates. The
// acoustic scoring needs a model and is covered by the real-model eval.

#include "arch/parakeet/vocabulary.h"

#include <cstdio>
#include <string>
#include <vector>

namespace pv = transcribe::parakeet;

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// Words with one token each, so a span's token range equals its word range.
std::vector<pv::VocabularyWord> words(const std::vector<std::string> & texts) {
    std::vector<pv::VocabularyWord> out;
    for (size_t i = 0; i < texts.size(); ++i) {
        pv::VocabularyWord w;
        w.first    = static_cast<int>(i);
        w.core_end = static_cast<int>(i) + 1;
        w.end      = static_cast<int>(i) + 1;
        w.text     = texts[i];
        out.push_back(w);
    }
    return out;
}

pv::VocabularyBias bias(const std::vector<std::string> & terms) {
    pv::VocabularyBias b;
    b.boost = 4.0f;
    for (const std::string & t : terms) {
        pv::VocabularyTerm term;
        term.text     = t;
        term.ids      = { 1 };
        term.letters  = pv::vocabulary_letters(t);
        term.skeleton = pv::vocabulary_skeleton(term.letters);
        b.terms.push_back(term);
    }
    return b;
}

bool has(const std::vector<pv::VocabularyCandidate> & cs, int first, int end, int term) {
    for (const auto & c : cs) {
        if (c.first == first && c.end == end && c.term == term) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    check(pv::vocabulary_letters("Open-Code!") == "opencode", "letters drop case and punctuation");
    check(pv::vocabulary_skeleton("abattoir") == "*ptr", "abattoir skeleton");
    check(pv::vocabulary_skeleton("abatwar") == "*ptr", "a bat war skeleton");
    check(pv::vocabulary_skeleton("avatar") == "*ptr", "avatar skeleton");
    check(pv::vocabulary_skeleton("scrawl") == "skrl", "scrawl skeleton");
    check(pv::vocabulary_edit_distance("kitten", "sitting") == 3, "edit distance");
    check(pv::vocabulary_edit_distance("", "abc") == 3, "edit distance from empty");

    {
        const auto b  = bias({ "Abattoir" });
        const auto cs = pv::find_vocabulary_candidates(words({ "put", "it", "in", "avatar", "please" }), b);
        check(has(cs, 3, 4, 0), "avatar is a candidate for Abattoir");
        const auto split = pv::find_vocabulary_candidates(words({ "the", "a", "bat", "war", "repo" }), b);
        check(has(split, 1, 4, 0), "a bat war is a candidate for Abattoir");
        const auto none = pv::find_vocabulary_candidates(words({ "we", "went", "to", "the", "store" }), b);
        check(none.empty(), "ordinary words are not candidates");
        const auto same = pv::find_vocabulary_candidates(words({ "open", "Abattoir" }), b);
        check(same.empty(), "a span already written as the term is skipped");
    }
    {
        // Short terms match only through sounds-like spellings.
        auto b = bias({ "Kit", "Hex" });
        check(pv::find_vocabulary_candidates(words({ "a", "kid", "said", "hacks" }), b).empty(),
              "short terms do not fuzzy match");
        b.spellings.push_back({ "kid", 0 });
        const auto cs = pv::find_vocabulary_candidates(words({ "a", "kid", "said" }), b);
        check(has(cs, 1, 2, 0) && cs.front().rank == 0.0f, "a sounds-like spelling matches first");
    }
    {
        // Case alone is not rescored; a split is.
        const auto b = bias({ "OpenCode", "Effect" });
        check(pv::find_vocabulary_candidates(words({ "an", "effect", "here" }), b).empty(), "case-only difference");
        check(has(pv::find_vocabulary_candidates(words({ "use", "open", "code", "now" }), b), 1, 3, 0),
              "open code is a candidate for OpenCode");
    }
    {
        // A boost of zero turns everything off.
        auto b  = bias({ "Abattoir" });
        b.boost = 0.0f;
        check(pv::find_vocabulary_candidates(words({ "avatar" }), b).empty(), "inactive bias has no candidates");
    }
    {
        // Bounded.
        std::vector<std::string> many;
        for (int i = 0; i < 200; ++i) {
            many.push_back("avatar");
        }
        const auto cs = pv::find_vocabulary_candidates(words(many), bias({ "Abattoir" }));
        check(static_cast<int>(cs.size()) <= pv::k_vocab_max_candidates, "candidates are capped");
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("parakeet vocabulary unit: ok\n");
    return 0;
}
