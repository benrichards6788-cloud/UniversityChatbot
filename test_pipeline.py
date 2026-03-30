"""
Coverage
--------
Unit tests   — sem_chunker.py pure functions (no model, no file I/O)
Integration  — sem_chunker.py with real MiniLM embeddings (group_semantically)
Integration  — retrieve_policies() against a synthetic in-memory FAISS index
System       — chunk_file() on a synthetic .txt fixture

Run with:
    pytest test_pipeline.py -v
"""

import json
import math
import re
import inspect
from pathlib import Path

import numpy as np
import pytest
import faiss

from sem_chunker import (
    approx_tokens,
    clean_text,
    split_sections,
    merge_small_sections,
    split_sentences,
    pack_runs_to_chunks,
    chunk_body,
    prettify_title,
    _normalize_heading,
    _looks_plausible_heading,
    group_semantically,
    chunk_file,
    DEFAULT_TARGET_TOKENS,
    DEFAULT_SIM_THRESHOLD,
)


# Adapter: handle source_file arg present in local version but not in
# the reference version — detected at import time so tests stay clean.
_PACK_HAS_SOURCE_FILE = "source_file" in inspect.signature(pack_runs_to_chunks).parameters

def _call_pack(runs, doc_title, section_title, **kwargs):
    if _PACK_HAS_SOURCE_FILE:
        kwargs.setdefault(
            "source_file",
            doc_title.lower().replace(" ", "_") + ".pdf"
        )
    return pack_runs_to_chunks(runs, doc_title, section_title, **kwargs)



# UNIT TESTS — approx_tokens

class TestApproxTokens:
    def test_empty_string_returns_one(self):
        assert approx_tokens("") == 1

    def test_short_text(self):
        assert approx_tokens("Hello") == 2

    def test_known_length(self):
        assert approx_tokens("a" * 400) == 100

    def test_long_text_scales_linearly(self):
        assert approx_tokens("x" * 1600) == approx_tokens("x" * 800) * 2

    def test_non_ascii_counted_by_chars(self):
        assert approx_tokens("é" * 40) == 10


# UNIT TESTS — clean_text

class TestCleanText:
    def test_removes_carriage_returns(self):
        assert "\r" not in clean_text("line1\r\nline2")

    def test_dehyphenates_line_breaks(self):
        assert "information" in clean_text("infor-\nmation")

    def test_replaces_bullet_characters(self):
        assert "- First item" in clean_text("\n• First item")

    def test_flattens_single_newlines_within_paragraph(self):
        assert "Hello World" in clean_text("Hello\nWorld")

    def test_collapses_multiple_newlines(self):
        assert "\n\n\n" not in clean_text("Para one\n\n\n\nPara two")

    def test_removes_dotted_leader_lines(self):
        assert "........" not in clean_text("Introduction ........ 3")

    def test_removes_standalone_page_numbers(self):
        lines = [l.strip() for l in clean_text("Some text\n\n42\n\nMore text").split("\n") if l.strip()]
        assert "42" not in lines

    def test_removes_strathclyde_boilerplate(self):
        assert "place of useful learning" not in clean_text("The place of useful learning").lower()

    def test_removes_charitable_body_boilerplate(self):
        assert "SC015263" not in clean_text("A charitable body SC015263")

    def test_normalises_whitespace(self):
        assert "  " not in clean_text("too   many     spaces")

    def test_preserves_content(self):
        result = clean_text("Students must submit appeals within 14 days.")
        assert "appeals" in result and "14 days" in result

    def test_empty_input_returns_empty(self):
        assert clean_text("") == ""

    def test_removes_file_merge_markers(self):
        result = clean_text("===== FILE: test.txt =====\nSome content")
        assert "===== FILE:" not in result
        assert "Some content" in result

# UNIT TESTS — heading helpers

class TestHeadingHelpers:
    def test_normalize_strips_trailing_colon(self):
        assert _normalize_heading("INTRODUCTION:") == "INTRODUCTION"

    def test_normalize_collapses_whitespace(self):
        assert _normalize_heading("SECTION   ONE") == "SECTION ONE"

    def test_normalize_replaces_newline(self):
        assert "\n" not in _normalize_heading("HEADING\nCONTINUED")

    def test_plausible_all_caps_short(self):
        assert _looks_plausible_heading("ACADEMIC APPEALS") is True

    def test_plausible_numbered_heading(self):
        assert _looks_plausible_heading("1.1 LATE SUBMISSION") is True

    def test_rejects_empty_string(self):
        assert _looks_plausible_heading("") is False

    def test_rejects_long_heading(self):
        assert _looks_plausible_heading("A" * 130) is False

    def test_rejects_sentence_ending_in_period(self):
        assert _looks_plausible_heading("This is a sentence that goes on and on and on.") is False

    def test_rejects_mostly_lowercase(self):
        assert _looks_plausible_heading("this looks like body text") is False

    def test_accepts_title_ending_in_colon(self):
        assert _looks_plausible_heading("Late Submission Policy:") is True


# UNIT TESTS — split_sections


class TestSplitSections:
    def test_returns_document_fallback_on_no_headings(self):
        sections = split_sections("Just some plain body text.")
        assert sections[0]["title"] == "Document"

    def test_detects_all_caps_heading(self):
        sections = split_sections("INTRODUCTION\n\nThis is the intro text.")
        assert any("INTRODUCTION" in s["title"] for s in sections)

    def test_detects_numbered_heading(self):
        sections = split_sections("1.1 APPEALS PROCESS\n\nStudents may appeal within 14 days.")
        assert any("APPEALS" in s["title"] for s in sections)

    def test_filters_contents_section(self):
        text = "CONTENTS\n\nIntroduction....1\n\nINTRODUCTION\n\nActual content."
        assert not any("CONTENTS" in s["title"] for s in split_sections(text))

    def test_body_attached_to_correct_heading(self):
        sections = split_sections("APPEALS\n\nStudents may appeal.")
        appeal_secs = [s for s in sections if "APPEALS" in s["title"]]
        assert appeal_secs and "appeal" in appeal_secs[0]["body"].lower()

    def test_handles_empty_input(self):
        assert isinstance(split_sections(""), list)

    def test_multiple_sections_in_order(self):
        text = "SECTION ONE\n\nContent one.\n\nSECTION TWO\n\nContent two."
        bodies = [s["body"] for s in split_sections(text)]
        assert any("one" in b for b in bodies)
        assert any("two" in b for b in bodies)


# UNIT TESTS — merge_small_sections

class TestMergeSmallSections:
    def _sec(self, title, words):
        return {"title": title, "body": "word " * words}

    def test_empty_input_returns_empty(self):
        assert merge_small_sections([]) == []

    def test_small_section_merged_forward(self):
        result = merge_small_sections(
            [self._sec("Small", 10), self._sec("Large", 200)],
            min_tokens=120, join_with_next=True
        )
        assert len(result) == 1

    def test_large_sections_not_merged(self):
        result = merge_small_sections(
            [self._sec("First", 200), self._sec("Second", 200)],
            min_tokens=120
        )
        assert len(result) == 2

    def test_preserves_body_content_after_merge(self):
        sections = [
            {"title": "Small", "body": "small content"},
            {"title": "Large", "body": "large content " * 50},
        ]
        result = merge_small_sections(sections, min_tokens=120, join_with_next=True)
        assert "small content" in result[0]["body"]
        assert "large content" in result[0]["body"]

    def test_empty_body_sections_skipped(self):
        sections = [{"title": "Empty", "body": ""}, {"title": "Full", "body": "real " * 50}]
        result = merge_small_sections(sections, min_tokens=120)
        assert all(s["body"].strip() for s in result)



# UNIT TESTS — split_sentences

class TestSplitSentences:
    def test_splits_on_period_capital(self):
        assert len(split_sentences("First sentence. Second sentence.")) == 2

    def test_splits_on_exclamation(self):
        assert len(split_sentences("Warning! Please read carefully.")) == 2

    def test_splits_on_question_mark(self):
        assert len(split_sentences("What is the deadline? Submit by Friday.")) == 2

    def test_empty_input_returns_empty_list(self):
        assert split_sentences("") == []

    def test_strips_whitespace_from_sentences(self):
        sents = split_sentences("  First.   Second.  ")
        assert all(s == s.strip() for s in sents)

    def test_single_sentence_returns_list_of_one(self):
        assert len(split_sentences("This is a single sentence")) == 1

    def test_content_preserved(self):
        sents = split_sentences("Appeals within 14 days. Late appeals not accepted.")
        full = " ".join(sents)
        assert "14 days" in full and "Late appeals" in full



# UNIT TESTS — chunk_body

class TestChunkBody:
    def test_strips_breadcrumb_header(self):
        assert chunk_body("[Policy: P] [Section: S]\n\nBody.") == "Body."

    def test_returns_full_text_if_no_double_newline(self):
        assert chunk_body("no breadcrumb") == "no breadcrumb"

    def test_only_strips_first_split(self):
        result = chunk_body("header\n\nfirst\n\nsecond")
        assert "second" in result



# UNIT TESTS — prettify_title

class TestPrettifyTitle:
    def test_replaces_underscores(self):
        assert "_" not in prettify_title("academic_appeals_policy")

    def test_replaces_hyphens(self):
        assert "-" not in prettify_title("late-submission-policy")

    def test_title_cases_mixed(self):
        assert prettify_title("academic_appeals_policy")[0].isupper()

    def test_preserves_all_caps(self):
        assert prettify_title("GDPR") == "GDPR"

    def test_collapses_extra_spaces(self):
        assert "  " not in prettify_title("academic__policy")



# UNIT TESTS — pack_runs_to_chunks


class TestPackRunsToChunks:
    def _runs(self, n, words=50):
        return [["word " * words] for _ in range(n)]

    def test_single_run_produces_one_chunk(self):
        assert len(_call_pack([["Short sentence."]], "P", "S", target_tokens=900)) == 1

    def test_runs_within_budget_stay_as_one_chunk(self):
        assert len(_call_pack(self._runs(3, 50), "P", "S", target_tokens=900)) == 1

    def test_large_runs_split_into_multiple_chunks(self):
        assert len(_call_pack(self._runs(5, 300), "P", "S", target_tokens=900)) >= 2

    def test_chunk_contains_breadcrumb(self):
        chunks = _call_pack([["Content."]], "Appeals Policy", "Section 2", target_tokens=900)
        assert all("Appeals Policy" in c["text"] and "Section 2" in c["text"] for c in chunks)

    def test_chunk_has_required_keys(self):
        chunks = _call_pack([["Some text."]], "T", "S", target_tokens=900)
        for c in chunks:
            assert {"text", "doc_title", "section", "tokens_est"}.issubset(c.keys())

    def test_tokens_est_is_positive_int(self):
        chunks = _call_pack([["Some text."]], "T", "S", target_tokens=900)
        assert all(isinstance(c["tokens_est"], int) and c["tokens_est"] > 0 for c in chunks)

    def test_empty_runs_returns_empty(self):
        assert _call_pack([], "T", "S") == []

    def test_token_estimate_within_bounds(self):
        chunks = _call_pack(self._runs(10, 100), "T", "S", target_tokens=500)
        # Each chunk should be non-empty and no single chunk should contain
        # all 10 runs (that would mean the budget is being ignored entirely)
        assert len(chunks) >= 2
        assert all(c["tokens_est"] > 0 for c in chunks)



# INTEGRATION TESTS — group_semantically


class TestGroupSemantically:
    def test_empty_list_returns_empty(self):
        assert group_semantically([]) == []

    def test_single_sentence_returns_one_run(self):
        result = group_semantically(["A single sentence."])
        assert result == [["A single sentence."]]

    def test_no_sentences_lost(self):
        sentences = ["First.", "Second.", "Third.", "Fourth."]
        all_sents = [s for run in group_semantically(sentences) for s in run]
        assert set(all_sents) == set(sentences)

    def test_max_run_cap_respected(self):
        sentences = [f"Sentence {i} about policy." for i in range(10)]
        for run in group_semantically(sentences, max_run=2):
            assert len(run) <= 2

    def test_returns_list_of_lists(self):
        runs = group_semantically(["A sentence.", "Another sentence."])
        assert isinstance(runs, list) and all(isinstance(r, list) for r in runs)

    def test_similar_sentences_grouped_at_low_threshold(self):
        sentences = [
            "Students may submit an academic appeal.",
            "Appeals must be submitted within 14 working days.",
            "The appeals process is managed by the Student Experience team.",
        ]
        all_sents = [s for run in group_semantically(sentences, sim_threshold=0.2) for s in run]
        assert len(all_sents) == 3



# INTEGRATION TESTS — chunk_file


class TestChunkFile:
    FIXTURE = """ACADEMIC APPEALS POLICY

INTRODUCTION

Students at the University have the right to appeal decisions made by the Board of Examiners. Appeals must be submitted within 14 working days of the notification of the decision. The appeals process is managed by the Student Experience team and all submissions are handled confidentially.

GROUNDS FOR APPEAL

Appeals may only be submitted on the following grounds. First, administrative error in the recording or calculation of marks. Second, personal circumstances that affected performance and were not previously disclosed to the Board. Third, procedural irregularities in the conduct of examinations or assessments that may have materially affected the student's performance on the day.

SUBMISSION PROCESS

Appeals must be submitted using the official appeals form available on MyPlace. Supporting documentation must be included with all submissions. Incomplete submissions will not be processed and will be returned to the student. Late submissions will not be accepted without prior written approval from the Head of Student Experience. Students are advised to retain copies of all submitted documentation for their own records.

OUTCOMES

The outcome of an appeal will be communicated to the student in writing within 28 working days. Where an appeal is upheld, the Board of Examiners will review the relevant decision at its next scheduled meeting. Students who remain dissatisfied may escalate to the University Appeals Committee within 14 working days of notification of the outcome.
"""

    def test_produces_at_least_one_chunk(self, tmp_path):
        p = tmp_path / "academic_appeals_policy.txt"
        p.write_text(self.FIXTURE, encoding="utf-8")
        assert len(chunk_file(p)) >= 1

    def test_all_chunks_have_required_keys(self, tmp_path):
        p = tmp_path / "academic_appeals_policy.txt"
        p.write_text(self.FIXTURE, encoding="utf-8")
        for c in chunk_file(p):
            assert {"text", "doc_title", "section", "tokens_est"}.issubset(c.keys())

    def test_doc_title_derived_from_filename(self, tmp_path):
        p = tmp_path / "academic_appeals_policy.txt"
        p.write_text(self.FIXTURE, encoding="utf-8")
        for c in chunk_file(p):
            assert "academic" in c["doc_title"].lower() or "Academic" in c["doc_title"]

    def test_token_budget_respected(self, tmp_path):
        p = tmp_path / "academic_appeals_policy.txt"
        p.write_text(self.FIXTURE, encoding="utf-8")
        for c in chunk_file(p, target_tokens=DEFAULT_TARGET_TOKENS):
            assert c["tokens_est"] < DEFAULT_TARGET_TOKENS * 1.5

    def test_all_chunks_contain_breadcrumb(self, tmp_path):
        p = tmp_path / "academic_appeals_policy.txt"
        p.write_text(self.FIXTURE, encoding="utf-8")
        for c in chunk_file(p):
            assert "[Policy:" in c["text"] and "[Section:" in c["text"]

    def test_key_content_preserved(self, tmp_path):
        p = tmp_path / "academic_appeals_policy.txt"
        p.write_text(self.FIXTURE, encoding="utf-8")
        all_text = " ".join(c["text"] for c in chunk_file(p)).lower()
        assert "appeals" in all_text and "14" in all_text

    def test_empty_file_returns_list(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert isinstance(chunk_file(p), list)



# INTEGRATION TESTS — retrieve_policies with synthetic FAISS index


class TestRetrievePoliciesIntegration:

    @pytest.fixture(autouse=True)
    def patch_retrieval_module(self, monkeypatch):
        from sentence_transformers import SentenceTransformer
        import query_policies as qp

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        synthetic_chunks = [
            {
                "text": "[Policy: Appeals Policy] [Section: Introduction]\n\nStudents may appeal examination results within 14 days.",
                "doc_title": "Appeals Policy",
                "section": "Introduction",
            },
            {
                "text": "[Policy: Late Submission Policy] [Section: Penalties]\n\nLate submissions receive a penalty of 5% per working day.",
                "doc_title": "Late Submission Policy",
                "section": "Penalties",
            },
            {
                "text": "[Policy: Extenuating Circumstances] [Section: Grounds]\n\nStudents may apply for extenuating circumstances due to illness.",
                "doc_title": "Extenuating Circumstances",
                "section": "Grounds",
            },
        ]

        # Build tiny FAISS index
        texts = [c["text"] for c in synthetic_chunks]
        embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
        synthetic_index = faiss.IndexFlatIP(embeddings.shape[1])
        synthetic_index.add(embeddings)

        monkeypatch.setattr(qp, "index", synthetic_index)
        monkeypatch.setattr(qp, "CHUNKS", synthetic_chunks)
        monkeypatch.setattr(qp, "EMB", model)

        # Patch BM25 if the module has it
        try:
            from rank_bm25 import BM25Okapi
            tokenised = [c["text"].lower().split() for c in synthetic_chunks]
            monkeypatch.setattr(qp, "BM25_INDEX", BM25Okapi(tokenised))
        except (ImportError, AttributeError):
            pass

    def test_returns_list(self):
        from query_policies import retrieve_policies
        assert isinstance(retrieve_policies("academic appeal", k=2), list)

    def test_returns_results(self):
        from query_policies import retrieve_policies
        assert len(retrieve_policies("appeal examination", k=2)) >= 1

    def test_result_has_required_keys(self):
        from query_policies import retrieve_policies
        r = retrieve_policies("late submission penalty", k=1)[0]
        assert {"score", "text", "doc_title", "section"}.issubset(r.keys())

    def test_appeal_query_returns_appeals_chunk(self):
        from query_policies import retrieve_policies
        titles = [r["doc_title"] for r in retrieve_policies("appeal examination results", k=3)]
        assert "Appeals Policy" in titles

    def test_extenuating_circumstances_query(self):
        from query_policies import retrieve_policies
        titles = [r["doc_title"] for r in retrieve_policies("illness extenuating circumstances", k=3)]
        assert "Extenuating Circumstances" in titles

    def test_score_is_numeric(self):
        from query_policies import retrieve_policies
        assert isinstance(retrieve_policies("policy", k=1)[0]["score"], (int, float))

    def test_empty_query_does_not_crash(self):
        from query_policies import retrieve_policies
        try:
            assert isinstance(retrieve_policies("", k=1), list)
        except Exception as e:
            pytest.fail(f"Empty query raised: {e}")

    def test_text_field_is_non_empty_string(self):
        from query_policies import retrieve_policies
        for r in retrieve_policies("university policy", k=2):
            assert isinstance(r["text"], str) and len(r["text"]) > 0


# SYSTEM TESTS — full pipeline

class TestSystemPipeline:
    POLICY_TEXT = """LATE SUBMISSION POLICY

OVERVIEW

This policy sets out the University's approach to submissions received after the published deadline. It applies to all students registered on taught programmes at the University of Strathclyde. All students are expected to submit coursework by the published deadline, which is set by the relevant School and communicated through MyPlace at the start of the academic year.

STANDARD PENALTY

Unless extenuating circumstances have been approved in advance, work submitted after the deadline will be subject to a standard penalty. Work submitted up to one working day late will receive a deduction of five percentage points from the mark awarded. Work submitted between one and five working days late will receive a deduction of twenty percentage points. Work submitted more than five working days late will receive a mark of zero. The penalty is applied to the awarded mark and does not require a separate decision by the marker.

EXTENUATING CIRCUMSTANCES

Students who are unable to submit on time due to circumstances beyond their control should apply for an extension before the deadline. Extension requests must be submitted via MyPlace using the official extenuating circumstances form. Medical evidence from a qualified practitioner should be provided where illness is cited. The decision on whether to grant an extension rests with the relevant Head of Department, whose decision is final at this stage.

APPEALS

Students who wish to appeal the application of the late submission penalty may do so using the standard academic appeals process. Appeals must be submitted within 14 working days of receiving the penalised mark. The grounds for appeal are limited to administrative error and exceptional personal circumstances not previously disclosed. All appeals are considered by the Student Appeals Committee, which meets on a monthly basis.
"""

    def test_pipeline_produces_chunks(self, tmp_path):
        p = tmp_path / "late_submission_policy.txt"
        p.write_text(self.POLICY_TEXT, encoding="utf-8")
        assert len(chunk_file(p)) >= 1

    def test_section_metadata_preserved(self, tmp_path):
        p = tmp_path / "late_submission_policy.txt"
        p.write_text(self.POLICY_TEXT, encoding="utf-8")
        assert len({c["section"] for c in chunk_file(p)}) >= 1

    def test_key_terms_present(self, tmp_path):
        p = tmp_path / "late_submission_policy.txt"
        p.write_text(self.POLICY_TEXT, encoding="utf-8")
        all_text = " ".join(c["text"] for c in chunk_file(p)).lower()
        for term in ["penalty", "extension", "appeals", "deadline"]:
            assert term in all_text

    def test_no_empty_chunks(self, tmp_path):
        p = tmp_path / "late_submission_policy.txt"
        p.write_text(self.POLICY_TEXT, encoding="utf-8")
        for c in chunk_file(p):
            assert len(chunk_body(c["text"]).strip()) > 0

    def test_all_chunks_have_doc_title(self, tmp_path):
        p = tmp_path / "late_submission_policy.txt"
        p.write_text(self.POLICY_TEXT, encoding="utf-8")
        for c in chunk_file(p):
            assert c["doc_title"] and len(c["doc_title"]) > 0