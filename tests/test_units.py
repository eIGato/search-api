"""Unit tests for the pure logic: chunking, thesaurus, rank fusion, summaries, LLM fallbacks."""

import asyncio
import itertools
import uuid
from typing import Any

import anthropic
import httpx2
import pytest

from search_api.chunking import chunk_text, split_sentences
from search_api.llm import LLMClient
from search_api.search import compact, fuse
from search_api.summary import extractive_summary
from search_api.thesaurus import expand_query
from tests.conftest import HashingEmbedder

# --- chunking ----------------------------------------------------------------------------------


def test_split_sentences_handles_punctuation_and_paragraphs() -> None:
    text = "First sentence. Second one?  Third!\n\nNew paragraph without dot"
    assert split_sentences(text) == [
        "First sentence.",
        "Second one?",
        "Third!",
        "New paragraph without dot",
    ]


def test_short_text_is_a_single_chunk() -> None:
    assert chunk_text("Just a few words.", max_words=10, overlap_words=2) == ["Just a few words."]


def test_chunks_respect_max_words_and_overlap() -> None:
    sentences = [f"Sentence number {i} has six words." for i in range(10)]
    chunks = chunk_text(" ".join(sentences), max_words=15, overlap_words=6)

    assert len(chunks) > 1
    assert all(len(c.split()) <= 15 for c in chunks)
    # Each chunk starts with the tail of the previous one.
    for previous, current in itertools.pairwise(chunks):
        assert current.split()[:6] == previous.split()[-6:]
    # Nothing is lost.
    for sentence in sentences:
        assert any(sentence in c for c in chunks)


def test_sentence_longer_than_window_is_split() -> None:
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), max_words=10, overlap_words=0)
    assert [len(c.split()) for c in chunks] == [10, 10, 5]
    assert " ".join(chunks).split() == words


def test_overlap_must_be_smaller_than_window() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", max_words=5, overlap_words=5)


# --- thesaurus ---------------------------------------------------------------------------------


def test_concept_query_expands_to_evidence() -> None:
    expansions = expand_query("address proof")
    assert "utility bill" in expansions
    assert "bank statement" in expansions
    # Evidence comes before alternative spellings of the concept.
    assert expansions.index("utility bill") < expansions.index("proof of address")


def test_evidence_query_expands_to_concept_but_not_siblings() -> None:
    expansions = expand_query("Utility Bills")  # case and plural insensitive
    assert "proof of address" in expansions
    assert "bank statement" not in expansions


def test_expansion_excludes_phrases_already_in_query() -> None:
    assert "address proof" not in expand_query("address proof")


def test_unrelated_query_has_no_expansions() -> None:
    assert expand_query("NevisWealth") == []
    assert expand_query("john.doe@neviswealth.com") == []


def test_matching_is_on_word_boundaries() -> None:
    # "pension" is evidence for retirement, but "suspension" must not trigger it.
    assert expand_query("account suspension") == []


def test_expansion_limit() -> None:
    assert len(expand_query("address proof", limit=3)) == 3


# --- client query normalization ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NevisWealth", "neviswealth"),
        ("nevis wealth", "neviswealth"),
        ("john.doe@neviswealth.com", "johndoeneviswealthcom"),
        ("O'Brien-Smith", "obriensmith"),
    ],
)
def test_compact(raw: str, expected: str) -> None:
    assert compact(raw) == expected


# --- rank fusion -------------------------------------------------------------------------------


def test_fuse_rewards_agreement_between_retrievers() -> None:
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    keyword = [(a, "**a** headline"), (b, "b headline")]
    semantic = [(a, 0.8, "a chunk"), (c, 0.7, "c chunk")]

    hits = fuse(keyword, semantic, rrf_k=60)

    assert hits[0].document_id == a
    by_id = {h.document_id: h for h in hits}
    assert by_id[a].score == 1.0
    assert by_id[a].matched_by == ["keyword", "semantic"]
    assert by_id[b].matched_by == ["keyword"]
    assert by_id[c].matched_by == ["semantic"]
    assert 0 < by_id[c].score < by_id[a].score
    # Keyword headlines are preferred as snippets; semantic-only hits fall back to the chunk.
    assert by_id[a].snippet == "**a** headline"
    assert by_id[c].snippet == "c chunk"


def test_fuse_truncates_long_chunk_snippets_on_word_boundary() -> None:
    doc = uuid.uuid4()
    long_chunk = "word " * 200
    [hit] = fuse([], [(doc, 0.5, long_chunk)], rrf_k=60)
    assert len(hit.snippet) <= 301
    assert hit.snippet.endswith("word…")


def test_fuse_empty() -> None:
    assert fuse([], [], rrf_k=60) == []


# --- extractive summary ------------------------------------------------------------------------


def test_extractive_summary_returns_short_text_unchanged() -> None:
    text = "One sentence. Two sentences."
    assert asyncio.run(extractive_summary(HashingEmbedder(), text)) == text


def test_extractive_summary_picks_central_sentences_in_order() -> None:
    text = (
        "The portfolio returned five percent this quarter. "
        "Lunch was served at noon. "
        "Portfolio equities returned seven percent this quarter. "
        "The portfolio bonds returned two percent this quarter. "
        "Parking is available downstairs."
    )
    summary = asyncio.run(extractive_summary(HashingEmbedder(), text, max_sentences=3))
    assert "Lunch" not in summary
    assert "Parking" not in summary
    assert summary.startswith("The portfolio returned five percent")


# --- LLM client fallbacks ------------------------------------------------------------------------


def _llm_raising(exc: Exception) -> LLMClient:
    llm = LLMClient(api_key="test", model="claude-haiku-4-5", timeout_seconds=1)

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise exc

    llm._client.messages.create = fail  # type: ignore[method-assign]
    llm._client.messages.parse = fail  # type: ignore[method-assign]
    return llm


@pytest.mark.parametrize(
    "exc",
    [
        anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example")),
        anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.example")),
        anthropic.RateLimitError(
            "rate limited",
            response=httpx2.Response(429, request=httpx2.Request("POST", "https://api.example")),
            body=None,
        ),
    ],
)
def test_llm_failures_degrade_gracefully(exc: Exception) -> None:
    llm = _llm_raising(exc)
    assert asyncio.run(llm.summarize("title", "content")) is None
    assert asyncio.run(llm.expand_query("address proof")) is None


def test_identical_concurrent_expansions_share_one_call() -> None:
    llm = LLMClient(api_key="test", model="claude-haiku-4-5", timeout_seconds=1)
    calls: list[str] = []

    async def fake_expand(query: str) -> list[str] | None:
        calls.append(query)
        await asyncio.sleep(0.05)
        return [f"{query} related"]

    llm._expand_query = fake_expand  # type: ignore[method-assign]

    async def run() -> list[list[str] | None]:
        return list(
            await asyncio.gather(
                llm.expand_query("tax"), llm.expand_query(" TAX "), llm.expand_query("pension")
            )
        )

    assert asyncio.run(run()) == [["tax related"], ["tax related"], ["pension related"]]
    assert calls == ["tax", "pension"]
    assert llm._expansions_in_flight == {}  # cleaned up
