import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split text into windows of at most `max_words` words, preferring sentence boundaries.

    Consecutive chunks share up to `overlap_words` trailing words so that a phrase cut at a
    boundary still appears whole in one of them. Sentences longer than `max_words` are split
    on word boundaries.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    words_per_sentence: list[list[str]] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        for start in range(0, len(words), max_words):
            words_per_sentence.append(words[start : start + max_words])

    chunks: list[str] = []
    current: list[str] = []
    for words in words_per_sentence:
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            overlap = current[-overlap_words:] if overlap_words else []
            # Keep the overlap only if the next sentence still fits after it.
            current = overlap if len(overlap) + len(words) <= max_words else []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks
