"""Domain thesaurus for query expansion.

Small embedding models capture general similarity well but miss domain knowledge such as
"a utility bill is accepted as proof of address". In wealth management the set of document
types an advisor deals with (KYC, tax, estate planning...) is finite and well known, so a curated
concept list is a cheap, deterministic and explainable way to cover it. Free-form semantics are
still handled by embeddings, and optionally by LLM query expansion.

Each concept has:
- aliases: ways to name the concept itself ("proof of address", "address proof").
- evidence: concrete documents or terms that satisfy it ("utility bill", "bank statement").

A query mentioning an alias is expanded with the other aliases and all evidence terms.
A query mentioning an evidence term is expanded with the aliases only: "utility bill" should find
documents labelled "proof of address", but not every other kind of proof of address.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    aliases: tuple[str, ...]
    evidence: tuple[str, ...]


CONCEPTS: tuple[Concept, ...] = (
    Concept(
        aliases=(
            "proof of address",
            "address proof",
            "proof of residence",
            "proof of residency",
            "address verification",
        ),
        evidence=(
            "utility bill",
            "electricity bill",
            "gas bill",
            "water bill",
            "council tax bill",
            "bank statement",
            "mortgage statement",
            "tenancy agreement",
            "lease agreement",
            "rental agreement",
        ),
    ),
    Concept(
        aliases=(
            "proof of identity",
            "identity proof",
            "identity document",
            "id document",
            "photo id",
            "identity verification",
        ),
        evidence=(
            "passport",
            "driving licence",
            "driver's license",
            "national id card",
            "identity card",
            "residence permit",
        ),
    ),
    Concept(
        aliases=("source of wealth", "source of funds", "origin of funds"),
        evidence=(
            "payslip",
            "salary",
            "employment contract",
            "inheritance",
            "sale of business",
            "property sale",
            "dividends",
        ),
    ),
    Concept(
        aliases=(
            "risk profile",
            "risk tolerance",
            "risk appetite",
            "risk assessment",
            "attitude to risk",
        ),
        evidence=(
            "investment policy statement",
            "suitability assessment",
            "suitability report",
            "risk questionnaire",
        ),
    ),
    Concept(
        aliases=("tax documents", "tax records", "tax filing"),
        evidence=("tax return", "w-2", "form 1099", "p60", "capital gains statement"),
    ),
    Concept(
        aliases=("estate planning", "estate plan", "succession planning"),
        evidence=(
            "last will and testament",
            "trust deed",
            "power of attorney",
            "beneficiary designation",
        ),
    ),
    Concept(
        aliases=("retirement planning", "retirement savings", "retirement"),
        evidence=(
            "pension",
            "401(k)",
            "individual retirement account",
            "annuity",
            "superannuation",
        ),
    ),
    Concept(
        aliases=("kyc", "know your customer", "customer due diligence", "client onboarding"),
        evidence=("proof of identity", "proof of address", "source of wealth", "aml check"),
    ),
    Concept(
        aliases=(
            "sustainable investing",
            "esg",
            "responsible investing",
            "impact investing",
            "ethical investing",
        ),
        evidence=(
            "esg preferences",
            "green bonds",
            "renewable energy",
            "fossil fuel exclusion",
            "climate",
        ),
    ),
    Concept(
        aliases=("portfolio performance", "investment performance"),
        evidence=("quarterly report", "portfolio report", "performance report", "valuation"),
    ),
)

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation and naive plural "s", so "Utility Bills" ~ "utility bill"."""
    words = _NON_WORD.sub(" ", text.lower()).split()
    return " ".join(
        w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words
    )


def _contains(haystack: str, phrase: str) -> bool:
    return f" {phrase} " in f" {haystack} "


def expand_query(query: str, limit: int = 16) -> list[str]:
    """Return related phrases for `query` (not including phrases the query already contains)."""
    normalized_query = _normalize(query)
    expansions: list[str] = []
    for concept in CONCEPTS:
        if any(_contains(normalized_query, _normalize(a)) for a in concept.aliases):
            # Evidence first: other spellings of the concept add little beyond the query itself.
            expansions.extend(concept.evidence)
            expansions.extend(concept.aliases)
        elif any(_contains(normalized_query, _normalize(e)) for e in concept.evidence):
            expansions.extend(concept.aliases)

    seen: set[str] = set()
    result: list[str] = []
    for phrase in expansions:
        key = _normalize(phrase)
        if key in seen or _contains(normalized_query, key):
            continue
        seen.add(key)
        result.append(phrase)
    return result[:limit]
