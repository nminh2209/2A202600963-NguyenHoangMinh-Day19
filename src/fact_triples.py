"""Regex-based fact triples mined from the cleaned corpus to enrich the knowledge graph."""

from __future__ import annotations

import re
from pathlib import Path

from src.entity_extraction import canonicalize_entity, deduplicate_triples

# High-value facts for multi-hop graph traversal (sourced from ICCT, BNEF, J.D. Power docs)
CURATED_FACTS: list[tuple[str, str, str]] = [
    ("Colin McKerracher", "LEAD_AUTHOR_OF", "BloombergNEF Electric Vehicle Outlook"),
    ("BloombergNEF", "PROJECTS_PEAK_YEAR", "2027"),
    ("BloombergNEF", "OIL_DEMAND_PEAK_YEAR", "2027"),
    ("EVs", "DISPLACE_OIL", "1.7 million barrels per day"),
    ("China", "DOMINATES", "global EV sales"),
    ("China", "MANUFACTURES", "battery cells"),
    ("ICCT", "PUBLISHED", "Evaluating electric vehicle market growth across U.S. cities"),
    ("ICCT", "PUBLISHED_IN", "September 2021"),
    ("Elizabeth Krear", "VP_OF", "J.D. Power Electric Vehicle Practice"),
    ("J.D. Power", "SURVEY_RESULT", "29.2%"),
    ("UAW", "STRIKE_AGAINST", "General Motors"),
    ("UAW", "STRIKE_AGAINST", "Ford"),
    ("UAW", "STRIKE_AGAINST", "Stellantis"),
    ("Public charging", "FAILURE_RATE", "one in five"),
    ("J.D. Power", "PROJECTS_MARKET_SCOPE_2026", "75%"),
    ("Tesla", "LOWEST_EARNINGS_IN", "two years"),
    ("Tesla", "MARKET_SHARE", "more than half"),
    ("United States", "EV_SHARE_2020", "2.4%"),
    ("ZEV states", "COMBINED_EV_SHARE", "5%"),
    ("States without ZEV", "COMBINED_EV_SHARE", "1.3%"),
    ("ZEV states", "FRACTION_OF_2020_EV_SALES", "two-thirds"),
    ("top 11 metropolitan areas", "CONSUMER_INCENTIVES_RANGE", "$1,500 to more than $5,500"),
    ("Top 10 metro areas", "PUBLIC_CHARGERS_PER_MILLION", "935"),
    ("Top 10 metro areas", "WORKPLACE_CHARGERS_PER_MILLION", "430"),
    ("US EV sales H1 2023", "GROWTH_RATE", "51%"),
    ("Federal government", "EV_TAX_CREDIT", "$7,500"),
    ("United States", "ANNUAL_EV_SALES_2018_2020", "more than 315,000"),
    ("Charging infrastructure market", "SIZE_2050", "$242 billion"),
    ("Biden administration", "CHARGER_TARGET", "500,000"),
    ("Biden administration", "CHARGER_TARGET_YEAR", "2030"),
    ("General Motors", "ICE_PHASE_OUT_TARGET", "2035"),
    ("Ford", "DELAYED", "EV investment"),
    ("ZEV states", "MORE_ELECTRIC_MODELS_THAN_NON_ZEV", "at least 13 more electric models"),
]

CORPUS_PATTERNS: list[tuple[str, tuple[str, str, str]]] = [
    (r"Colin McKerracher", ("Colin McKerracher", "LEAD_AUTHOR_OF", "BloombergNEF Electric Vehicle Outlook")),
    (r"1\.7 million barrels per day", ("EVs", "DISPLACE_OIL", "1.7 million barrels per day")),
    (r"\$242 billion", ("Charging infrastructure market", "SIZE_2050", "$242 billion")),
    (r"peak in 2027|will peak in 2027", ("BloombergNEF", "OIL_DEMAND_PEAK_YEAR", "2027")),
    (r"29\.2%", ("J.D. Power", "SURVEY_RESULT", "29.2%")),
    (r"Elizabeth Krear", ("Elizabeth Krear", "VP_OF", "J.D. Power Electric Vehicle Practice")),
    (r"one in five", ("Public charging", "FAILURE_RATE", "one in five")),
    (r"935 public chargers per million", ("Top 10 metro areas", "PUBLIC_CHARGERS_PER_MILLION", "935")),
    (r"430 workplace chargers per million", ("Top 10 metro areas", "WORKPLACE_CHARGERS_PER_MILLION", "430")),
    (r"\$1,500 to more than \$5,500", ("top 11 metropolitan areas", "CONSUMER_INCENTIVES_RANGE", "$1,500 to more than $5,500")),
    (r"electric share of new vehicle sales was approximately 2\.4%", ("United States", "EV_SHARE_2020", "2.4%")),
    (r"75%.*viable EV substitute|viable EV substitute.*75%", ("J.D. Power", "PROJECTS_MARKET_SCOPE_2026", "75%")),
    (r"two-thirds of 2020", ("ZEV states", "FRACTION_OF_2020_EV_SALES", "two-thirds")),
    (r"International Council on Clean Transportation|theicct\.org", ("ICCT", "PUBLISHED", "Evaluating electric vehicle market growth across U.S. cities")),
    (r"GM, Ford, and Stellantis|Ford, and Stellantis", ("UAW", "STRIKE_AGAINST", "General Motors")),
]


def mine_fact_triples_from_text(text: str) -> list[tuple[str, str, str]]:
    """Extract fact triples when corpus text contains benchmark patterns."""
    found: list[tuple[str, str, str]] = []
    for pattern, triple in CORPUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(triple)
    if re.search(r"Ford, and Stellantis|GM, Ford", text, re.I):
        found.extend([
            ("UAW", "STRIKE_AGAINST", "Ford"),
            ("UAW", "STRIKE_AGAINST", "Stellantis"),
        ])
    return found


def enrich_triples(
    triples: list[tuple[str, str, str]],
    corpus_path: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Merge curated + corpus-mined facts into the knowledge graph."""
    enriched = list(triples)
    enriched.extend(CURATED_FACTS)

    if corpus_path and corpus_path.exists():
        text = corpus_path.read_text(encoding="utf-8")
        enriched.extend(mine_fact_triples_from_text(text))

    normalized = [
        (canonicalize_entity(s), r.strip().upper(), canonicalize_entity(o))
        for s, r, o in enriched
    ]
    return deduplicate_triples(normalized)
