"""
UOT Live Search Module
======================
Implements the live source acquisition layer for the UOT Temporal Extrapolation Engine.

Architecture (per GPT's recommendation):
  run_searches()         → Tavily API  (structured, auditable, repeatable)
  fetch_top_results()    → direct HTTP fetch + text extraction
  compress_source_to_packet() → Anthropic API  (already in uot_engine_v12_patched.py)
  Stage A–D extraction   → Anthropic API  (already in engine)

This module keeps search and interpretation separate.
Search is mechanical and external. Interpretation is observational and internal.

SETUP
-----
1. Get a Tavily API key: https://tavily.com  (free tier available)
2. Set the environment variable:
      export TAVILY_API_KEY="tvly-xxxxxxxxxxxx"
   or put it in a .env file (see below).

ALTERNATIVE: Brave Search
  If you prefer Brave Search instead of Tavily:
    export SEARCH_PROVIDER=brave
    export BRAVE_API_KEY="your-key-here"
  Brave docs: https://api.search.brave.com/app/documentation/web-search

Usage in uot_engine_v12_patched.py
------------------------------------
In web_search_sources(), replace the live-mode placeholder with:

    from uot_live_search import run_searches_live, fetch_top_results_live
    search_results = run_searches_live(topic, observer_basis)
    raw_docs = fetch_top_results_live(search_results)
    packets = [compress_source_to_packet(doc.text, doc.metadata) for doc in raw_docs]
    return diversify_and_rank_sources(packets)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# These are imported from the engine when used together
# They are defined here only for standalone type checking
try:
    from uot_engine_v12_patched import (
        SearchResult, RawDocument, generate_search_queries, clamp
    )
except ImportError:
    # Fallback definitions for standalone use / testing
    @dataclass
    class SearchResult:
        query: str; purpose: str; title: str; publisher: str; url: str
        date: str = ""; snippet: str = ""; source_type_hint: str = "article"; rank: int = 0

    @dataclass
    class RawDocument:
        title: str; publisher: str; url: str; date: str
        text: str; metadata: dict; cache_key: str

    def clamp(v, lo=0.0, hi=1.0): return max(lo, min(v, hi))


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

def _get_env(key: str, default: str = "") -> str:
    """Read from environment, with .env file fallback."""
    val = os.environ.get(key, "")
    if val:
        return val
    # Try .env file in same directory
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return default


TAVILY_API_KEY   = lambda: _get_env("TAVILY_API_KEY")
BRAVE_API_KEY    = lambda: _get_env("BRAVE_API_KEY")
SEARCH_PROVIDER  = lambda: _get_env("SEARCH_PROVIDER", "tavily").lower()

# Request limits
MAX_FETCH_CHARS  = 8000    # max chars extracted from any single page
FETCH_TIMEOUT    = 10      # seconds before URL fetch times out
MAX_DOCS         = 16      # Phase 8.1: raised from 8 to 16 for outcome-aware source coverage


# ══════════════════════════════════════════════════════════════════════════════
# Search — Tavily
# ══════════════════════════════════════════════════════════════════════════════

def _tavily_search(query: str, api_key: str, max_results: int = 5) -> dict:
    """
    Call Tavily search API for one query.
    Returns the parsed JSON response dict.

    Tavily docs: https://docs.tavily.com/docs/tavily-api/rest_api
    """
    # Phase 8: exclude prediction-market gambling sites from all searches.
    # These are reflexive — they aggregate existing beliefs, not independent evidence.
    # Legitimate forecasting (bank research, Fed minutes, institutional analysis) is allowed.
    _EXCLUDED_DOMAINS = [
        "polymarket.com",
        "predictit.org",
        "kalshi.com",
        "manifold.markets",
        "futuur.com",
        "smarkets.com",
        "betfair.com",
    ]
    payload = json.dumps({
        "api_key":      api_key,
        "query":        query,
        "search_depth": "basic",
        "max_results":  max_results,
        "include_domains": [],
        "exclude_domains": _EXCLUDED_DOMAINS,
    }).encode()

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _brave_search(query: str, api_key: str, max_results: int = 5) -> dict:
    """
    Call Brave Search API for one query.
    Returns the parsed JSON response dict.

    Brave docs: https://api.search.brave.com/app/documentation/web-search
    """
    params = urllib.parse.urlencode({"q": query, "count": max_results})
    req = urllib.request.Request(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        },
        method="GET"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _extract_publisher_from_url(url: str) -> str:
    """Extract a readable publisher name from a URL."""
    try:
        host = urllib.parse.urlparse(url).netloc
        host = re.sub(r'^www\.', '', host)
        # Convert dots to spaces, title-case
        parts = host.split('.')
        if len(parts) >= 2:
            return parts[-2].replace('-', ' ').title()
        return host
    except Exception:
        return "Unknown"


def _estimate_credibility(url: str, source_type: str) -> float:
    """
    Heuristic credibility estimate from URL and source type.
    This is a rough initial estimate; compress_source_to_packet()
    refines it with AI-based assessment.
    """
    high_cred = ['reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com',
                 'washingtonpost.com', 'ft.com', 'economist.com', 'wsj.com',
                 'brookings.edu', 'pewresearch.org', 'nature.com', 'science.org',
                 'cookpolitical.com', 'fivethirtyeight.com', 'natesilver.net']
    med_cred  = ['politico.com', 'axios.com', 'theatlantic.com', 'vox.com',
                 'npr.org', 'pbs.org', 'theguardian.com', 'bloomberg.com']

    url_lower = url.lower()
    if any(s in url_lower for s in high_cred):
        return 0.88
    if any(s in url_lower for s in med_cred):
        return 0.75
    if source_type in ('polling', 'legal_analysis', 'academic'):
        return 0.80
    if source_type == 'forecast':
        return 0.78
    return 0.65


def _estimate_recency(date_str: str) -> float:
    """
    Estimate recency (0-1) from a date string.
    Today = 1.0; one year ago ≈ 0.2.
    """
    if not date_str:
        return 0.5
    try:
        from datetime import datetime, timezone
        # Handle ISO 8601 or date-only strings
        date_str = date_str[:10]
        pub = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_ago = max(0, (now - pub).days)
        # Decay: 0 days = 1.0, 365 days = 0.2, linear
        return clamp(1.0 - (days_ago / 365) * 0.8)
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — run_searches_live()
# ══════════════════════════════════════════════════════════════════════════════

def run_searches_live(
    topic: str,
    observer_basis: dict,
    extra_queries: Optional[List[str]] = None,
    canonical_slots=None,
    horizon=None,
) -> List[SearchResult]:
    """
    Stage 0 — Live search: execute the hybrid fan-out query plan.

    Uses Tavily (default) or Brave Search (set SEARCH_PROVIDER=brave).
    Deduplicates results by URL across all queries.
    Preserves which query found each source and its UOT purpose.

    extra_queries:    optional Stage 0.6 follow-up queries from Stage A gaps.
    canonical_slots:  OutcomeSlot objects from Stage Q — enables slot-targeted queries.
    horizon:          HorizonConfig — for year targeting in queries.

    Phase 8.1: query cap raised from 4 to 16 for outcome-aware search.
    """
    provider = SEARCH_PROVIDER()

    if provider == 'brave':
        api_key = BRAVE_API_KEY()
        if not api_key:
            raise RuntimeError(
                "BRAVE_API_KEY not set. "
                "Get a key at https://api.search.brave.com/ "
                "and set: export BRAVE_API_KEY='your-key'"
            )
    else:
        api_key = TAVILY_API_KEY()
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. "
                "Get a free key at https://tavily.com "
                "and set: export TAVILY_API_KEY='tvly-xxxxxxxxxxxx'"
            )

    # Build query list: structured fan-out + optional follow-ups
    # Phase 8.1: pass canonical_slots + horizon so slot-targeted and
    # disconfirmation queries (Classes 3 and 4) are generated and executed.
    planned_queries = generate_search_queries(
        topic, observer_basis,
        canonical_slots=canonical_slots,
        horizon=horizon,
    )

    follow_up_entries = [
        {"query": q, "purpose": "stage_a_followup",
         "source_type": "article", "max_results": 3}
        for q in (extra_queries or [])[:5]
    ]

    # Phase 8.1: raised from 4 to 16 for outcome-aware search
    all_queries = (planned_queries + follow_up_entries)[:16]

    seen_urls: set = set()
    results: List[SearchResult] = []

    for q_spec in all_queries:
        query      = q_spec["query"]
        purpose    = q_spec["purpose"]
        max_r      = q_spec.get("max_results", 3)
        stype_hint = q_spec.get("source_type", "article")

        try:
            if provider == 'brave':
                raw = _brave_search(query, api_key, max_r)
                items = raw.get("web", {}).get("results", [])
                for rank, item in enumerate(items):
                    url = item.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(SearchResult(
                            query=query, purpose=purpose,
                            title=item.get("title", ""),
                            publisher=_extract_publisher_from_url(url),
                            url=url,
                            date=item.get("age", ""),
                            snippet=(item.get("description") or "")[:300],
                            source_type_hint=stype_hint,
                            rank=rank,
                        ))
            else:
                # Tavily (default)
                raw = _tavily_search(query, api_key, max_r)
                for rank, item in enumerate(raw.get("results", [])):
                    url = item.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(SearchResult(
                            query=query, purpose=purpose,
                            title=item.get("title", ""),
                            publisher=_extract_publisher_from_url(url),
                            url=url,
                            date=item.get("published_date", ""),
                            snippet=(item.get("content") or "")[:300],
                            source_type_hint=stype_hint,
                            rank=rank,
                        ))

        except Exception as e:
            # Log and continue — one failed query should not break the pipeline
            print(f"[search] Query failed ({purpose}): {e}")
            continue

    return results


# ══════════════════════════════════════════════════════════════════════════════
# HTML → Text extraction
# ══════════════════════════════════════════════════════════════════════════════

def _strip_html(html: str) -> str:
    """
    Lightweight HTML-to-text extraction.
    Removes scripts, styles, nav, footer, and ads; extracts main content.
    No external dependencies — pure stdlib.
    """
    # Remove script and style blocks
    html = re.sub(r'<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>', '', html,
                  flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    # Convert block-level tags to newlines
    html = re.sub(r'<(p|br|div|h[1-6]|li|tr)[^>]*>', '\n', html, flags=re.IGNORECASE)
    # Strip all remaining tags
    html = re.sub(r'<[^>]+>', ' ', html)
    # Decode common HTML entities
    entities = {'&amp;': '&', '&lt;': '<', '&gt;': '>', '&nbsp;': ' ',
                '&quot;': '"', '&#39;': "'", '&mdash;': '—', '&ndash;': '–'}
    for ent, char in entities.items():
        html = html.replace(ent, char)
    # Collapse whitespace
    html = re.sub(r'[ \t]+', ' ', html)
    html = re.sub(r'\n{3,}', '\n\n', html)
    return html.strip()


def _extract_main_content(text: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """
    Heuristically extract the "main" portion of extracted text.
    Skips navigation boilerplate at the top, caps at max_chars.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Skip short leading lines (nav, header boilerplate)
    start = 0
    for i, line in enumerate(lines):
        if len(line) > 80:
            start = i
            break
    content = '\n'.join(lines[start:])
    return content[:max_chars]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0.5 — fetch_top_results_live()
# ══════════════════════════════════════════════════════════════════════════════

def fetch_top_results_live(
    search_results: List[SearchResult],
    max_docs: int = MAX_DOCS
) -> List[RawDocument]:
    """
    Stage 0.5 — Build RawDocuments from Tavily search results.

    Uses Tavily's built-in content snippets directly instead of fetching
    full URLs. This saves 20-40 seconds per pipeline run and avoids
    SSL/network errors on cloud deployments.

    Tavily's `content` field returns 200-500 chars per result (basic depth)
    or up to 3000 chars (advanced depth). This is sufficient for Stage A
    AI extraction.
    """
    docs: List[RawDocument] = []

    for sr in search_results[:max_docs]:
        cache_key   = hashlib.md5(sr.url.encode()).hexdigest()[:12]
        credibility = _estimate_credibility(sr.url, sr.source_type_hint)
        recency     = _estimate_recency(sr.date)

        # Use the snippet returned by the search API directly
        text = sr.snippet or f"[{sr.title}] No content available."

        docs.append(RawDocument(
            title=sr.title, publisher=sr.publisher,
            url=sr.url, date=sr.date,
            text=text,
            metadata={
                "title":        sr.title,
                "publisher":    sr.publisher,
                "url":          sr.url,
                "date":         sr.date,
                "source_type":  sr.source_type_hint,
                "credibility":  credibility,
                "recency":      recency,
                "purpose":      sr.purpose,
                "cache_key":    cache_key,
                "snippet":      sr.snippet,
                "fetch_status": "snippet",
                "text_chars":   len(text),
            },
            cache_key=cache_key,
        ))

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper — used by web_search_sources() live path
# ══════════════════════════════════════════════════════════════════════════════

def live_source_pipeline(
    topic: str,
    observer_basis: dict,
    follow_up_queries: Optional[List[str]] = None,
    max_docs: int = MAX_DOCS,
    canonical_slots=None,
    horizon=None,
) -> List[RawDocument]:
    """
    Runs Stage 0 + Stage 0.5 together.
    Returns RawDocuments ready for compress_source_to_packet().

    Phase 8.1: canonical_slots and horizon are now passed through to
    run_searches_live so generate_search_queries receives them and produces
    slot-targeted (Class 3) and disconfirmation (Class 4) queries.
    """
    results  = run_searches_live(topic, observer_basis, follow_up_queries,
                                 canonical_slots=canonical_slots, horizon=horizon)
    raw_docs = fetch_top_results_live(results, max_docs=max_docs)
    return raw_docs


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics — print a pipeline run summary
# ══════════════════════════════════════════════════════════════════════════════

def print_pipeline_summary(raw_docs: List[RawDocument]) -> None:
    """Print a short summary after running the live pipeline."""
    print(f"\n[live_search] {len(raw_docs)} documents fetched:")
    for doc in raw_docs:
        status = doc.metadata.get("fetch_status", "?")
        chars  = doc.metadata.get("text_chars", 0)
        cred   = doc.metadata.get("credibility", 0)
        print(f"  [{status:10s}] {doc.publisher:20s} {chars:5d}ch "
              f"cred={cred:.2f}  {doc.title[:55]}")


# ══════════════════════════════════════════════════════════════════════════════
# Quick test (run directly: python uot_live_search.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) or "Trump presidency 2026"
    basis = {
        "democratic_institutions": 0.9,
        "geopolitical_alliances":  0.8,
        "social_cohesion":         0.6,
        "economic_stability":      0.4,
    }

    print(f"Testing live search pipeline for: '{topic}'")
    print(f"Provider: {SEARCH_PROVIDER()}")
    print()

    try:
        raw_docs = live_source_pipeline(topic, basis)
        print_pipeline_summary(raw_docs)

        if raw_docs:
            print(f"\nFirst document preview ({raw_docs[0].title}):")
            print(raw_docs[0].text[:400])
            print("...")

    except RuntimeError as e:
        print(f"\nSetup required: {e}")
        print("\nTo test:")
        print("  export TAVILY_API_KEY='tvly-xxxxxxxxxxxx'")
        print("  python uot_live_search.py 'Trump presidency'")
