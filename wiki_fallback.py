"""
Wikipedia fallback for when course-document retrieval finds nothing relevant.

Used by app.py when the FAISS+BM25+reranker pipeline returns no candidate
chunks for the selected course, so the assistant doesn't just dead-end with
"not found" -- it tries to answer from Wikipedia instead, clearly labeled
as outside the uploaded course material.
"""

import requests

WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
REQUEST_TIMEOUT = 8  # seconds -- don't let a hung request stall the chat


HEADERS = {
    "User-Agent": "UniRAG/1.0 (https://github.com/oluwaisrael/rag-course-app)"
}

DEBUG = False 


def wikipedia_search(query: str):
    """
    Search Wikipedia for the given query and return a short summary + URL
    of the best-matching article.

    Returns a dict like:
        {
            "title": "Insurable interest",
            "extract": "Insurable interest is a doctrine...",
            "url": "https://en.wikipedia.org/wiki/Insurable_interest"
        }

    Returns None if nothing relevant was found, the network failed, or the
    response was malformed -- the caller (app.py) is expected to handle
    None by falling back to a plain "couldn't find this anywhere" message.
    """
    title = _find_best_matching_title(query)
    if not title:
        return None

    summary = _fetch_summary(title)
    return summary


def _find_best_matching_title(query: str):
    """Step 1: use Wikipedia's search API to find the top matching page title."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }

    try:
        response = requests.get(WIKI_SEARCH_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if DEBUG:
            print(f"[wiki_fallback] search request failed: {e}")
        return None

    try:
        data = response.json()
        results = data["query"]["search"]
    except (ValueError, KeyError) as e:
        if DEBUG:
            print(f"[wiki_fallback] search response malformed: {e} | raw: {response.text[:300]}")
        return None

    if not results:
        if DEBUG:
            print(f"[wiki_fallback] no search results for query: {query!r}")
        return None

    title = results[0]["title"]
    if DEBUG:
        print(f"[wiki_fallback] matched title: {title!r}")
    return title


def _fetch_summary(title: str):
    """Step 2: fetch the page summary for a known title via the REST summary endpoint."""
    url = WIKI_SUMMARY_URL.format(title=title.replace(" ", "_"))

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if DEBUG:
            print(f"[wiki_fallback] summary request failed for {title!r}: {e}")
        return None

    try:
        data = response.json()
    except ValueError as e:
        if DEBUG:
            print(f"[wiki_fallback] summary response not JSON: {e} | raw: {response.text[:300]}")
        return None

    extract = data.get("extract")
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page")

    if not extract or not page_url:
        if DEBUG:
            print(f"[wiki_fallback] summary missing extract/url for {title!r} | data keys: {list(data.keys())}")
        return None

    return {
        "title": data.get("title", title),
        "extract": extract,
        "url": page_url,
    }