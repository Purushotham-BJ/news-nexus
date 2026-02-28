import os
import re
from langchain.tools import tool
from langchain_ollama import ChatOllama
from retrieval import retrieve_documents


@tool
def lookup_policy_docs(query: str) -> str:
    """
    Search internal vector database for relevant policy documents.
    Returns document content along with source file links.
    """

    # Clean accidental structured inputs from LLM tool calls
    if isinstance(query, str) and "{" in query:
        query = query.replace("{", "").replace("}", "").replace("value:", "").strip()

    docs = retrieve_documents(query, k=3)

    if not docs:
        return f"No documents found internally relevant for the query: {query}"

    results = []

    for item in docs:
        # Handle both (doc, score) and doc-only formats safely
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc = item
            score = None

        source_name = doc.metadata.get("source", "unknown PDF")
        basename = os.path.basename(source_name)
        safe_source_path = source_name.replace("\\", "/")

        results.append(
            f"Content:\n{doc.page_content}\n\n"
            f"Source: {basename}\n"
            f"Score: {score if score is not None else 'N/A'}\n"
            f"SourceLink: file:///{safe_source_path}"
        )

    return "\n\n---\n\n".join(results)


@tool
def web_search_stub(query: str) -> str:
    """
    Perform a web search using DuckDuckGo and return top 5 results.
    """

    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        result = list(ddgs.text(query, max_results=5))

    if not result:
        return "No results found"

    formatted_results = []

    for res in result:
        formatted_results.append(
            f"Title: {res.get('title')}\n"
            f"Link: {res.get('href')}\n"
            f"Snippet: {res.get('body')}"
        )

    return "\n\n---\n\n".join(formatted_results)


@tool
def rss_feed_search(query: str) -> str:
    """
    Search predefined RSS feeds for articles matching query keywords.
    """

    import feedparser

    FEEDS = [
        "https://www.technologyreview.com/feed/",
        "https://openai.com/news/rss.xml",
        "https://techcrunch.com/feed/",
    ]

    results = []
    keywords = set(query.lower().split())

    for url in FEEDS:
        feed = feedparser.parse(url)

        for entry in feed.entries[:10]:
            text_to_search = (
                entry.title + " " + entry.get("summary", "")
            ).lower()

            # safer word matching
            words = set(re.findall(r"\w+", text_to_search))

            if keywords & words:
                results.append(
                    f"Title: {entry.title}\n"
                    f"Link: {entry.link}"
                )

    return "\n\n---\n\n".join(results) if results else "No matching RSS entries found"


def get_llm_with_tools():
    """
    Initialize ChatOllama model and bind available tools.
    """

    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    tools = [
        lookup_policy_docs,
        web_search_stub,
        rss_feed_search
    ]

    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, tools