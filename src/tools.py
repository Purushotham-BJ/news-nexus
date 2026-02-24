import os
from langchain.tools import tool
from langchain_ollama import ChatOllama
from retrieval import retrieve_documents

@tool
def lookup_policy_docs(query: str) -> str:
    if isinstance(query, str) and "{" in query:
        query = query.replace("{", "").replace("}", "").replace("value:", "")

    docs = retrieve_documents(query, k=3)

    if not docs:
        return f"No documents found internally relevant for the query: {query}"

    results = []

    for doc, score in docs:
        source_name = doc.metadata.get("source", "unknown PDF")
        basename = os.path.basename(source_name)
        safe_source_path = source_name.replace("\\", "/")

        results.append(
            f"Content: {doc.page_content}\n"
            f"SourceLink: [{basename}](file:///{safe_source_path})"
        )

    return "\n\n".join(results)


@tool
def web_search_stub(query: str) -> str:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        result = list(ddgs.text(query, max_results=5))

    if not result:
        return "No results found"

    formatted_results = []
    for res in result:
        formatted_results.append(
            f"Title: {res.get('title')}\n"
            f"Link: [{res.get('title')}]({res.get('href')})\n"
            f"Snippet: {res.get('body')}"
        )

    return "\n\n---\n".join(formatted_results)


@tool
def rss_feed_search(query: str) -> str:
    import feedparser

    FEEDS = [
        "https://www.technologyreview.com/feed/",
        "https://openai.com/news/rss.xml",
        "https://techcrunch.com/feed/",
    ]

    results = []
    keywords = query.lower().split()

    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            text_to_search = (entry.title + " " + entry.get("summary", "")).lower()

            if any(kw in text_to_search for kw in keywords):
                results.append(
                    f"Title: {entry.title}\n"
                    f"Link: [{entry.title}]({entry.link})"
                )

    return "\n\n---\n".join(results) if results else "No matching RSS entries found"


def get_llm_with_tools():
    llm = ChatOllama(model="llama3.2", temperature=0)
    tools = [lookup_policy_docs, web_search_stub, rss_feed_search]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools