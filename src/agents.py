import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from tools import get_llm_with_tools, lookup_policy_docs, web_search_stub, rss_feed_search


# ----------------------------
# Agent State
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    researcher_data: List[str]
    chart_data: List[dict]


# ----------------------------
# LLM Setup
# ----------------------------
llm_with_tools, tools = get_llm_with_tools()
llm = ChatOllama(model="llama3.2", temperature=0)


# ----------------------------
# Researcher Node
# ----------------------------
def researcher_node(state: AgentState):
    print("\n--- (Agent: Researcher) Gathering Data ---")

    last_message = state["messages"][-1]
    sys_msg = SystemMessage(content="You are a data gatherer. Use tools when needed.")

    response = llm_with_tools.invoke([sys_msg, last_message])
    research_findings = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            q = str(tool_args.get("query", ""))

            if tool_name == "lookup_policy_docs":
                res = lookup_policy_docs.invoke(q)
            elif tool_name == "web_search_stub":
                res = web_search_stub.invoke(q)
            elif tool_name == "rss_feed_search":
                res = rss_feed_search.invoke(q)
            else:
                res = "Unknown tool"

            research_findings.append(f"Source: {tool_name}\nData:\n{res}")

    return {
        "messages": [response],
        "researcher_data": research_findings,
    }


# ----------------------------
# Analyst Node
# ----------------------------
def analyst_node(state: AgentState):
    print("\n--- (Agent: Analyst) Extracting Insights ---")

    raw_data = "\n\n".join(state.get("researcher_data", []))

    prompt = f"""
You are a senior analyst.
Extract trends, patterns, and numeric insights from the data below.

{raw_data}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [response],
        "chart_data": []
    }


# ----------------------------
# Writer Node
# ----------------------------
def writer_node(state: AgentState):
    print("\n--- (Agent: Writer) Creating HTML Newsletter ---")

    analyst_insights = state["messages"][-1].content

    prompt = f"""
You are a newsletter editor.

Compile the analysis into a professional HTML newsletter.

CRITICAL:
- Preserve all markdown links [Title](URL)
- Convert them into clickable HTML links:
  <a href="URL">Title</a>

TRENDS & ANALYSIS:
{analyst_insights}
"""

    response = llm.invoke(prompt)

    return {"messages": [response]}


# ----------------------------
# Build Workflow
# ----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)

workflow.set_entry_point("Researcher")

workflow.add_edge("Researcher", "Analyst")
workflow.add_edge("Analyst", "Writer")
workflow.add_edge("Writer", END)

app = workflow.compile()

# ----------------------------
# Run Application
# ----------------------------
if __name__ == "__main__":

    user_topic = "latest AI trends and internal productivity reports"

    inputs = {
        "messages": [HumanMessage(content=user_topic)],
        "researcher_data": [],
        "chart_data": []
    }

    final_output = None

    for step in app.stream(inputs):
        print(step)

        if "Writer" in step:
            final_output = step["Writer"]["messages"][-1].content

    print("\n\n=== Final Newsletter (HTML) ===\n")
    print(final_output)