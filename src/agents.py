import operator
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    researcher_data: List[str]
    chart_data: List[dict]

from tools import get_llm_with_tools, lookup_policy_docs, web_search_stub, rss_feed_search
llm,llm_with_tools, tools = get_llm_with_tools()

def researcher_node(state: AgentState):
    print("\n--- (Agent: Researcher) is gathering data ---")
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

            research_findings.append(f"Source: {tool_name}\nData: {res}")

    return {
        "messages": [response],
        "researcher_data": research_findings,
    }

def analyst_node(state: AgentState):
    raw_data = "\n\n".join(state["researcher_data"])
    prompt = f"You are a senior analyst. extract trends and numeric data.\n{raw_data}"
    response = llm.ivoke(prompt)
    return {"messages": [response], "chart_data": []}

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", END)

app = workflow.compile()