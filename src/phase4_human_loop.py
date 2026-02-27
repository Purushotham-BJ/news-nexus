import os
import operator
from typing import Literal, TypedDict
from langgraph.graph import StateGraph,END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agents import (AgentState, researcher_node, analyst_node, writer_node,llm_with_tools)

def human_approval_node(state:AgentState):
    """
    This node acts as a pause point .
    the graph stops here, allowing the human inspect the state.
    """
    return state

def route_after_human(state:AgentState)->Literal["Writer","__end__"]:
    last_msg=state["messages"][-1].content.lower()
    if "approve" in last_msg:
        return "__end__"
    else:
        print("\n --- [System]Feedback recived.Routing back to writer...---")
        return "Writer"

workflow = StateGraph(AgentState)
workflow.add_node("Researcher",researcher_node)
workflow.add_node("Analyst",analyst_node)
workflow.add_node("Writer",writer_node)
workflow.add_node("human_approval",human_approval_node)

workflow.set_entry_point("Researcher")
workflow.add_edge("Resaercher","Analyst")
workflow.add_edge("Analyst","Writer")
workflow.add_edge("writer","human_approval")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory,interrupt_before =["human_approval"])

if __name__ == "__main__":
    print("=================================================")
    user_topic = input("Enter a topic : (eg.,'AI trends in 2026')")
    config = {"configurable":{"thread_id":"session_phase4"}}
    inputs = {"messages":[HumanMessage(content=user_topic)],"researcher_data":[]}
    for output in app.stream(inputs,config):
        pass
    while True:
        state=app.get_state(config)
        if not state.values:
            print("Error:No state found") 
            break
        current_draft = state.values['messages'][-1].content.lower()
