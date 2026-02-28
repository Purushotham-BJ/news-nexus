import operator
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Import your existing nodes
from agents import (
    AgentState,
    researcher_node,
    analyst_node,
    writer_node,
)

# =====================================================
# Human Approval Node (Pause Point)
# =====================================================
def human_approval_node(state: AgentState):
    """
    Pause point for human review.
    Execution will stop here until resumed.
    """
    return state


# =====================================================
# Conditional Routing After Human Feedback
# =====================================================
def route_after_human(state: AgentState) -> Literal["Writer", "__end__"]:
    last_msg = state["messages"][-1].content.lower()

    if "approve" in last_msg:
        print("\n--- [System] Approved. Ending workflow. ---")
        return "__end__"
    else:
        print("\n--- [System] Feedback received. Routing back to Writer... ---")
        return "Writer"


# =====================================================
# Build Workflow
# =====================================================
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("human_approval", human_approval_node)

workflow.set_entry_point("Researcher")

workflow.add_edge("Researcher", "Analyst")
workflow.add_edge("Analyst", "Writer")
workflow.add_edge("Writer", "human_approval")

# 🔥 This enables feedback routing
workflow.add_conditional_edges(
    "human_approval",
    route_after_human
)

# =====================================================
# Memory + Interrupt Configuration
# =====================================================
memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_approval"]
)

# =====================================================
# CLI Execution Mode (Optional)
# =====================================================
if __name__ == "__main__":

    print("==============================================")
    topic = input("Enter a topic (e.g., AI trends 2026): ")

    config = {"configurable": {"thread_id": "hitl_session"}}

    inputs = {
        "messages": [HumanMessage(content=topic)],
        "researcher_data": [],
        "chart_data": []
    }

    # Run until human approval interrupt
    for _ in app.stream(inputs, config):
        pass

    while True:
        state = app.get_state(config)

        if not state.values:
            print("Error: No state found.")
            break

        current_draft = state.values["messages"][-1].content

        print("\n=========== CURRENT DRAFT ===========\n")
        print(current_draft)
        print("\n====================================")

        feedback = input(
            "\nType 'approve' to finalize OR provide feedback to improve:\n> "
        )

        if feedback.strip().lower() == "approve":
            print("\nWorkflow completed successfully ✅")
            break

        # Resume graph with human feedback
        app.update_state(
            config,
            {"messages": [HumanMessage(content=feedback)]}
        )

        for _ in app.stream(None, config):
            pass