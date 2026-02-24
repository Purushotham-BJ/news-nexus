import operator
from typing import Annotated,List,TypedDict
from langgraph.graph import StateGraph,END
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,AIMessage
from langchain_ollama import Chatollama 

class Agentstate(TypedDict):
    messages: Annotated[list[BaseMessage],operator.add]
    researcher_data : List[str]
    chart_data : List[dict]

def researcher_node(state: Agentstate) :
    print('\n---(Agent:Researcher) is gathering data ---')
    return {"messages": state['messages']}

workflow = StateGraph(Agentstate)
workflow.add_node("researcher", researcher_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "END")

workflow.compile()