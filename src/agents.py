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
    last_message = state('message')