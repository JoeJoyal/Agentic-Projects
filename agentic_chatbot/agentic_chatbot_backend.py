from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langgraph.graph.message import add_messages # from langgraph.reducers import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI() 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    messages = state["messages"]
    # sent to llm
    response = llm.invoke(messages)
    # response store state
    return {'messages': [response]}


# Memory  
checkpoint = MemorySaver()

# State
graph = StateGraph(ChatState) 

# Add nodes to the graph
graph.add_node('chat_node', chat_node)

# Add edges to the graph
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)