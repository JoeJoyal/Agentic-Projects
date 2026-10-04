from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langgraph.graph.message import add_messages # from langgraph.reducers import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sqlite3

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
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# State
graph = StateGraph(ChatState) 

# Add nodes to the graph
graph.add_node('chat_node', chat_node)

# Add edges to the graph
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)

# CONFIG = {"configurable": {"thread_id": "default_thread"}}

# res = chatbot.invoke(
#     {"messages": [HumanMessage(content="give me a fullform of mca")]},
#     config=CONFIG
# )

# print(res)

# threads = checkpoint.list(None) # I don't looking for specific chats
# print(threads)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)
