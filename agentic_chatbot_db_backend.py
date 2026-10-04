import sqlite3

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI()


# ============================================================
# State
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# Chat Node
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# SQLite Connection
# ============================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)


# ============================================================
# LangGraph SQLite Checkpoint
# ============================================================

checkpoint = SqliteSaver(conn)


# ============================================================
# Create Graph
# ============================================================

graph = StateGraph(ChatState)


graph.add_node(
    "chat_node",
    chat_node
)


graph.add_edge(
    START,
    "chat_node"
)


graph.add_edge(
    "chat_node",
    END
)


# ============================================================
# Compile Graph
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpoint
)


# ============================================================
# Chat Metadata Table
# ============================================================

conn.execute("""
CREATE TABLE IF NOT EXISTS chat_metadata (
    thread_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()


# ============================================================
# Save Chat Title
# ============================================================

def save_chat_title(thread_id, title):

    conn.execute(
        """
        INSERT INTO chat_metadata (
            thread_id,
            title
        )
        VALUES (?, ?)

        ON CONFLICT(thread_id)
        DO UPDATE SET
            title = excluded.title
        """,
        (
            thread_id,
            title
        )
    )

    conn.commit()


# ============================================================
# Get Chat Title
# ============================================================

def get_chat_title(thread_id):

    cursor = conn.execute(
        """
        SELECT title
        FROM chat_metadata
        WHERE thread_id = ?
        """,
        (
            thread_id,
        )
    )

    row = cursor.fetchone()

    if row:
        return row[0]

    return "New Conversation"


# ============================================================
# Get All Conversations
# ============================================================

def get_all_threads():

    cursor = conn.execute(
        """
        SELECT
            thread_id,
            title
        FROM chat_metadata
        ORDER BY created_at DESC
        """
    )

    return cursor.fetchall()