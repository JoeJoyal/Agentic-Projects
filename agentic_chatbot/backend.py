
import os
import ast
import operator
import sqlite3
import threading
from pathlib import Path
from typing import Annotated

import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Load the .env file from the same folder as this Python file
load_dotenv(BASE_DIR / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

OPENWEATHERMAP_API_KEY = (
    os.getenv("OPENWEATHERMAP_API_KEY")
    or os.getenv("OPENWEATHER_API_KEY")
)

DB_PATH = BASE_DIR / "faiss_db"
SQLITE_PATH = BASE_DIR / "chatbot.db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4

print("OpenAI key loaded:", bool(OPENAI_API_KEY))
print("Tavily key loaded:", bool(TAVILY_API_KEY))
print("Alpha Vantage key loaded:", bool(ALPHA_VANTAGE_API_KEY))
print("Weather key loaded:", bool(OPENWEATHERMAP_API_KEY))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing from the .env file.")


SYSTEM_PROMPT = """
You are a helpful Agentic AI Assistant with access to multiple tools.

Follow these instructions carefully.

1. GENERAL QUESTIONS
   - Answer general questions clearly, accurately, and concisely.
   - Answer directly when no tool is required.
   - Do not invent facts or claim to have performed actions that did not occur.

2. DOCUMENT RETRIEVAL (RAG)
   - Use `rag_tool` for questions about uploaded PDFs, manuals, reports,
     resumes, and other indexed documents.
   - Always retrieve relevant document content before answering
     document-specific questions.
   - Base document-specific answers on the retrieved content.
   - If the retrieved context does not contain the answer, explicitly
     state that the information was not found in the available documents.
   - Do not invent document facts or treat unsupported assumptions as facts.
   - Mention the source filename and page number when available.
   - If no document is indexed, explain the limitation and ask the user
     to upload and index the relevant PDF.

3. WEB SEARCH
   - Use `search_tool` for current events, recent information, and
     questions that require internet research.
   - Base search-related answers on the tool results.
   - If web search fails or is unavailable, explain the limitation.
   - Do not fabricate search results or citations.

4. CALCULATIONS
   - Use `calculator` for arithmetic and mathematical expressions
     that require calculation.
   - Present the result clearly.
   - If calculation fails, explain the error instead of guessing.

5. STOCK PRICE
   - Use `get_stock_price` when the user asks for a stock's latest
     available price or quote.
   - Use the returned tool result to answer.
   - Do not describe a quote as real-time unless the tool confirms
     that it is real-time.
   - If the stock quote is unavailable, explain why when possible.

6. STOCK PURCHASE
   - Use `purchase_stock` when the user explicitly requests a
     simulated stock purchase.
   - This tool is a mock implementation. It does not contact a
     brokerage or execute a real financial transaction.
   - Clearly describe the result as simulated.
   - Report the stock symbol, quantity, and status from the tool result.
   - Never claim that a real stock purchase was completed.

7. WEATHER
   - Use `get_weather` when the user asks about current weather
     for a city or location.
   - Present available conditions, temperature, and other returned
     weather details accurately.
   - If the weather service fails or its API key is missing,
     explain that weather information is unavailable.

8. TOOL EXECUTION
   - Select the appropriate tool based on the user's request.
   - Use multiple tools when necessary to answer a question.
   - After receiving a tool result, provide a clear and useful final answer.
   - If a tool returns an error, explain the limitation.
   - Never fabricate tool outputs or claim a tool succeeded without
     supporting evidence.

9. HUMAN-IN-THE-LOOP (HITL)
   - The purchase_stock tool requires human approval.
   - When an approval interrupt occurs, wait for the human decision.
   - Do not claim a purchase was approved unless the tool returns
     an approved result.
   - The purchase_stock tool is a mock implementation only.
   - Other tools may execute normally when selected by the model.

10. RESPONSE STYLE
    - Use simple, professional language.
    - Organize longer answers with headings or bullet points.
    - Distinguish verified information from uncertainty.
    - Stay focused on the user's actual question.
"""



# ============================================================
# 2. EMBEDDINGS: BGE-M3
# ============================================================

class BGEEmbeddings(Embeddings):
    """
    LangChain-compatible BGE-M3 embeddings.

    The SentenceTransformer model is loaded only when an embedding
    operation is first requested.
    """

    def __init__(self, model_name="BAAI/bge-m3"):
        self.model_name = model_name
        self._model = None
        self._model_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed_documents(self, texts):
        if not texts:
            return []

        model = self._get_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(self, text):
        model = self._get_model()
        vector = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vector.tolist()


embeddings = BGEEmbeddings()


# ============================================================
# 3. FAISS VECTOR STORE
# ============================================================

vector_store = None
vector_store_lock = threading.RLock()


def _faiss_index_exists():
    return (DB_PATH / "index.faiss").is_file() and (
        DB_PATH / "index.pkl"
    ).is_file()


def _load_vector_store():
    """
    Load the persisted FAISS index when it exists.
    Otherwise return None.
    """
    global vector_store

    with vector_store_lock:
        if not _faiss_index_exists():
            vector_store = None
            return None

        vector_store = FAISS.load_local(
            folder_path=str(DB_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

        return vector_store


def index_uploaded_pdf(file_path):
    """
    Load, split, embed, and index one PDF.

    Adds chunks to an existing FAISS index instead of replacing it.
    Returns the number of chunks indexed.

    NOTE: FAISS deserialization uses pickle-backed data. Only load
    an index you created yourself or otherwise trust.
    """
    global vector_store

    file_path = Path(file_path).resolve()

    # 1. Validate the file
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be indexed.")

    if file_path.stat().st_size == 0:
        raise ValueError("The uploaded PDF is empty.")

    # 2. Extract PDF pages
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    if not documents:
        raise ValueError("No pages could be loaded from this PDF.")

    # 3. Split pages into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created. The PDF may contain "
            "scanned images or no extractable text."
        )

    # 4. Store useful metadata
    for chunk in chunks:
        chunk.metadata["source"] = file_path.name

        # PyPDFLoader normally supplies zero-based page numbers.
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = (
                int(chunk.metadata["page"]) + 1
            )

    # 5. Create or update the vector store
    with vector_store_lock:
        DB_PATH.mkdir(parents=True, exist_ok=True)

        if _faiss_index_exists():
            if vector_store is None:
                vector_store = FAISS.load_local(
                    folder_path=str(DB_PATH),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True,
                )

            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
            )

        # 6. Persist the updated index
        vector_store.save_local(str(DB_PATH))

    print(f"PDF indexed successfully: {file_path.name}")
    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")

    return len(chunks)


def get_retriever():
    """
    Return a retriever over the current FAISS index.
    """
    with vector_store_lock:
        if vector_store is None:
            _load_vector_store()

        if vector_store is None:
            raise FileNotFoundError(
                f"No FAISS index found at '{DB_PATH}'. "
                "Upload and index a PDF first."
            )

        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVER_K},
        )


# Load an existing index during backend initialization.
# A missing index is valid until the user uploads a PDF.
try:
    _load_vector_store()
    if vector_store is not None:
        print(f"Loaded existing FAISS index: {DB_PATH}")
    else:
        print("No FAISS index exists yet. Upload a PDF to begin.")
except Exception as exc:
    print(
        "Could not load the existing FAISS index. "
        "Check the index files and embedding model. "
        f"Error: {type(exc).__name__}: {exc}"
    )
    vector_store = None


# ============================================================
# 4. RAG RETRIEVAL TOOL
# ============================================================

@tool
def rag_tool(query: str) -> str:
    """
    Search uploaded PDF documents for relevant information.
    Use this tool for questions about uploaded files, manuals,
    reports, resumes, and other indexed documents.
    """
    try:
        retriever = get_retriever()
        results = retriever.invoke(query)

        if not results:
            return "No relevant information was found in the indexed PDFs."

        formatted_results = []

        for index, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get(
                "page_number",
                doc.metadata.get("page"),
            )

            if page is not None:
                try:
                    page = int(page)
                    if "page_number" not in doc.metadata:
                        page += 1
                    page_label = str(page)
                except (TypeError, ValueError):
                    page_label = str(page)
            else:
                page_label = "Unknown"

            formatted_results.append(
                f"[DOC-{index}]\n"
                f"Source: {source}\n"
                f"Page: {page_label}\n"
                f"Content:\n{doc.page_content}"
            )

        return "\n\n".join(formatted_results)

    except Exception as exc:
        return (
            "Document retrieval failed. "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# 5. WEB SEARCH TOOL
# ============================================================

if TAVILY_API_KEY:
    search_tool = TavilySearchResults(
        max_results=5,
        api_key=TAVILY_API_KEY,
    )
else:
    @tool
    def search_tool(query: str) -> str:
        """Search the web for current information."""
        return (
            "Web search is unavailable because TAVILY_API_KEY "
            "is not configured."
        )


# ============================================================
# 6. SAFE CALCULATOR TOOL
# ============================================================

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate_math(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("This arithmetic operator is not allowed.")

        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)

        # Prevent excessively large exponent calculations.
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100 or abs(left) > 1_000_000:
                raise ValueError("Exponent calculation is too large.")

        return _ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("This unary operator is not allowed.")

        return _ALLOWED_OPERATORS[op_type](
            _evaluate_math(node.operand)
        )

    raise ValueError("Only basic arithmetic expressions are supported.")


@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression, such as (25 * 4) / 2."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_math(tree.body)
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculation failed: {exc}"


# ============================================================
# 7. WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city."""
    if not OPENWEATHERMAP_API_KEY:
        return (
            "Weather lookup is unavailable because "
            "OPENWEATHERMAP_API_KEY is not configured."
        )

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": OPENWEATHERMAP_API_KEY,
                "units": "metric",
            },
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]

        return (
            f"Weather in {data.get('name', city)}:\n"
            f"Conditions: {description}\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%"
        )

    except requests.RequestException as exc:
        return f"Weather service request failed: {exc}"
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return f"Could not interpret the weather response: {exc}"


# ============================================================
# 8. STOCK PRICE TOOL
# ============================================================


@tool
def get_stock_price(symbol: str) -> str:
    """Retrieve the latest available stock quote for a ticker symbol."""

    if not ALPHA_VANTAGE_API_KEY:
        return (
            "Stock lookup failed: ALPHA_VANTAGE_API_KEY is missing. "
            "Configure it in the .env file and restart the application."
        )

    symbol = symbol.strip().upper()

    if not symbol:
        return "Stock lookup failed: Please provide a valid ticker symbol."

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY,
            },
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

        # Print diagnostic information without exposing the API key.
        print(f"[Stock Debug] Symbol: {symbol}")
        print(f"[Stock Debug] Response fields: {list(data.keys())}")

        quote = data.get("Global Quote", {})
        price = quote.get("05. price")

        if price:
            return (
                f"Stock: {quote.get('01. symbol', symbol)}\n"
                f"Price: ${float(price):,.2f} USD\n"
                f"Latest trading day: "
                f"{quote.get('07. latest trading day', 'Unknown')}"
            )

        # Alpha Vantage may explain why it did not return a quote.
        message = (
            data.get("Note")
            or data.get("Information")
            or data.get("Error Message")
        )

        if message:
            print(f"[Stock Debug] API message: {message}")
            return f"Stock quote unavailable for {symbol}: {message}"

        print(f"[Stock Debug] Unexpected API response: {data}")
        return (
            f"No stock quote was returned for {symbol}. "
            "Check the ticker symbol and API response."
        )

    except requests.RequestException as exc:
        print(f"[Stock Debug] Request error: {exc}")
        return f"Stock API request failed: {type(exc).__name__}: {exc}"

    except (ValueError, TypeError) as exc:
        print(f"[Stock Debug] Response parsing error: {exc}")
        return f"Could not parse the stock API response: {exc}"



# ============================================================
# STOCK PURCHASE TOOL — MOCK IMPLEMENTATION
# ============================================================


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing shares after human approval.
    This does not execute a real brokerage transaction.
    """
    symbol = symbol.strip().upper()

    if not symbol:
        return {
            "status": "error",
            "message": "Stock symbol cannot be empty."
        }

    if quantity <= 0:
        return {
            "status": "error",
            "message": "Quantity must be greater than zero."
        }

    # Pause graph execution and request human input.
    decision = interrupt(
        f"Approve simulated purchase of {quantity} shares "
        f"of {symbol}? Enter yes or no."
    )

    if isinstance(decision, str) and decision.strip().lower() == "yes":
        return {
            "status": "approved",
            "message": (
                f"Simulated purchase approved for "
                f"{quantity} shares of {symbol}."
            ),
            "symbol": symbol,
            "quantity": quantity
        }

    return {
        "status": "cancelled",
        "message": (
            f"Simulated purchase of {quantity} shares "
            f"of {symbol} was declined."
        ),
        "symbol": symbol,
        "quantity": quantity
    }



# ============================================================
# 9. CHAT METADATA: SQLITE
# ============================================================


def _initialize_chat_metadata():
    """
    Create the chat metadata table and migrate older schemas
    without deleting existing conversation records.
    """
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(SQLITE_PATH), timeout=30) as conn:
        # Create the table if it does not exist.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_metadata (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Inspect the existing table schema.
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(chat_metadata)"
            ).fetchall()
        }

        # Migrate an older table that has no updated_at column.
        if "updated_at" not in columns:
            conn.execute("""
                ALTER TABLE chat_metadata
                ADD COLUMN updated_at TIMESTAMP
            """)

            conn.execute("""
                UPDATE chat_metadata
                SET updated_at = CURRENT_TIMESTAMP
                WHERE updated_at IS NULL
            """)

        conn.commit()


def save_chat_title(thread_id, title):
    """Save or update the display title for a conversation."""
    if not thread_id:
        raise ValueError("thread_id cannot be empty.")

    title = (title or "New Chat").strip() or "New Chat"

    _initialize_chat_metadata()

    with sqlite3.connect(str(SQLITE_PATH), timeout=30) as conn:
        conn.execute(
            """
            INSERT INTO chat_metadata (thread_id, title, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(thread_id)
            DO UPDATE SET
                title = excluded.title,
                updated_at = CURRENT_TIMESTAMP
            """,
            (str(thread_id), title),
        )
        conn.commit()


def get_chat_title(thread_id):
    """Return the saved title or None if no title is stored."""
    _initialize_chat_metadata()

    with sqlite3.connect(str(SQLITE_PATH), timeout=30) as conn:
        row = conn.execute(
            "SELECT title FROM chat_metadata WHERE thread_id = ?",
            (str(thread_id),),
        ).fetchone()

    return row[0] if row else None


def get_all_threads():
    """Return conversation thread IDs for the Streamlit sidebar."""
    _initialize_chat_metadata()

    with sqlite3.connect(str(SQLITE_PATH), timeout=30) as conn:
        rows = conn.execute(
            """
            SELECT thread_id
            FROM chat_metadata
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [row[0] for row in rows]


def delete_chat_title(thread_id):
    """Delete chat display metadata. This does not delete checkpoints."""
    _initialize_chat_metadata()

    with sqlite3.connect(str(SQLITE_PATH), timeout=30) as conn:
        conn.execute(
            "DELETE FROM chat_metadata WHERE thread_id = ?",
            (str(thread_id),),
        )
        conn.commit()


# ============================================================
# 10. LANGGRAPH AGENT
# ============================================================

tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_weather,
    rag_tool,
    purchase_stock,
]

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


def chatbot_node(state: MessagesState):
    """
    Run the model with the system prompt and current conversation.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
builder.add_edge("tools", "chatbot")


# ============================================================
# 11. SQLITE CHECKPOINTER
# ============================================================

# Keep one connection alive for the lifetime of this module.
# check_same_thread=False permits use from Streamlit's threads.
_checkpoint_connection = sqlite3.connect(
    str(SQLITE_PATH),
    check_same_thread=False,
    timeout=30,
)

checkpointer = SqliteSaver(_checkpoint_connection)
checkpointer.setup()

chatbot = builder.compile(checkpointer=checkpointer)

print("LangGraph RAG chatbot backend initialized.")



# -------------------
# CLI with HITL
# -------------------
if __name__ == "__main__":
    import uuid

    print("🤖 Agentic Chatbot CLI — HITL Enabled\n")
    print("Type 'exit' to quit.\n")

    # Use a fresh thread for testing to avoid stale checkpoints.
    thread_id = f"hitl-test-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Conversation started: {thread_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Start a new user turn in this conversation.
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )

            # Resolve any HITL interrupts before accepting another turn.
            while result.get("__interrupt__"):
                for event in result["__interrupt__"]:
                    print(f"\nHITL Approval Required: {event.value}")

                decision = input("Your decision (yes/no): ").strip().lower()

                result = chatbot.invoke(
                    Command(resume=decision),
                    config=config,
                )

            # Print the most recent assistant response with content.
            assistant_messages = [
                msg
                for msg in result.get("messages", [])
                if isinstance(msg, AIMessage)
                and msg.content
                and not msg.tool_calls
            ]

            if assistant_messages:
                print(f"\nBot: {assistant_messages[-1].content}\n")
            else:
                print("\nBot: No final assistant response was returned.\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as exc:
            print(
                f"\nChatbot error: {type(exc).__name__}: {exc}\n"
            )

