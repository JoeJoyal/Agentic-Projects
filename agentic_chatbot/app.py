import uuid
import hashlib
import inspect
import traceback
from pathlib import Path

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

import backend as rag_backend


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploaded_pdfs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RAG Agentic Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Agentic Chatbot with LangGraph and RAG")
st.caption(
    "Chat with PDFs, search the web, perform calculations, "
    "retrieve stock quotes, check weather, and approve gated actions."
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def extract_text(content) -> str:
    """Convert LangChain message content into displayable text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)

    return str(content) if content is not None else ""


def safe_display_filename(filename: str) -> str:
    return Path(filename or "uploaded.pdf").name


def get_graph_config(thread_id: str) -> dict:
    """Use the same thread ID for normal turns and HITL resume."""
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "rag_agentic_chat",
    }


def generate_chat_title(user_message: str) -> str:
    """Generate a short conversation title."""
    try:
        response = rag_backend.llm.invoke(
            [
                (
                    "system",
                    "Create a conversation title of no more than "
                    "5 words. Return only the title.",
                ),
                ("human", user_message[:2000]),
            ]
        )
        title = extract_text(response.content).strip().strip('"').strip("'")
        return title[:100] or "New Conversation"
    except Exception as exc:
        print(f"Title generation failed: {type(exc).__name__}: {exc}")
        return user_message.strip()[:50] or "New Conversation"


# ============================================================
# 3. CONVERSATION MANAGEMENT
# ============================================================

def reset_chat() -> None:
    """Start a new thread without deleting previous conversations."""
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_hitl"] = None


def load_conversation(thread_id: str):
    """Load messages from the LangGraph SQLite checkpoint."""
    try:
        state = rag_backend.chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        if not state:
            return []
        return state.values.get("messages", [])
    except Exception as exc:
        st.error(f"Unable to load conversation: {type(exc).__name__}: {exc}")
        return []


def restore_conversation(thread_id: str) -> None:
    """Restore visible messages and any pending HITL interrupt."""
    messages = load_conversation(thread_id)
    restored_history = []

    for message in messages:
        if isinstance(message, HumanMessage):
            content = extract_text(message.content)
            if content.strip():
                restored_history.append({"role": "user", "content": content})

        elif isinstance(message, AIMessage):
            # Do not show intermediate tool-call messages.
            if getattr(message, "tool_calls", None):
                continue

            content = extract_text(message.content)
            if content.strip():
                restored_history.append({"role": "assistant", "content": content})

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = restored_history
    st.session_state["pending_hitl"] = None

    # If this checkpoint is paused at an interrupt, restore the approval panel.
    try:
        state = rag_backend.chatbot.get_state(config=get_graph_config(thread_id))
        tasks = getattr(state, "tasks", ()) if state else ()
        for task in tasks or ():
            interrupts = getattr(task, "interrupts", ()) or ()
            if interrupts:
                event = interrupts[0]
                st.session_state["pending_hitl"] = {
                    "thread_id": thread_id,
                    "prompt": str(getattr(event, "value", event)),
                }
                return
    except Exception as exc:
        print(f"Could not restore pending HITL state: {exc}")


def get_saved_threads():
    """Return [(thread_id, title), ...] for the sidebar."""
    raw_threads = rag_backend.get_all_threads()
    threads = []

    for item in raw_threads or []:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            thread_id = str(item[0])
            title = str(item[1] or "")
        else:
            thread_id = str(item)
            try:
                title = rag_backend.get_chat_title(thread_id) or ""
            except Exception:
                title = ""

        threads.append((thread_id, title))

    return threads


def save_current_chat_title(thread_id: str, user_message: str) -> None:
    """Save a title only if this thread does not already have one."""
    try:
        if not rag_backend.get_chat_title(thread_id):
            rag_backend.save_chat_title(
                thread_id,
                generate_chat_title(user_message),
            )
    except Exception as exc:
        # A title failure should not prevent chatting.
        print(f"Unable to save chat title: {type(exc).__name__}: {exc}")


# ============================================================
# 4. PDF UPLOAD AND INDEXING
# ============================================================

def save_and_index_pdf(uploaded_file):
    """
    Save and index one uploaded PDF.
    Returns (success, original_filename, chunks_added).
    """
    original_filename = safe_display_filename(uploaded_file.name)

    if not original_filename.lower().endswith(".pdf"):
        st.error(f"{original_filename} is not a PDF file.")
        return False, original_filename, 0

    try:
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            raise ValueError("The uploaded PDF is empty.")

        content_hash = hashlib.sha256(file_bytes).hexdigest()
        stored_filename = f"{content_hash}_{original_filename}"
        file_path = UPLOAD_FOLDER / stored_filename

        if not file_path.exists():
            file_path.write_bytes(file_bytes)

        # The supplied backend accepts index_uploaded_pdf(file_path).
        index_function = rag_backend.index_uploaded_pdf
        try:
            signature = inspect.signature(index_function)
            supports_original_filename = (
                "original_filename" in signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
        except (TypeError, ValueError):
            supports_original_filename = False

        with st.spinner(f"Indexing {original_filename}..."):
            if supports_original_filename:
                result = index_function(
                    str(file_path),
                    original_filename=original_filename,
                )
            else:
                result = index_function(str(file_path))

        if isinstance(result, dict):
            chunks_added = result.get("chunks_added", result.get("chunks", 0))
        elif isinstance(result, int):
            chunks_added = result
        else:
            chunks_added = None

        if chunks_added is None:
            st.success(f"{original_filename} was processed by the backend.")
        elif chunks_added > 0:
            st.success(
                f"{original_filename} indexed successfully ({chunks_added} chunks)."
            )
        else:
            st.success(
                f"{original_filename} processed. "
                "The backend reported zero new chunks."
            )

        return True, original_filename, chunks_added

    except Exception as exc:
        st.error(
            f"Failed to index {original_filename}: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return False, original_filename, 0


# ============================================================
# 5. TOOL PROGRESS
# ============================================================

def show_tool_progress(tool_name: str, placeholder) -> None:
    """Show a friendly status while a LangChain tool is running."""
    name = (tool_name or "").lower()

    if "calculator" in name:
        message = "🔢 Calculating..."
    elif "weather" in name:
        message = "🌤️ Getting weather information..."
    elif "stock" in name:
        message = "📈 Retrieving stock price..."
    elif "rag_tool" in name:
        message = "📚 Searching the PDF knowledge base..."
    elif "search" in name or "tavily" in name:
        message = "🔎 Searching the web..."
    else:
        message = f"🔧 Executing tool: {tool_name or 'unknown'}..."

    placeholder.info(message)


def show_tool_complete(tool_name: str, placeholder) -> None:
    """Show a friendly status after a LangChain tool completes."""
    name = (tool_name or "").lower()

    if "calculator" in name:
        message = "✅ Calculation completed"
    elif "weather" in name:
        message = "✅ Weather lookup completed"
    elif "stock" in name:
        message = "✅ Stock lookup completed"
    elif "rag_tool" in name:
        message = "✅ PDF retrieval completed"
    elif "search" in name or "tavily" in name:
        message = "✅ Web search completed"
    else:
        message = f"✅ {tool_name or 'Tool'} completed"

    placeholder.success(message)


class StreamlitToolProgressHandler(BaseCallbackHandler):
    """Record tool events safely without updating Streamlit in callbacks."""

    def __init__(self):
        self.events = []
        self.active_tools = {}

    def on_tool_start(
        self, serialized, input_str, *, run_id=None, **kwargs
    ):
        tool_name = (serialized or {}).get("name", "unknown")
        key = str(run_id) if run_id is not None else tool_name

        self.active_tools[key] = tool_name
        self.events.append({
            "status": "start",
            "tool_name": tool_name,
            "run_id": key,
        })

    def on_tool_end(self, output, *, run_id=None, **kwargs):
        key = str(run_id) if run_id is not None else None
        tool_name = self.active_tools.pop(key, "unknown") if key else "unknown"

        self.events.append({
            "status": "complete",
            "tool_name": tool_name,
            "run_id": key,
        })

    def on_tool_error(self, error, *, run_id=None, **kwargs):
        key = str(run_id) if run_id is not None else None
        tool_name = self.active_tools.pop(key, "unknown") if key else "unknown"

        self.events.append({
            "status": "error",
            "tool_name": tool_name,
            "run_id": key,
            "error": str(error),
        })


# ============================================================
# 6. LANGGRAPH HITL HELPERS
# ============================================================

def get_interrupt_prompt(result):
    """Return the first LangGraph interrupt prompt, if present."""
    if not isinstance(result, dict):
        return None

    interrupts = result.get("__interrupt__", []) or []
    if not interrupts:
        return None

    event = interrupts[0]
    return str(getattr(event, "value", event))


def get_final_assistant_text(result) -> str:
    """Extract the latest final assistant message from the graph result."""
    if not isinstance(result, dict):
        return ""

    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            # Ignore AI messages that only contain tool calls.
            if getattr(message, "tool_calls", None):
                continue

            content = extract_text(message.content).strip()
            if content:
                return content

    return ""


def stream_assistant_response(user_input: str) -> None:
    """
    Invoke the graph, record tool events, and detect HITL interrupts.
    Streamlit UI updates happen on the main thread.
    """
    thread_id = st.session_state["thread_id"]
    config = get_graph_config(thread_id)

    with st.chat_message("assistant"):
        try:
            # Callback records events only; it does not update Streamlit UI.
            tool_progress_handler = StreamlitToolProgressHandler()
            config["callbacks"] = [tool_progress_handler]

            with st.spinner("Thinking and executing tools..."):
                result = rag_backend.chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )

            # Display tool events on the Streamlit thread.
            for event in tool_progress_handler.events:
                tool_name = event.get("tool_name", "unknown")
                status = event.get("status")

                if status == "start":
                    show_tool_progress(tool_name, st)
                elif status == "complete":
                    show_tool_complete(tool_name, st)
                elif status == "error":
                    st.error(
                        f"Tool execution failed: "
                        f"{event.get('error', 'Unknown error')}"
                    )

            # Check whether the graph paused for human approval.
            interrupt_prompt = get_interrupt_prompt(result)

            if interrupt_prompt is not None:
                st.session_state["pending_hitl"] = {
                    "thread_id": thread_id,
                    "prompt": interrupt_prompt,
                }
                st.info(
                    "The workflow is paused and waiting for your decision."
                )
                st.rerun()
                return

            # Display the final assistant response.
            answer = get_final_assistant_text(result)

            if answer:
                st.markdown(answer)
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": answer}
                )
            else:
                st.info(
                    "The assistant did not return a text response. "
                    "Check the backend terminal for errors."
                )

        except Exception as exc:
            st.error(
                f"Error while processing your question: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()


def resume_hitl(decision: str) -> None:
    """Resume an interrupted graph using its original thread/checkpoint."""

    pending = st.session_state.get("pending_hitl")

    if not pending:
        st.warning("There is no pending approval request.")
        return

    thread_id = pending["thread_id"]
    config = get_graph_config(thread_id)

    # Record callback events instead of updating Streamlit from worker threads.
    tool_progress_handler = StreamlitToolProgressHandler()
    config["callbacks"] = [tool_progress_handler]

    try:
        with st.spinner("Processing your decision..."):
            result = rag_backend.chatbot.invoke(
                Command(resume=decision),
                config=config,
            )

        # Display recorded tool events on the Streamlit thread.
        for event in tool_progress_handler.events:
            tool_name = event.get("tool_name", "unknown")

            if event["status"] == "start":
                show_tool_progress(tool_name, st)

            elif event["status"] == "complete":
                show_tool_complete(tool_name, st)

            elif event["status"] == "error":
                st.error(
                    f"❌ Tool {tool_name} failed: "
                    f"{event.get('error', 'Unknown error')}"
                )

        # Check whether the graph paused for another approval.
        interrupt_prompt = get_interrupt_prompt(result)

        if interrupt_prompt is not None:
            st.session_state["pending_hitl"] = {
                "thread_id": thread_id,
                "prompt": interrupt_prompt,
            }
            st.rerun()
            return

        # Display the final assistant response.
        answer = get_final_assistant_text(result)

        if answer:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": answer,
            })

        st.session_state["pending_hitl"] = None
        st.rerun()

    except Exception as exc:
        st.error(
            f"Unable to resume the workflow: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()



# ============================================================
# 6. INITIALIZE SESSION STATE
# ============================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None


# ============================================================
# 7. SIDEBAR: CONVERSATIONS AND PDF UPLOADS
# ============================================================

with st.sidebar:
    st.header("💬 Conversation Threads")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    try:
        saved_threads = get_saved_threads()
        if saved_threads:
            for saved_thread_id, title in saved_threads:
                button_label = title or "New Conversation"
                if saved_thread_id == st.session_state["thread_id"]:
                    button_label = f"🟢 {button_label}"

                if st.button(
                    button_label,
                    key=f"thread_{saved_thread_id}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if saved_thread_id == st.session_state["thread_id"]
                        else "secondary"
                    ),
                ):
                    restore_conversation(saved_thread_id)
                    st.rerun()
        else:
            st.info("No saved conversations yet.")
    except Exception as exc:
        st.error(
            f"Unable to load conversation list: {type(exc).__name__}: {exc}"
        )

    st.divider()
    st.header("📎 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if st.button("📚 Index selected PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one PDF.")
        else:
            successful_files = []
            failed_files = []

            for uploaded_file in uploaded_files:
                success, filename, _chunks_added = save_and_index_pdf(uploaded_file)
                if success:
                    successful_files.append(filename)
                else:
                    failed_files.append(filename)

            if successful_files:
                st.success(
                    f"Successfully processed {len(successful_files)} PDF(s)."
                )
            if failed_files:
                st.warning(
                    "These files could not be indexed: " + ", ".join(failed_files)
                )

    st.caption(
        "PDF files are saved in uploaded_pdfs. "
        "The backend manages the FAISS knowledge base."
    )


# ============================================================
# 8. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 9. HUMAN-IN-THE-LOOP APPROVAL PANEL
# ============================================================

pending = st.session_state.get("pending_hitl")

if pending:
    st.warning("⏸️ This workflow is waiting for your decision.")
    st.subheader("🔐 Human-in-the-Loop Approval")
    st.write(pending["prompt"])
    st.caption(
        "This stock purchase tool is a simulation only. "
        "It does not contact a brokerage or execute a real transaction."
    )

    approve_col, reject_col = st.columns(2)

    with approve_col:
        if st.button(
            "✅ Approve",
            key="hitl_approve",
            type="primary",
            use_container_width=True,
        ):
            resume_hitl("yes")

    with reject_col:
        if st.button(
            "❌ Reject",
            key="hitl_reject",
            use_container_width=True,
        ):
            resume_hitl("no")


# ============================================================
# 10. CHAT INPUT WITH PDF ATTACHMENTS
# ============================================================

chat_input = st.chat_input(
    "Ask a question or attach a PDF...",
    accept_file="multiple",
    file_type=["pdf"],
    disabled=st.session_state.get("pending_hitl") is not None,
)


# ============================================================
# 11. PROCESS QUESTION AND ATTACHED FILES
# ============================================================

if chat_input:
    user_input = (chat_input.text or "").strip()
    attached_files = list(chat_input.files or [])

    successful_files = []
    failed_files = []

    if attached_files:
        with st.status(
            "📎 Processing attached PDFs...",
            expanded=True,
        ) as upload_status:
            for uploaded_file in attached_files:
                st.write(f"Processing: {uploaded_file.name}")
                success, filename, _chunks_added = save_and_index_pdf(uploaded_file)
                if success:
                    successful_files.append(filename)
                else:
                    failed_files.append(filename)

            upload_status.update(
                label=(
                    "PDF processing completed"
                    if not failed_files
                    else "PDF processing completed with errors"
                ),
                state="complete" if not failed_files else "error",
                expanded=False,
            )

    # If a PDF is attached without a question, ask the agent to summarize it.
    if not user_input and successful_files:
        user_input = (
            "Summarize the following uploaded PDF document(s): "
            + ", ".join(successful_files)
            + ". Use rag_tool to retrieve relevant passages from the "
            "indexed documents. Include source filenames and page numbers "
            "when available. If the retrieved context is insufficient, "
            "state that clearly."
        )

    if not user_input:
        if failed_files:
            st.warning(
                "No PDF was indexed successfully. Fix the indexing error "
                "or enter a question."
            )
    else:
        if failed_files:
            st.warning(
                "Some PDFs could not be indexed and may not be searchable: "
                + ", ".join(failed_files)
            )

        current_thread_id = st.session_state["thread_id"]
        if not st.session_state["message_history"]:
            save_current_chat_title(current_thread_id, user_input)

        st.session_state["message_history"].append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)
            if successful_files:
                st.caption("📎 Processed PDFs: " + ", ".join(successful_files))

        stream_assistant_response(user_input)
