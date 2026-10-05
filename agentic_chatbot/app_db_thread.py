import uuid
import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from agentic_chatbot_db_backend import chatbot, llm, get_all_threads, save_chat_title



# ============================================================
# Generate a unique UUID for each conversation
# ============================================================

def generate_thread_id():

    return str(uuid.uuid4())


# ============================================================
# Generate conversation title
# ============================================================

def generate_chat_title(user_message):

    prompt = f"""
    Generate a short topic title for the following user message.

    User message:
    {user_message}

    Rules:
    - Maximum 5 words
    - Capture the main topic
    - Make the title meaningful
    - Do not use quotation marks
    - Do not use the words "Chat" or "Conversation"
    - Return only the title
    """

    response = llm.invoke(prompt)

    return response.content.strip()


# ============================================================
# Create a completely new chat conversation
# ============================================================

def reset_chat():

    # --------------------------------------------------------
    # Generate a completely new UUID
    # --------------------------------------------------------

    new_thread_id = generate_thread_id()


    # --------------------------------------------------------
    # Make the new UUID the current conversation
    # --------------------------------------------------------

    st.session_state["thread_id"] = new_thread_id


    # --------------------------------------------------------
    # Clear messages displayed in the UI
    # --------------------------------------------------------

    st.session_state["message_history"] = []


# ============================================================
# Load a previous conversation
# ============================================================

def load_conversation(thread_id):

    # --------------------------------------------------------
    # Get saved state from LangGraph
    # --------------------------------------------------------

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )


    # --------------------------------------------------------
    # Return saved messages
    # --------------------------------------------------------

    return state.values.get(
        "messages",
        []
    )


# ============================================================
# Main application title
# ============================================================

st.title("Agentic Chatbot with LangGraph")


# ============================================================
# Initialize message history
# ============================================================

# Temporary UI state.
#
# This controls which messages are currently displayed
# on the Streamlit screen.

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


# ============================================================
# Initialize current thread ID
# ============================================================

# Each conversation gets a unique UUID.

if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


# ============================================================
# SIDEBAR - Conversation Threads
# ============================================================

st.sidebar.title("Conversation Threads")


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    # Create a new UUID
    reset_chat()

    # Refresh Streamlit
    st.rerun()


# ============================================================
# Load conversations from SQLite
# ============================================================

# SQLite is the permanent source of conversation metadata.
#
# Example:
#
# [
#     ("uuid-1", "RAG Architecture"),
#     ("uuid-2", "Kubernetes Scaling"),
#     ("uuid-3", "LangGraph Memory")
# ]

threads = get_all_threads()


# ============================================================
# DISPLAY ALL CONVERSATIONS
# ============================================================

for thread_id, title in threads:

    # --------------------------------------------------------
    # Highlight currently selected conversation
    # --------------------------------------------------------

    if thread_id == st.session_state["thread_id"]:

        button_label = f"🟢 {title}"

    else:

        button_label = title


    # --------------------------------------------------------
    # Create sidebar button
    # --------------------------------------------------------

    if st.sidebar.button(
        button_label,
        key=f"thread_{thread_id}",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Switch to selected conversation
        # ----------------------------------------------------

        st.session_state["thread_id"] = thread_id


        # ----------------------------------------------------
        # Load messages from LangGraph SQLite checkpoint
        # ----------------------------------------------------

        messages = load_conversation(
            thread_id
        )


        # ----------------------------------------------------
        # Temporary list for Streamlit UI
        # ----------------------------------------------------

        temp_message = []


        # ----------------------------------------------------
        # Convert LangChain messages to Streamlit format
        # ----------------------------------------------------

        for message in messages:

            if isinstance(
                message,
                HumanMessage
            ):

                temp_message.append({
                    "role": "user",
                    "content": message.content
                })


            elif isinstance(
                message,
                AIMessage
            ):

                temp_message.append({
                    "role": "assistant",
                    "content": message.content
                })


        # ----------------------------------------------------
        # Update Streamlit message history
        # ----------------------------------------------------

        st.session_state["message_history"] = temp_message


        # ----------------------------------------------------
        # Refresh application
        # ----------------------------------------------------

        st.rerun()


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================


# ------------------------------------------------------------
# Display messages from current conversation
# ------------------------------------------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(
        message["role"]
    ):

        st.text(
            message["content"]
        )


# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

user_input = st.chat_input(
    "Type your message here:"
)


# ------------------------------------------------------------
# Process user message
# ------------------------------------------------------------

if user_input:

    # ========================================================
    # 1. Check whether this is the first message
    # ========================================================

    is_first_message = (
        len(
            st.session_state["message_history"]
        ) == 0
    )


    # ========================================================
    # 2. Generate and save conversation title
    # ========================================================

    if is_first_message:

        # Generate title using LLM
        title = generate_chat_title(
            user_input
        )

        # Save title permanently in SQLite
        save_chat_title(
            st.session_state["thread_id"],
            title
        )


    # ========================================================
    # 3. Save user message in UI history
    # ========================================================

    st.session_state["message_history"].append({

        "role": "user",

        "content": user_input

    })


    # ========================================================
    # 4. Display user message
    # ========================================================

    with st.chat_message("user"):

        st.text(
            user_input
        )


    # ========================================================
    # 5. Create LangGraph configuration
    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    # ========================================================

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata":{
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace"

    }


    # ========================================================
    # 6. Generate AI response
    # ========================================================

    with st.chat_message("assistant"):

        ai_message = st.write_stream(

            message_chunk.content

            for message_chunk, metadata
            in chatbot.stream(

                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },

                config=CONFIG,

                stream_mode="messages"

            )

            if isinstance(
                message_chunk,
                AIMessage
            )

            and message_chunk.content
        )


    # ========================================================
    # 7. Save AI response in UI history
    # ========================================================

    st.session_state["message_history"].append({

        "role": "assistant",

        "content": ai_message

    })