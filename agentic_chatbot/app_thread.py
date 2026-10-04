import uuid
import streamlit as st
from agentic_chatbot_backend import chatbot, llm 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ============================================================
# Generate a unique UUID for each conversation
# ============================================================

def generate_thread_id():

    return str(uuid.uuid4())


# ============================================================
# Add a thread to the conversation list
# ============================================================

def add_thread(thread_id):

    # Prevent duplicate thread IDs
    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(
            thread_id
        )

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

    # Generate a completely new UUID
    new_thread_id = generate_thread_id()


    # Make the new UUID the current conversation
    st.session_state["thread_id"] = new_thread_id


    # Clear the current conversation messages
    st.session_state["message_history"] = []


    # Add the new UUID to the conversation list
    add_thread(new_thread_id)


# ============================================================
# Load a previous conversation
# ============================================================

def load_conversation(thread_id):

    # Get the saved state from LangGraph
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )


    # Return saved messages
    # If no messages exist, return an empty list
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

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ============================================================
# Initialize current thread ID
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# ============================================================
# Initialize conversation thread list
# ============================================================

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

# ============================================================
# Initialize conversation titles
# ============================================================

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    }
}

# ============================================================
# SIDEBAR - Conversation Threads
# ============================================================

# Display sidebar title
st.sidebar.title("Conversation Threads")


# ============================================================
# NEW CHAT
# ============================================================

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    # Create a new UUID and reset the conversation
    reset_chat()

    # Refresh Streamlit application
    st.rerun()


# ============================================================
# DISPLAY ALL CONVERSATIONS
# ============================================================

# Reverse the list so newest conversations appear first
for thread_id in st.session_state["chat_threads"][::-1]:

    # --------------------------------------------------------
    # Get topic/title associated with this UUID
    # --------------------------------------------------------

    title = st.session_state["chat_titles"].get(
        thread_id,
        "New Conversation"
    )


    # --------------------------------------------------------
    # Highlight currently selected conversation
    # --------------------------------------------------------

    if thread_id == st.session_state["thread_id"]:

        button_label = f"🟢 {title}"

    else:

        button_label = title


    # --------------------------------------------------------
    # Create sidebar conversation button
    # --------------------------------------------------------

    if st.sidebar.button(
        button_label,
        key=f"thread_{thread_id}",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # Set selected thread as current thread
        # ----------------------------------------------------

        st.session_state["thread_id"] = thread_id


        # ----------------------------------------------------
        # Load conversation from LangGraph
        # ----------------------------------------------------

        messages = load_conversation(
            thread_id
        )


        # ----------------------------------------------------
        # Convert LangGraph messages to Streamlit format
        # ----------------------------------------------------

        temp_message = []


        for message in messages:

            # User message
            if isinstance(
                message,
                HumanMessage
            ):

                role = "user"


            # AI message
            elif isinstance(
                message,
                AIMessage
            ):

                role = "assistant"


            # Ignore ToolMessage and other message types
            else:

                continue


            # Add converted message
            temp_message.append({

                "role": role,

                "content": message.content

            })


        # ----------------------------------------------------
        # Replace current UI conversation
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
# Display messages from the currently selected conversation
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
    # 2. Generate conversation topic
    # ========================================================

    if is_first_message:

        title = generate_chat_title(
            user_input
        )

        # Store title against current UUID
        st.session_state["chat_titles"][
            st.session_state["thread_id"]
        ] = title


    # ========================================================
    # 3. Save user message in Streamlit session
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
    # ========================================================

    CONFIG = {
        "configurable": {
            "thread_id":
                st.session_state["thread_id"]
        }
    }


    # ========================================================
    # 6. Generate AI response
    # ========================================================

    with st.chat_message("assistant"):

        ai_message = st.write_stream(

            # Return only AI message content
            message_chunk.content

            # Stream response from LangGraph
            for message_chunk, metadata
            in chatbot.stream(

                # Send latest user message
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },

                # Current UUID
                config=CONFIG,

                # Stream individual messages
                stream_mode="messages"

            )

            # Only display AI messages
            if isinstance(
                message_chunk,
                AIMessage
            )

            # Ignore empty chunks
            and message_chunk.content
        )


    # ========================================================
    # 7. Save AI response in Streamlit UI history
    # ========================================================

    st.session_state["message_history"].append({

        "role": "assistant",

        "content": ai_message

    })
 

