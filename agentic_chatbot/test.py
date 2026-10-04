import uuid
import streamlit as st
from agentic_chatbot_backend import chatbot, llm 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Generate a unique thread ID for each session or conversation
def generate_thread_id():
    return str(uuid.uuid4())

# Add a new thread ID to the session state if it doesn't exist
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times 
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Create a completely new chat conversation  
# def reset_chat():

#     # Generate and assign a new thread ID
#     st.session_state["thread_id"] = generate_thread_id()

#     # Clear the current chat messages from the UI
#     st.session_state["message_history"] = []

#     # Add the new thread to the conversation list
#     add_thread(st.session_state["thread_id"])

def reset_chat():

    # Generate a completely new UUID
    new_thread_id = generate_thread_id()

    # Make it the current conversation
    st.session_state["thread_id"] = new_thread_id

    # Clear messages from the current UI
    st.session_state["message_history"] = []

    # Add new thread to the thread list
    add_thread(new_thread_id)

# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(config={"configurable":{"thread_id": thread_id}})
    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])

# Display the main application title
st.title("Agentic Chatbot with LangGraph")

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Create message_history when the app runs for the first time
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Create a list for storing all conversation thread IDs  
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []


# ============= Sidebar threading features ==============

# display the sidebar title
st.sidebar.title("Conversation Threads")

# Create a new chat button for starting a new conversation
if st.sidebar.button("➕ New Chat", use_container_width=True):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()

# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    # Get the conversation topic/title
    title = st.session_state["chat_titles"].get(thread_id, "New Conversation")

    # Highlight the currently selected conversation
    if thread_id == st.session_state["thread_id"]:
        button_label = f"🟢 {title}"
    else:
        button_label = title

    # Create sider button using the topic instead of UUID
    if st.sidebar.button(
        button_label,
        key=f"thread_{thread_id}"
    ):
        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages into streamlit's required message format
        temp_message = []

        # Loop through all saved message
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such a ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_message.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_message

        # Return the application to display the loaded messages
        st.rerun()


# ============= Main chat interface ====================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])

# Create the chat input box for user messages
user_input = st.chat_input("Type your message here:")

# Run this block after the user submits a message
if user_input:

    # Save the users message in Streamlit session state
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # Display the user's message in the chat interface
    with st.chat_message('user'):
        st.text(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this thread ID to manage the conversation state and history
    CONFIG = {'configurable': {'thread_id': st.session_state["thread_id"]}}

    # Create the assistant chat-message container for displaying the AI's response
    with st.chat_message('assistant'):

        # Stream the assistant response token by token 
        ai_message = st.write_stream(

            # Return only the content of AI message chunks  
            message_chunk.content 

            # Stream message from the LangGraph chatbot   
            for message_chunk, metadata in chatbot.stream(
                    # Send the latest user message to the chatbot for processing
                    {'messages': [HumanMessage(content=user_input)]}, 
                    # use the current conversation thread ID for state management
                    config=CONFIG,
                    # Stream individual message chunks instead of the entire message at once
                    stream_mode = 'messages'
                )

                # Display only AI messages
                # This prevents tool and user messages from appearing
                if isinstance(message_chunk, AIMessage)
            )
    
    # Save the complete assistant response in Streamlit session state 
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message}) 
 

