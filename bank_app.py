import streamlit as st
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.bank.graph import create_bank_graph

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Multi-Agent Bank System", page_icon="🏦")
st.title("🏦 Enterprise Multi-Agent Banking")

# Initialize the LangGraph compiled application
@st.cache_resource
def get_graph():
    return create_bank_graph()

app_graph = get_graph()

# Initialize Chat History and Thread ID in Session State
if "bank_messages" not in st.session_state:
    st.session_state.bank_messages = []
if "thread_id" not in st.session_state:
    # Generate a unique ID for this user's session memory
    st.session_state.thread_id = str(uuid.uuid4())

# Render existing chat history
for msg in st.session_state.bank_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Handle new user input
if prompt := st.chat_input("Ask about loans, rates, or branch locations..."):
    st.chat_message("user").write(prompt)
    st.session_state.bank_messages.append(HumanMessage(content=prompt))
    
    initial_state = {
        "messages": st.session_state.bank_messages
    }
    
    # Define configuration containing the thread_id for the MemorySaver
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.spinner("Processing request..."):
        # Execute graph with memory checkpointer
        result = app_graph.invoke(initial_state, config=config)
        final_msg = result["messages"][-1]
        
        st.chat_message("assistant").write(final_msg.content)
        st.session_state.bank_messages.append(final_msg)
        
        # Display the intent routing for debugging/showcase purposes
        st.caption(f"🔄 Routed via: {result.get('next_route', 'UNKNOWN')}")
