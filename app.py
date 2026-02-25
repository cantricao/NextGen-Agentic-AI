import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.graph import create_bank_graph

# Load environment variables
load_dotenv()

# Initialize Streamlit Page
st.set_page_config(page_title="NextGen Bank AI", page_icon="🏦", layout="centered")
st.title("🏦 NextGen Multi-Agent Bank")
st.caption("Powered by LangGraph & Gemini 2.5 Flash")

# Initialize Graph and Session State
@st.cache_resource
def get_graph():
    return create_bank_graph()

app_graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for User Context Settings
with st.sidebar:
    st.header("⚙️ User Context")
    user_location = st.text_input("Current Location", value="Hanoi, Vietnam")
    st.markdown("---")
    st.info("Try asking:\n- 'I earn 5000 and my debt is 2000. Calculate my DTI.'\n- 'Where is the nearest branch?'")

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    st.chat_message("user").write(prompt)
    
    # Construct input state
    input_message = HumanMessage(content=prompt)
    st.session_state.messages.append(input_message)
    
    initial_state = {
        "messages": st.session_state.messages,
        "user_location": user_location,
        "num_steps": 0
    }
    
    # Invoke Graph
    with st.spinner("Processing through Agentic Workflow..."):
        try:
            # Stream the execution to show which agent is working
            for output in app_graph.stream(initial_state, {"recursion_limit": 15}):
                for node_name, state_update in output.items():
                    # Just to show logs in Streamlit
                    pass 
            
            # The final state contains the updated messages
            final_state = app_graph.invoke(initial_state)
            final_messages = final_state.get("messages", [])
            
            # Extract the last AI message
            if final_messages:
                last_msg = final_messages[-1]
                if isinstance(last_msg, AIMessage):
                    st.chat_message("assistant").write(last_msg.content)
                    st.session_state.messages.append(last_msg)
                    
        except Exception as e:
            st.error(f"Agent Loop Error: {str(e)}")
