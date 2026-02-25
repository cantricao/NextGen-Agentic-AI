import streamlit as st
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.clinical.graph import create_clinical_graph

# Load environment variables (API Keys)
load_dotenv()

st.set_page_config(page_title="Clinical AI Agent", page_icon="🏥", layout="centered")
st.title("🏥 Clinical AI with Semantic Cache")
st.caption("Demonstrating Multi-tenant Security & Sub-200ms Caching")

@st.cache_resource
def get_graph():
    return create_clinical_graph()

app_graph = get_graph()

# Sidebar for Patient Context
with st.sidebar:
    st.header("🔐 Tenant Context")
    patient_id = st.text_input("Patient ID (Namespace)", value="PT-88902")
    st.markdown("---")
    st.info("💡 **Test the Cache:**\n1. Ask 'What is the treatment for flu?' (Takes ~2s)\n2. Ask it again, or ask 'How to treat influenza?' (Takes < 0.2s via Cache!)")

if "clin_messages" not in st.session_state:
    st.session_state.clin_messages = []

# Display chat history
for msg in st.session_state.clin_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Input box
if prompt := st.chat_input("Enter clinical query..."):
    st.chat_message("user").write(prompt)
    st.session_state.clin_messages.append(HumanMessage(content=prompt))
    
    initial_state = {
        "messages": st.session_state.clin_messages,
        "query": prompt,
        "patient_id": patient_id
    }
    
    with st.spinner("Analyzing..."):
        # Execute the LangGraph workflow
        # Create a configuration dictionary containing the thread_id.
	# We use patient_id as the thread_id to isolate conversational memory per patient.
	config = {"configurable": {"thread_id": patient_id}}

	# Pass the config to the invoke method to enable persistent memory
	result = app_graph.invoke(initial_state, config=config)
        final_msg = result["messages"][-1]
        
        # Display response and latency metrics
        st.chat_message("assistant").write(final_msg.content)
        
        cache_status = "✅ Hit" if result.get('cache_hit', False) else "❌ Miss"
        st.caption(f"⏱️ Latency: {result.get('response_latency', 0):.3f} seconds | Cache: {cache_status}")
        
        st.session_state.clin_messages.append(final_msg)
