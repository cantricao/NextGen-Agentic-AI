# 🚀 NextGen Agentic AI Ecosystem: Enterprise Multi-Agent & RAG Infrastructure

## 📌 Executive Summary
An enterprise-grade, multi-agent AI ecosystem demonstrating high-performance orchestration, real-time tool execution, and privacy-centric data retrieval. This repository moves beyond basic wrappers to implement **Stateful Graphs, Semantic Caching, and Multi-tenant Vector Isolation.**

Engineered for production-readiness, the architecture focuses on **Deterministic Execution**, **Sub-second Latency**, and **Zero-Hallucination** guardrails, specifically tailored for Healthcare and Financial domains.



---

## 🏗️ Core Architectures & Microservices

### 1. 🏥 Clinical RAG Agent (Privacy-First & High-Performance)
A specialized healthcare assistant designed for Electronic Health Records (EHR) with strict security boundaries.

* **Redis-Backed Semantic Caching:** Implements a vector-similarity cache layer using `sentence-transformers`. It intercepts repeated queries by calculating **Cosine Similarity (Threshold: 0.75)**, reducing API latency from **~3.0s to <100ms** and cutting LLM token costs significantly.
* **Multi-tenant Data Isolation:** Leverages **ChromaDB** with dynamic **Metadata Filtering**. Patient records are logically isolated using unique hashed identifiers, ensuring that the Retrieval phase never leaks data across different patient contexts.
* **Audit-Ready Checkpoints:** Integrated with LangGraph’s `SqliteSaver` to persist conversation states, providing a compliant audit trail for medical interactions.

### 2. 🏦 Enterprise Multi-Agent Banking System
A fully autonomous coordinator that manages financial inquiries through strict intent-based routing.

* **Stateful Agentic Routing:** Uses **LangGraph** to manage a directed acyclic graph (DAG). The **Semantic Router** classifies intents (Loan vs. FAQ) to prevent cross-domain contamination.
* **Deterministic Tool Execution:** * **Loan Agent:** Extracts financial entities to execute a deterministic `calculate_dti` tool.
    * **FAQ Agent:** Utilizes an in-memory **FAISS** vector store for blazingly fast retrieval of banking policies from CSV datasets.
* **Conversational Persistence:** Employs `MemorySaver` to maintain thread-specific context, allowing the agent to handle complex, multi-turn follow-up questions.

### 3. 🔌 Advanced Agentic Protocols (Next-Gen PoC)
Proof-of-concept implementations for standardized, decoupled AI communication located in `src/protocols/`
* **Model Context Protocol (MCP) over SSE:** A decoupled Client-Server architecture allowing LLMs to dynamically discover and execute remote tools via RPC.
* **Agent-to-Agent (A2A) Hierarchical Delegation:** Implements a Supervisor Agent (Bank Manager) that autonomously routes tasks to specialized Worker Agents (Loan Specialist, Support Specialist) to prevent context window overload.

* **Dual Client Approaches:** Includes both low-level native OpenAI SDK integration (`llm_call_mcp_sse.py`) and high-level framework orchestration (`agent_call_mcp_sse.py`).
```Plain text
+-------------------+        Server-Sent Events (SSE)         +-----------------------+
|   LLM / Agent     | --------------------------------------> |    FastMCP Server     |
|   (The Brain)     | <-------------------------------------- |  (The Infrastructure) |
|                   |        JSON-RPC Tool Execution          |                       |
+-------------------+                                         +-------+-------+-------+
                                                                      |       |
                                                                [Bank_Tools] [Vector_DB]
```
---

## 🛡️ Technical Hardening (Engineering Excellence)
* **Zero-Hallucination Guardrails:** Tools are designed with strict "I don't know" fallbacks. If the RAG similarity score is below the safety margin, the system redirects to human support.
* **Output Sanitization:** Implemented an interceptor layer to parse raw JSON tool-leaks, ensuring the UI only renders clean, professional natural language.
* **Infrastructure Testing:** Built a strict, non-LLM integration test suite (`test_mcp_manual.py`) to verify network and protocol layers before generative model deployment.

---

## 📸 System in Action (Technical Evidence)

**1. Clinical Agent: Semantic Cache Hit (Sub-100ms)**
![Clinical no cache](image/Clinical_cache.png)
![Clinical cache](image/Clinical_miss.png)
**2. Bank Agent: Autonomous Routing & Calculation**
![Bank DIT](image/Bank_DTI.png)
![Bank Location](image/Bank_Location.png)
![Bank FQA](image/Bank_QA.png)

**3. Advanced Protocol Execution (Terminal Demo)**
![llm call mcp](image/llm_call_mcp.png)
![agents call mcp](image/agents_call_mcp.png)


---

## ⚙️ Quickstart & Deployment

**1. Prerequisites:**
```bash
# Install Redis (Required for Clinical Semantic Cache)
sudo apt-get install redis-server
redis-server --daemonize yes

# Clone the repository and install dependencies
git clone [https://github.com/cantricao/NextGen-Agentic-AI.git](https://github.com/cantricao/NextGen-Agentic-AI.git)
cd NextGen-Agentic-AI
pip install -r requirements.txt
```


**2. Launch the Applications via Streamlit:** Ensure your .env contains your GOOGLE\_API\_KEY or GEMINI\_API\_KEY.

```bash
# Run Medical Agent (Multi-tenant RAG + Redis)
streamlit run clinical_app.py

# Run Banking Agent (Multi-agent Graph + FAISS)
streamlit run bank_app.py
```

**3. Execute Advanced Protocols (Terminal Demos):**

```bash
# Terminal 1: Start the FastMCP SSE Server in the background
python -m src.protocols.mcp_server

# Terminal 2: Run the Non-LLM Infrastructure Integration Test
python -m tests.protocols.test_mcp_manual

# Terminal 3: Run low-level RPC llm client
python -m src.protocols.llm_call_mcp_sse

# Terminal 4: Run high-level ReAct Agent
python -m src.protocols.mcp_agent_client
```

👨‍💻 About the Author
----------------------

**Tri Cao Can** AI Engineer & Data Analyst | Biomedical Data Science Specialist

With over 3 years of professional experience in developing machine learning models and automated data pipelines , I hold a Master of Data Analytics from QUT. My core focus lies at the intersection of computational biology, genomic data analysis, and scalable AI infrastructure. This repository reflects my passion for building secure, data-driven systems that solve complex translational challenges.

* **Email:** cantricao@gmail.com
* **LinkedIn:** [linkedin.com/in/cao-tri-can](https://www.linkedin.com/in/cao-tri-can-08188b21b/)
* **Portfolio:** [Notion Portfolio](https://cumbersome-tachometer-03f.notion.site/)
* **GitHub:** [github.com/cantricao](http://github.com/cantricao)
