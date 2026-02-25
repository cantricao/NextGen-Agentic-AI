# 🚀 NextGen Agentic AI Ecosystem

This repository showcases a collection of advanced, production-ready AI Agent architectures. It demonstrates the evolution from simple LLM calls to complex, multi-agent systems capable of autonomous reasoning, tool utilization, and inter-agent communication using the latest protocols (MCP, Google A2A).

## 🧠 Key Architectures & Projects

### 1. Enterprise Multi-Agent Banking System
An autonomous banking assistant built with **LangGraph**.
* **Routing Logic:** Dynamically categorizes user intent (FAQ, Transaction, Support).
* **Tool Calling:** Integrates RAG tools (Vector DB) for policy retrieval and Web Search for real-time competitor analysis.
* **Cyclic Reasoning:** Uses state graphs to manage complex, multi-step customer interactions.

### 2. Clinical Agent with Semantic Caching 
A privacy-first RAG architecture designed for the healthcare domain.
* **Semantic Caching:** Implemented vector-based caching (`sentence-transformers`) to reduce latency to <200ms and cut API costs by ~60% for repetitive queries.
* **Multi-tenant Security:** Isolated cache namespaces per `patient_id` to strictly prevent context leakage.

### 3. Bleeding-Edge Protocols (MCP & A2A)
Exploring the forefront of agent connectivity and orchestration:
* **Model Context Protocol (MCP):** Implementation of an MCP server utilizing OpenAI Agents and Gemini 2.5 to standardize tool and data source connections. 
* **Agent-to-Agent (A2A):** Demonstrating dynamic agent discovery and communication using Google's A2A framework. 

### 4. Open-Source Function Calling 
* Engineered function-calling capabilities for open-source models (running locally), enabling offline autonomous actions without relying on proprietary APIs like OpenAI.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain, Google A2A
* **Models:** Gemini 2.5 Pro/Flash, OpenAI, Open-source LLMs
* **Protocols:** MCP (Model Context Protocol)
* **Performance:** Semantic Caching, Long-term Context Memory

---
*Built with passion for transforming innovative ideas into real-world, AI-powered products.*
