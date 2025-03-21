# Cybersecurity Insights Chatbot

This repository contains a RAG chatbot designed to answer cybersecurity and tech-related questions. The chatbot uses a combination of direct response and tool-based document retrieval to provide concise, contextually relevant answers. It leverages the OpenAI GPT model, vector search with Chroma, and a state graph architecture to dynamically process user queries.

## Features

- **Document Retrieval:** Uses a Chroma vector store to search and retrieve key documents on cybersecurity topics. I used Chroma vectorstore since I only need "in-memory" storage for a test application and the documents are not that heavy. If the documents requires a lot of storage (a book), then I will be using Pinecone (a cloud vector database) instead.
- **Hybrid Response System:** Determines whether to answer directly using existing knowledge or retrieve additional information via a tool.
- **State Graph Architecture:** Implements a conversational flow using LangGraph with conditional tool invocations.
- **Session Management:** Supports session-based interactions by requiring a unique session ID from the user.

## Limitation
- The code will always print retrieved docs even though the query does not require tool invocation. The assumption is we always use RAG for your queries.

## Requirements

- Python 3.11+
- Required libraries:
  - `langchain_openai`
  - `langchain_core`
  - `langchain_chroma`
  - `langgraph`
  - `python-dotenv`
- An OpenAI API key. Create a `.env` file in the root directory with:

## How to run?
1. run git clone