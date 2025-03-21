# Import necessary libraries
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize chat model and embeddings
chat_model = ChatOpenAI(model = "gpt-4o-mini")
embed_model = OpenAIEmbeddings(model = "text-embedding-3-small")

# Define the documents
documents = [
    Document(
        page_content = "Cybersecurity is the practice of protecting systems and networks from attacks. It includes measures like firewalls, intrusion detection, and encryption.",
        metadata = {"source": "Document 1"},
    ),
    Document(
        page_content = "A zero-day vulnerability is an undisclosed flaw in software that attackers can exploit before the vendor issues a fix.",
        metadata = {"source": "Document 2"},
    ),
    Document(
        page_content = "Two-factor authentication (2FA) enhances security by requiring users to provide two forms of verification before gaining access.",
        metadata = {"source": "Document 3"},
    ),
    Document(
        page_content = "Machine learning models require large datasets and are often fine-tuned to improve accuracy for specific tasks.",
        metadata = {"source": "Document 4"},
    ),
    Document(
        page_content = "The CIA triad—Confidentiality, Integrity, and Availability—is a foundational concept in cybersecurity.",
        metadata = {"source": "Document 5"},
    ),
]

# Embed documents to in-memory vectorstore (ChromaDB)
vector_store = Chroma(embedding_function = embed_model)
ids = vector_store.add_documents(documents = documents)

# Define system prompt for the chatbot
SYSTEM_PROMPT = """
    You are an expert assistant whose primary goal is to answer user questions concisely.
    Don't answer very long unless needed. 
    When a user asks a question, you must first determine if the answer can be provided directly from your existing knowledge. 
    If the answer is not sufficiently clear or complete, then you should consider retrieving additional or updated information using the designated tool. 
    The tools are specifically available for questions related to cybersecurity and tech insights.

    When processing query:

    1. Direct Answer: If the question is general or can be answered directly, provide a clear and direct answer without tool usage.
    
    2. Tool Retrieval: For questions involving cybersecurity and tech insights. Use the tool only when it improves the accuracy and relevance of your answer.

    3. Responsibility: Always ensure that your answer is helpful, precise, contextually relevant, and short. 

    4. Clarity: If you are unsure whether to answer directly or use a tool, briefly explain your reasoning before proceeding.
"""

# Define the tools
def retriever_tools():
    @tool
    def document_retrieval(query: str) -> str:
        """
        Use this tool for extracting key cybersecurity and tech insights, including best practices, vulnerabilities, 2FA, and the CIA triad.
        """
        batch_results = vector_store.similarity_search(query, k = 2)
        serialized = "\n\n".join(
                (f"Content: {batch.page_content}") for batch in batch_results
            )
            
        return serialized

    all_tools = [document_retrieval]

    return all_tools

# Returns the document also
def document_retrieval_metadata(query: str):
    """
    Use this tool for extracting key cybersecurity and tech insights, including best practices, vulnerabilities, 2FA, and the CIA triad.
    """
    batch_results = vector_store.similarity_search(query, k = 2)
    
    page_contents = [result.page_content for result in batch_results]

    return {"query": query, "retrieved_docs": page_contents}

# Define class state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the agent graph
def init_agent_executor(checkpoint, tools = retriever_tools(), prompt = SYSTEM_PROMPT, llm = chat_model):
    # Define prompt
    PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder(variable_name = "messages"),
        ]
    )

    # Bind LLM, tool, and prompt
    bind_model = PROMPT | llm.bind_tools(tools)

    # Define chatbot state
    def chatbot(state: State):
        return {"messages": [bind_model.invoke(state["messages"])]}
    
    # Build the graph
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools = tools))

    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    # Compile the graph
    graph = graph_builder.compile(checkpointer = checkpoint)

    return graph

# Function to process user message
def chatbot_response(memory, session_id: str, query: str):
    graph = init_agent_executor(checkpoint= memory)
    
    # For unique session id
    configs = {"configurable": {"thread_id": session_id}}

    # Print chatbot response
    output = graph.invoke(
        {"messages": {"role": "user", "content": query}}, 
        config = configs,
    )
    ai_response = output["messages"][-1].content

    # Reformat to the required output
    retrieved_metadata = document_retrieval_metadata(query)
    retrieved_metadata["answer"] = ai_response

    return retrieved_metadata

if __name__ == "__main__":
    # Define session id
    session_id = input("Please enter your session ID: ")
    memory = MemorySaver()

    while True:
        query = input("Please enter your query: ")
        print(chatbot_response(memory, session_id, query))