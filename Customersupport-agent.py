import os
import csv
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai.tools import tool

# LangChain imports for PDF processing and FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables (API Keys)
load_dotenv()

# ==========================================
# 1. FAISS Database Initialization
# ==========================================
@st.cache_resource(show_spinner="Initializing Database...")
def initialize_knowledge_base():
    """Reads data.pdf, chunks it, and creates a local FAISS index if it doesn't exist."""
    db_path = "faiss_index"
    pdf_path = "data.pdf"
    
    if not os.path.exists(db_path):
        if not os.path.exists(pdf_path):
            st.error(f"Error: {pdf_path} not found in the project directory.")
            return False
            
        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings and save to FAISS
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(db_path)
    return True

# ==========================================
# 2. Custom Tools Defined as Functions
# ==========================================
@tool("FAISS_RAG_Search")
def rag_search_tool(query: str) -> str:
    """Search the local FAISS database for answers to customer queries. Use this FIRST."""
    try:
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=3)
        
        if not docs:
            return "DATA_NOT_FOUND"
        
        context = "\n".join([doc.page_content for doc in docs])
        return f"Retrieved Context: {context}"
    except Exception:
        return "DATA_NOT_FOUND"

@tool("Local_CSV_Logger")
def csv_logger_tool(log_entry: str) -> str:
    """Logs the query and answer to a CSV. Input format must be: 'Query | Answer'"""
    try:
        query, answer = log_entry.split("|", 1)
    except ValueError:
        query, answer = "Unknown Query", log_entry
        
    file_exists = os.path.isfile('support_logs.csv')
    with open('support_logs.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Customer Query', 'Final Answer'])
        writer.writerow([query.strip(), answer.strip()])
    return "Log saved."

search_tool = SerperDevTool()

# ==========================================
# 3. Agent Definitions
# ==========================================
rag_agent = Agent(
    role="Knowledge Base Specialist",
    goal="Answer queries accurately using ONLY the FAISS database.",
    backstory="You strictly retrieve data. If the database context doesn't answer the question, or returns 'DATA_NOT_FOUND', you pass the query on.",
    verbose=True,
    allow_delegation=False,
    tools=[rag_search_tool] # Updated to use the function tool
)

web_agent = Agent(
    role="Web Research Fallback Specialist",
    goal="Search the web for answers ONLY IF the first agent could not find the answer.",
    backstory="You analyze the previous agent's output. If they said 'WEB_SEARCH_REQUIRED', use the web search tool to find the answer. Otherwise, pass their valid answer along untouched.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

logger_agent = Agent(
    role="Quality Assurance & Logger",
    goal="Log all queries and finalized answers to the local database.",
    backstory="You ensure a permanent record is kept before returning the final text.",
    verbose=True,
    allow_delegation=False,
    tools=[csv_logger_tool] # Updated to use the function tool
)

# ==========================================
# 4. Multi-Agent Execution Logic
# ==========================================
def process_query(customer_query: str) -> str:
    task1 = Task(
        description=f"Answer this query using the RAG tool: '{customer_query}'. If the tool returns 'DATA_NOT_FOUND', or if the context doesn't contain the answer, output EXACTLY: 'WEB_SEARCH_REQUIRED'. Do NOT hallucinate.",
        expected_output="A complete answer from the database, OR the string 'WEB_SEARCH_REQUIRED'.",
        agent=rag_agent
    )

    task2 = Task(
        description=f"If Task 1 output is 'WEB_SEARCH_REQUIRED', use web search to answer: '{customer_query}'. If Task 1 provided a valid answer, output that exact answer without using your tools.",
        expected_output="The accurate answer to the customer's query.",
        agent=web_agent
    )

    task3 = Task(
        description=f"Format this string: '{customer_query} | '. Pass it to the CSV Logger tool. Then, output ONLY the clean, final answer to show the user.",
        expected_output="Just the final formatted answer text.",
        agent=logger_agent
    )

    support_crew = Crew(
        agents=[rag_agent, web_agent, logger_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        memory=True, 
        verbose=True 
    )

    result = support_crew.kickoff()
    return str(result)

# ==========================================
# 5. Streamlit Frontend UI
# ==========================================
st.set_page_config(page_title="AI Support Team", page_icon="🎧", layout="centered")

st.markdown("""
    
""", unsafe_allow_html=True)

st.title("🎧 Multi-Agent Customer Support")
st.caption("Powered by CrewAI, RAG, and Web Fallback")

# Initialize DB on first load
db_ready = initialize_knowledge_base()

if not db_ready:
    st.stop()

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("The AI team is working on your query..."):
            try:
                response = process_query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")