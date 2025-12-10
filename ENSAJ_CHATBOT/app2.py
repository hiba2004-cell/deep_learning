import streamlit as st
import os
import pathlib
from langchain_ollama import ChatOllama
# from langchain.chains import RetrievalQA


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.tools import Tool 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredFileLoader


# CONFIG
st.set_page_config(page_title="ENSAJ Chatbot", page_icon="🎓")
st.title("🎓 ENSAJ AI Assistant")

#os.environ["GOOGLE_API_KEY"] = "AIzaSyCin7VTQNOzkJ3LZLI7CmouZsgRwsUOYV0"


# ------------------------------
# 1️⃣ LOAD OR CREATE DATABASE
# ------------------------------
DB_DIR = "./ensaj_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_vectorstore():
    """Load existing DB or create it from URLs if missing."""
    
    if not pathlib.Path(DB_DIR).exists() or len(os.listdir(DB_DIR)) == 0:
        st.write("🧠 Building ENSAJ database (first-time setup)...")

        # urls = [
        #     "https://www.ucd.ac.ma/"
        # ]

        # loader_web = WebBaseLoader(urls)
        # documents_web = loader_web.load()

        file_path = "C:/Users/USER/OneDrive/Desktop/liste_locale.php.html"

        loader_local = UnstructuredFileLoader(file_path)
        documents_local = loader_local.load()



        vectorstore = Chroma.from_documents(
            documents_local,
            embedding=embeddings,
            persist_directory=DB_DIR
        )

        st.success("✔ ENSAJ knowledge base created!")
        return vectorstore

    else:
        st.write("📚 Loading ENSAJ knowledge base...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


# Only initialize once
if "agent" not in st.session_state:

    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # ------------------------------
    # 2️⃣ MEMORY (RAG) TOOL
    # ------------------------------
    def search_ensaj_memory(query: str):
        docs = retriever.invoke(str(query))
        return "\n\n".join([doc.page_content for doc in docs])

    rag_tool = Tool(
        name="search_ensaj_documents",
        func=search_ensaj_memory,
        description="Use for ENSAJ history, majors, departments."
    )

    # ------------------------------
    # 3️⃣ WEB SEARCH TOOL
    # ------------------------------
    # web_search = DuckDuckGoSearchRun()
    # web_tool = Tool(
    #     name="web_search", 
    #     func=web_search.invoke, 
    #     description="Use for latest news, events, updates."
    # )

    # tools = [rag_tool, web_tool]
    tools = [rag_tool]

    # ------------------------------
    # 4️⃣ CREATE AGENT
    # ------------------------------
    llm = ChatOllama(
    model="qwen3-vl:4b",  
    temperature=0,
    )
    st.session_state.agent = create_agent(llm, tools)
    st.session_state.chat_history = []
    st.rerun()
    # qa = RetrievalQA.from_chain_type(
    # llm=llm,
    # retriever=retriever,
    # chain_type="stuff",
    # )



# --------------------------------
# 5️⃣ DISPLAY CHAT
# --------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --------------------------------
# 6️⃣ HANDLE INPUT
# --------------------------------
user_input = st.chat_input("Ask me about ENSAJ...")
if user_input:

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            system_prompt = (
                "You are the ENSAJ Assistant. "
                "First use ENSAJ documents, then the internet if needed. "
                "Always respond in the user's language."
            )

            response = st.session_state.agent.invoke({
                "messages": [HumanMessage(content=system_prompt + "\nUser: " + user_input)]
            })

            bot_reply = response['messages'][-1].content
            st.markdown(bot_reply)

    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
