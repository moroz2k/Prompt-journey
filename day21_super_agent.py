import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent

# --- 1. МОДЕЛЬ ---
model = ChatOpenAI(
    model="qwen2.5-7b-instruct",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.0
)

# --- 2. БАЗА ЗНАНИЙ (как в Дне 19) ---
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
DB_DIR = "./chroma_langchain_db"

# (Предполагаем, что база уже создана в Дне 19. Если нет — раскомментируй блок ниже)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 3. СОЗДАЕМ ИНСТРУМЕНТЫ ---

# Инструмент 1: Поиск по PDF
@tool
def search_pdf(query: str) -> str:
    """ОБЯЗАТЕЛЬНО используй этот инструмент для поиска любых фактов, слов или контекста из PDF-документа (например, про Lorem ipsum). 
    Если пользователь спрашивает про содержание документа, ты ДОЛЖЕН вызвать эту функцию, а не отвечать из своих знаний."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Инструмент 2: Калькулятор
@tool
def multiply(a: int, b: int) -> int:
    """Умножает два числа."""
    return a * b

# --- 4. СОЗДАЕМ АГЕНТА ---
tools = [search_pdf, multiply]
agent_executor = create_react_agent(model, tools)

# --- 5. ЗАПУСК ---
print("--- СУПЕР-АГЕНТ ГОТОВ ---")
question = "Какие слова идут после Lorem ipsum? И сколько будет 15 умножить на 4?"
print(f"Вопрос: {question}\n")

# stream() покажет все шаги агента
for chunk in agent_executor.stream({"messages": [("user", question)]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()