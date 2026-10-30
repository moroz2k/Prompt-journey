import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. НАСТРОЙКА МОДЕЛИ (Qwen) ---
model = ChatOpenAI(
    model="qwen2.5-7b-instruct",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.0
)

# --- 2. НАСТРОЙКА ЭМБЕДДИНГОВ ---
# Используем ту же русскоязычную модель, но теперь через обертку LangChain
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

DB_DIR = "./chroma_langchain_db"

# --- 3. ЗАГРУЗКА И НАРЕЗКА PDF (если базы нет) ---
if not os.path.exists(DB_DIR):
    print("База не найдена. Читаем PDF и нарезаем текст...")
    loader = PyPDFLoader("D:\\prompt\\test.pdf") # Укажи свой путь
    docs = loader.load()
    
    # LangChain сам режет текст умнее: он старается не разрывать абзацы
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    print(f"Создаем базу из {len(splits)} кусков...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
else:
    print("Загружаем существующую базу...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Превращаем базу в "искатель" (Retriever)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 4. ПРОМПТ ДЛЯ RAG ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — полезный ассистент. Отвечай на вопрос, используя ТОЛЬКО предоставленный контекст. Если ответа нет, скажи 'Я не знаю'."),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}")
])

# --- 5. СБОРКА ЦЕПОЧКИ ---
# RunnablePassthrough берет вопрос пользователя и передает его дальше
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | model 
    | StrOutputParser()
)

# --- 6. ЗАПУСК ---
print("\n--- RAG НА LANGCHAIN ГОТОВ ---")
# Попробуй задать вопрос, на который есть ответ в тексте
question = "Какие слова идут после Lorem ipsum?"
print(f"Вопрос: {question}")

answer = rag_chain.invoke(question)
print(f"\nОтвет: {answer}")