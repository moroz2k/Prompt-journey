import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pypdf import PdfReader

# --- НАСТРОЙКИ ---
PDF_PATH = "D:\\prompt\\test.pdf"  # Путь к твоему PDF (поменяй, если нужно)
DB_PATH = "./chroma_db_pdf"        # Папка для базы данных
COLLECTION_NAME = "pdf_knowledge_v2"
LLM_MODEL = "qwen2.5-7b-instruct"

# --- 1. НАСТРОЙКА БАЗЫ ДАННЫХ ---
print("Инициализация базы данных...")
client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

# --- 2. ЗАГРУЗКА PDF (если база пуста) ---
if collection.count() == 0:
    print("База пуста. Читаем PDF и нарезаем текст...")
    
    if not os.path.exists(PDF_PATH):
        print(f"ОШИБКА: Файл {PDF_PATH} не найден! Проверь путь.")
        exit()
        
    reader = PdfReader(PDF_PATH)
    full_text = ""
    for page in reader.pages:
        if page.extract_text():
            full_text += page.extract_text() + "\n"
            
    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += (chunk_size - overlap)
        return chunks

    # Берем первые 50 кусков для примера (чтобы не ждать вечность)
    chunks = chunk_text(full_text)[:50]
    
    print(f"Загружаем {len(chunks)} кусков в базу... (это может занять минуту)")
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks
    )
    print("Загрузка завершена!")
else:
    print(f"База уже содержит {collection.count()} кусков. Загрузка не требуется.")

# --- 3. НАСТРОЙКА LLM ---
llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# --- 4. ИСТОРИЯ ДИАЛОГА ---
chat_history = [
    {"role": "system", "content": "Ты — полезный ассистент. Отвечай на вопросы, используя предоставленный контекст. Если ответа нет в контексте, скажи об этом."}
]

def ask_pdf_with_memory(question):
    # 1. Ищем контекст в базе
    results = collection.query(query_texts=[question], n_results=2)
    docs = results.get('documents', [[]])
    
    if docs and docs[0]:
        context = "\n...\n".join(docs[0])
    else:
        context = "Контекст не найден"
    
    # 2. Формируем запрос к модели
    user_message = f"""Используй следующий контекст из документа для ответа:
    
{context}
    
Вопрос: {question}"""
    
    # 3. Создаем временный список сообщений (история + новый вопрос)
    current_messages = chat_history + [{"role": "user", "content": user_message}]
    
    # 4. Отправляем в модель
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=current_messages,
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    
    # 5. Сохраняем в историю только чистый вопрос и ответ
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    
    return answer

# --- 5. ЦИКЛ ДИАЛОГА ---
print("\n--- ЧАТ С ДОКУМЕНТОМ ЗАПУЩЕН ---")
print("Напишите 'выход', чтобы завершить.\n")

while True:
    user_input = input("Вы: ")
    if user_input.lower() in ['выход', 'exit', 'quit']:
        print("До встречи!")
        break
        
    print("Бот думает...")
    answer = ask_pdf_with_memory(user_input)
    print(f"Бот: {answer}\n")