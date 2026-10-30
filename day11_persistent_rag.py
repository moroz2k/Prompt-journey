import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# --- ЧАСТЬ 1: НАСТРОЙКА БАЗЫ ---
# Используем PersistentClient, чтобы сохранять данные в папку ./chroma_db
client = chromadb.PersistentClient(path="./chroma_db")

# Наша русскоязычная модель эмбеддингов
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# get_or_create_collection: если коллекция есть — откроет, если нет — создаст
collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=ef
)

# Добавляем новые данные (если они уже есть, Chroma может выдать ошибку дубликата ID).
# Поэтому мы сначала проверим, пустая ли база.
if collection.count() == 0:
    print("База пуста. Заполняем...")
    collection.add(
        ids=["1", "2", "3", "4", "5"],
        documents=[
            "Москва — столица России, крупнейший город страны.",
            "Длина реки Волги составляет 3530 километров.",
            "Байкал — самое глубокое озеро в мире.",
            "Питон (Python) — язык программирования, созданный Гвидо ван Россумом.",
            "Эверест — высочайшая вершина Земли."
        ]
    )
else:
    print(f"База уже содержит {collection.count()} документов.")

# --- ЧАСТЬ 2: НАСТРОЙКА QWEN ---
llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model_name = "qwen2.5-7b-instruct"

# --- ЧАСТЬ 3: ФУНКЦИЯ RAG ---
def ask_rag(question):
    # 1. Ищем контекст в Chroma
    results = collection.query(query_texts=[question], n_results=1)
    context = results['documents'][0][0]
    
    # 2. Формируем промпт
    prompt = f"""Ответь на вопрос, используя ТОЛЬКО этот контекст.
Если в контексте нет ответа, напиши "Я не знаю".

Контекст: {context}
Вопрос: {question}"""
    
    # 3. Отправляем в Qwen
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Ты — полезный ассистент."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content, context

# --- ЧАСТЬ 4: ЗАПУСК ---
question = "Сколько лет Москве?"
print(f"\nВопрос: {question}")

answer, found_context = ask_rag(question)

print(f"Контекст из базы: {found_context}")
print(f"Ответ Qwen: {answer}")

# Теперь попробуй перезапустить скрипт и поменять вопрос. 
# База не будет пересоздаваться, данные останутся.