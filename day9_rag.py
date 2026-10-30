from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --- ЧАСТЬ 1: ПОИСК (Retrieval) ---
print("Загрузка модели эмбеддингов...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

knowledge_base = [
    "Москва — столица России, крупнейший город страны.",
    "Длина реки Волги составляет 3530 километров.",
    "Байкал — самое глубокое озеро в мире.",
    "Питон (Python) — язык программирования, созданный Гвидо ван Россумом.",
    "Эверест — высочайшая вершина Земли."
]

print("Кодирование базы знаний...")
embeddings = embedder.encode(knowledge_base)

def find_context(question):
    # Кодируем вопрос и ищем самый близкий факт
    q_emb = embedder.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    best_idx = np.argmax(scores)
    return knowledge_base[best_idx]

# --- ЧАСТЬ 2: ГЕНЕРАЦИЯ (Generation) ---
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model_name = "qwen2.5-7b-instruct"

def ask_qwen(question, context):
    # Формируем промпт, используя знания из Дня 6 (борьба с галлюцинациями)
    prompt = f"""Ответь на вопрос, используя ТОЛЬКО следующий контекст.
Если в контексте нет ответа, так и напиши "Я не знаю".

Контекст: {context}

Вопрос: {question}"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Ты — умный ассистент. Отвечай кратко и по делу."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# --- ЧАСТЬ 3: ЗАПУСК ---
print("\n--- СИСТЕМА RAG ГОТОВА ---")
user_question = "Какая длина реки Волга?" # Попробуй поменять на "Какая гора самая высокая?" или "Сколько лет Москве?"

print(f"Вопрос: {user_question}")

# 1. Ищем контекст
context = find_context(user_question)
print(f"Найденный контекст: {context}")

# 2. Отправляем в Qwen
print("Генерация ответа...")
answer = ask_qwen(user_question, context)

print("\n--- ОТВЕТ МОДЕЛИ ---")
print(answer)