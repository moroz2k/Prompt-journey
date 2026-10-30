from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Загружаем модель
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. База знаний
knowledge_base = [
    "Москва — столица России, крупнейший город страны.",
    "Длина реки Волги составляет 3530 километров.",
    "Байкал — самое глубокое озеро в мире.",
    "Питон (Python) — язык программирования, созданный Гвидо ван Россумом.",
    "Эверест — высочайшая вершина Земли."
]

# 3. Кодируем базу знаний
embeddings = model.encode(knowledge_base)

# 4. Задаем вопрос и кодируем его отдельно
question = "Какая самая высокая гора?"
question_embedding = model.encode([question]) # Оборачиваем в список, так как модель ждет массив

# 5. Считаем сходство между ВОПРОСОМ и всей БАЗОЙ
# question_embedding — это матрица 1x384, embeddings — матрица 5x384
# cosine_similarity вернет массив из 5 чисел (сходство с каждым фактом)
scores = cosine_similarity(question_embedding, embeddings)[0]

# 6. Находим индекс самого похожего факта
best_index = np.argmax(scores)

# 7. Выводим результат
print(f"Вопрос: {question}")
print(f"Самый похожий факт (сходство {scores[best_index]:.3f}):")
print(knowledge_base[best_index])