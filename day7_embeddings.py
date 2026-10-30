from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Загружаем легкую модель для эмбеддингов (она скачается автоматически, весит ~100 МБ)
# Это не генеративная LLM, она только переводит текст в векторы.
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Список фраз для теста
sentences = [
    "Собака бежит по парку",
    "Пёс гуляет на улице",
    "Кошка спит на диване",
    "Я программирую на Python"
]

# 3. Превращаем текст в векторы (массивы чисел)
embeddings = model.encode(sentences)

# 4. Считаем косинусное сходство (от -1 до 1, где 1 — это полное совпадение по смыслу)
from sklearn.metrics.pairwise import cosine_similarity

# Сравниваем первую фразу (Собака бежит...) со всеми остальными
scores = cosine_similarity([embeddings[0]], embeddings)[0]

# 5. Выводим результат
for i, score in enumerate(scores):
    print(f"Сходство с фразой '{sentences[i]}': {score:.3f}")