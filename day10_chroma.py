import chromadb
from chromadb.utils import embedding_functions

# 1. Создаем клиент ChromaDB
# Client() создает временную базу в памяти (исчезнет после закрытия скрипта).
# Для сохранения на диск используй: chromadb.PersistentClient(path="./my_chroma_db")
client = chromadb.Client()

# 2. Настраиваем функцию эмбеддингов
# Мы используем ту же русскоязычную модель, что и в Дне 7-9.
# Без этой строки Chroma будет использовать свою дефолтную модель (которая может быть хуже для русского).
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. Создаем коллекцию (аналог таблицы в БД) и привязываем к ней нашу модель эмбеддингов
collection = client.create_collection(
    name="knowledge_base",
    embedding_function=ef
)

# 4. Добавляем документы
# Обрати внимание: Chroma сама вызовет ef и превратит текст в векторы.
# Каждому документу нужен уникальный id.
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

# 5. Задаем вопрос и ищем ответ
question = "Какая самая высокая гора?"

# query_texts автоматически закодирует вопрос и найдет ближайшие векторы в базе
results = collection.query(
    query_texts=[question],
    n_results=1  # Просим только один самый похожий результат
)

# 6. Выводим результат
# Chroma возвращает словарь. Нас интересует ключ 'documents', внутри которого список списков.
print("Вопрос:", question)
print("Найденный факт:", results['documents'][0][0])