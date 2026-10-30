import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model_name = "qwen2.5-7b-instruct"
client = chromadb.PersistentClient(path="./chroma_db")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=ef
)

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


# 1. Наша реальная Python-функция
def add_numbers(a: int, b: int) -> int:
    """Складывает два числа."""
    return a + b

def ask_rag(question):
    # 1. Ищем контекст в Chroma
    results = collection.query(query_texts=[question], n_results=1)
    return results['documents'][0][0]

# 2. Описание функции для модели (формат JSON Schema)
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Складывает два числа и возвращает результат.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "Первое число"},
                    "b": {"type": "integer", "description": "Второе число"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Ищет информацию в базе данных и возвращает результат",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Вопрос пользователя"}
                    },
                    "required": ["question"]
                }
            }
        }
]

# 3. Запрос пользователя
messages = [
    {"role": "user", "content": "Сколько будет 10 плюс 20, и какая гора самая высокая?"}
]

print("Отправляем запрос в Qwen...")

# 4. Первый вызов: даем модели инструменты
response = llm_client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=tools,
    tool_choice="auto" # Разрешаем модели самой решить, вызывать ли инструмент
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

# 5. Проверяем, решила ли модель вызвать функцию
if tool_calls:
    print("Модель решила вызвать инструмент!")
    messages.append(response_message) # Добавляем ответ модели в историю
    
    # Обрабатываем каждый вызов (их может быть несколько)
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"Вызов функции: {function_name} с аргументами {function_args}")
        
        # Вызываем нашу реальную Python-функцию
        if function_name == "add_numbers":
            function_response = add_numbers(
                a=function_args.get("a"),
                b=function_args.get("b")
            )
        elif function_name == "search_knowledge_base":
            function_response = ask_rag(question=function_args.get("question"))

        messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": str(function_response),
                                    }
                                )


    
    # 7. Второй вызов: модель получает результат функции и пишет финальный ответ
    second_response = llm_client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    print("\n--- ФИНАЛЬНЫЙ ОТВЕТ ---")
    print(second_response.choices[0].message.content)
else:
    print("Модель не стала вызывать функцию. Ответ:")
    print(response_message.content)