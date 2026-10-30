import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model_name = "qwen2.5-7b-instruct"

# 1. Наша реальная Python-функция
def add_numbers(a: int, b: int) -> int:
    """Складывает два числа."""
    return a + b

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
    }
]

# 3. Запрос пользователя
messages = [
    {"role": "user", "content": "Сколько будет 123 плюс 456?"}
]

print("Отправляем запрос в Qwen...")

# 4. Первый вызов: даем модели инструменты
response = client.chat.completions.create(
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
            
            # 6. Отправляем результат обратно в модель
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
            )
    
    # 7. Второй вызов: модель получает результат функции и пишет финальный ответ
    second_response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    print("\n--- ФИНАЛЬНЫЙ ОТВЕТ ---")
    print(second_response.choices[0].message.content)
else:
    print("Модель не стала вызывать функцию. Ответ:")
    print(response_message.content)