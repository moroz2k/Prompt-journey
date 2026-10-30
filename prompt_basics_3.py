from openai import OpenAI

# 1. Подключаемся к локальному серверу LM Studio
# API-ключ может быть любым, LM Studio не требует его для локальных запросов
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio" 
)

# 2. Формируем запрос
# В LM Studio имя модели можно посмотреть на вкладке Server. 
# Обычно оно совпадает с именем папки/файла модели. 
# Если сервер выдаст ошибку "model not found", скопируй точное имя из консоли сервера.
model_name = "qwen2.5-7b-instruct" 

print("Отправляем запрос...")

text_for_model = """Москва — столица России. 
Население составляет около 13 миллионов человек.
Это очень большой город."""

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            # Системный промпт (роль)
            {"role": "system", "content": "Ты — полезный помощник. Отвечай коротко и по делу."},
            # Пользовательский запрос
            {"role": "user", "content": f"Ответь на вопрос, используя ТОЛЬКО этот текст:\n{text_for_model}\n\nВопрос: Какое население у Лондона?"}
        ],
        temperature=0.0, # Низкая температура для точного кода
        max_tokens=500
    )

    # 3. Выводим результат
    print("\n--- ОТВЕТ МОДЕЛИ ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nОшибка: {e}")
    print("Проверь, запущен ли сервер в LM Studio и совпадает ли имя модели.")