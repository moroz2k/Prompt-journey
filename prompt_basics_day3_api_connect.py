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

try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            # Системный промпт (роль)
            {"role": "system", "content": "Ты — senior Python-разработчик. Отвечай кратко и только кодом."},
            # Пользовательский запрос
            {"role": "user", "content": "Напиши функцию на Python, которая проверяет, является ли строка палиндромом."}
        ],
        temperature=0.2, # Низкая температура для точного кода
        max_tokens=500
    )

    # 3. Выводим результат
    print("\n--- ОТВЕТ МОДЕЛИ ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nОшибка: {e}")
    print("Проверь, запущен ли сервер в LM Studio и совпадает ли имя модели.")
