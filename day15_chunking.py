from pypdf import PdfReader

# 1. Извлекаем текст из PDF
reader = PdfReader("D:\\prompt\\test.pdf")
full_text = ""
for page in reader.pages:
    # Добавляем текст страницы и перенос строки
    if page.extract_text():
        full_text += page.extract_text() + "\n"

print(f"Всего символов в документе: {len(full_text)}")

# 2. Функция для нарезки текста (Chunking)
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Режет текст на куски.
    chunk_size: максимальная длина одного куска в символах.
    overlap: сколько символов из конца предыдущего куска повторить в начале следующего.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Берем кусок от start до start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Сдвигаем начало следующего куска назад на размер перекрытия
        start += (chunk_size - overlap)
        
    return chunks

# 3. Применяем нарезку
chunks = chunk_text(full_text, chunk_size=500, overlap=50)

print(f"Текст разбит на {len(chunks)} кусков.\n")

# 4. Выводим первые два куска, чтобы посмотреть, как это выглядит
for i, chunk in enumerate(chunks[:2]):
    print(f"--- КУСОК {i+1} ---")
    print(chunk)
    print("\n")