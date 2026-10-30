from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Наша модель
model = ChatOpenAI(
    model="qwen2.5-7b-instruct",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.0
)

# 2. Наш инструмент (декоратор @tool делает из обычной функции - инструмент для агента)
@tool
def multiply(a: int, b: int) -> int:
    """Умножает два числа."""
    return a * b

# 3. Создаем ReAct-агента (Reason + Act)
# LangGraph сам создаст граф: модель думает -> вызывает инструмент -> модель думает снова -> финальный ответ
tools = [multiply]
agent_executor = create_react_agent(model, tools)

# 4. Запускаем
print("Спрашиваем агента...")
messages = {"messages": [("user", "Сколько будет 15 умножить на 4?")]}

# stream() позволяет видеть каждый шаг агента (как он думает)
for chunk in agent_executor.stream(messages, stream_mode="values"):
    chunk["messages"][-1].pretty_print()