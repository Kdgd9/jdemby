# config.py
TELEGRAM_TOKEN = "8178994454:AAG4y3LbgM3UteO2WO-nViyElmUAaPSDv5E"
MAX_QUESTION_LENGTH = 5000  # Максимальная длина вопроса
MAX_CONTEXT_LENGTH = 50  # Обычный контекст

MAX_IMAGE_PROMPT_LENGTH = 1500  # Максимальная длина промпта

MAX_RETRIES = 10  # Максимальное количество попыток отправки
MAX_MESSAGE_PARTS = 5  # Максимальное количество частей для одного ответа

THINKING_MESSAGES = [
    "🧠 Думаю...",
    "💭 Хммм...",
    "⚙️ Бип-боп...",
    "🔍 А почему...",
    "🤔 Хммммммм...",
    "📝 Пфф...",
    "🌐 Ну и ну...",
    "⏳ Щас щас..."
]



# Настройки моделей
PROVIDERS = {
    "GPT-3.5 (Blackbox)": {
        "model_name": "gpt-4o-mini",
        "provider": "Blackbox"
    },
    "GPT-4 (Blackbox)": {
        "model_name": "deepseek-v3",
        "provider": "Blackbox"
    },
    "Gemini 1.5 Flash (Websim)": {
        "model_name": "gemini-1.5-flash",
        "provider": "Websim",
    },
    "Gemini-1.5-pro (Websim)": {
        "model_name": "gemini-1.5-pro",
        "provider": "Websim",
    },
}

DEFAULT_MODEL = "Gemini-1.5-pro (Websim)"

# blacklist: Liaobots GigaChat