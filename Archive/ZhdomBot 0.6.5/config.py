# config.py
TELEGRAM_TOKEN = "8178994454:AAG4y3LbgM3UteO2WO-nViyElmUAaPSDv5E"
MAX_QUESTION_LENGTH = 5000  # Максимальная длина вопроса
MAX_CONTEXT_LENGTH = 150  # Обычный контекст

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
        "Gemini 1.5 Flash (Websim)": {
        "model_name": "gemini-1.5-flash",
        "provider": "Websim",
    },
        "Gemini 1.5 Pro (Websim)": {
        "model_name": "gemini-1.5-pro",
        "provider": "Websim",
    },
        "GPT-4 (Yqcloud)": {
        "model_name": "gpt-4",
        "provider": "Yqcloud",
    },
        "Claude (Glider)": {
        "model_name": "llama-3.1-8b",
        "provider": "Glider",
    },
}

DEFAULT_MODEL = "Gemini 1.5 Pro (Websim)"

# blacklist: Liaobots GigaChat Pizzagpt DDG TypeGPT ChatGptEs