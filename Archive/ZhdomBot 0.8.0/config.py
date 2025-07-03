# config.py
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

RANDOM_MESSAGES = [
    "Ребята, вы не забыли покормить уточек? 🦆",
    "Напоминаю, что сегодня отличный день! 🌞",
    "Кто-нибудь хочет поиграть в слова? 🎮",
    "Помните пить воду! 💧",
    "Как ваши успехи сегодня? 😊"
]
RANDOM_MESSAGE_INTERVAL = 3600  # 1 час по умолчанию
MIN_INTERVAL = 60

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
        "Claude??? (Glider)": {
        "model_name": "llama-3.1-8b",
        "provider": "Glider",
    },
        "gpt-4o (OpenaiChat)": {
        "model_name": "gpt-4o",
        "provider": "OpenaiChat",
    },
        "GPT-4? (PollinationsAI)": {
        "model_name": "llama-3.3-70b",
        "provider": "PollinationsAI",
    },
        "qwen-2.5-coder-32 (PollinationsAI)": {
        "model_name": "qwen-2.5-coder-32",
        "provider": "PollinationsAI",
    },
        "o1-mini (PollinationsAI)": {
        "model_name": "o1-mini",
        "provider": "PollinationsAI",
    },
        "Local Ollama (Ollama)": {
        "model_name": "Ollama",
        "provider": "Ollama",
    },
        "claude-3.5-sonnet (Liaobots)": {
        "model_name": "claude-3.5-sonnet",
        "provider": "Liaobots",
    },
        "llama-3.3-70b (LambdaChat)": {
        "model_name": "llama-3.3-70b",
        "provider": "LambdaChat",
    },
        "deepseek-v3 (LambdaChat)": {
        "model_name": "deepseek-v3",
        "provider": "LambdaChat",
    },
        "MiniMax (HailuoAI)": {
        "model_name": "MiniMax",
        "provider": "HailuoAI",
    },
        "gpt-4 (Goabror)": {
        "model_name": "gpt-4",
        "provider": "Goabror",
    },
        "gemini-1.5-flash (GizAI)": {
        "model_name": "gemini-1.5-flash",
        "provider": "GizAI",
    },
        "gemini-2.0-flash (Dynaspark)": {
        "model_name": "gemini-2.0-flash",
        "provider": "Dynaspark",
    },
        "claude-3.7-sonnet (Blackbox)": {
        "model_name": "claude-3.7-sonnet",
        "provider": "Blackbox",
    },
}

DEFAULT_MODEL = "Gemini 1.5 Pro (Websim)"
# blacklist: Liaobots GigaChat Pizzagpt DDG TypeGPT ChatGptEs


IMAGE_PROVIDERS_CONFIG = {
        "Flux (ARTA)": {
        "model_name": "flux", 
        "provider": "ARTA",
    },
        "sdxl-turbo (ImageLabs)": {
        "model_name": "sdxl-turbo", 
        "provider": "ImageLabs",
    },
        "dall-e-3 (PollinationsImage)": {
        "model_name": "dall-e-3", 
        "provider": "PollinationsImage",
    },
        "Flux (Websim)": {
        "model_name": "flux", 
        "provider": "Websim",
    },
}

DEFAULT_IMAGE_MODEL = "Flux (ARTA)" 


