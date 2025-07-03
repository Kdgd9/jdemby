# config.py
TELEGRAM_TOKEN = "8178994454:AAG4y3LbgM3UteO2WO-nViyElmUAaPSDv5E"
MAX_QUESTION_LENGTH = 5000  # Максимальная длина вопроса
MAX_CONTEXT_LENGTH = 50  # Обычный контекст
MAX_LONG_CONTEXT_LENGTH = 500  # Расширенный контекст

MAX_CONCURRENT_IMAGE_REQUESTS = 5  # Максимум одновременных запросов
IMAGE_REQUEST_TIMEOUT = 90  # Таймаут генерации в секундах
MAX_IMAGE_PROMPT_LENGTH = 1500  # Максимальная длина промпта
RATE_LIMIT_PER_USER = 3  # Максимум запросов в минуту от одного пользователя

SHORT_TERM_MAX_LENGTH = 50  # Пар сообщений в краткосрочной памяти
LONG_TERM_MAX_ITEMS = 500   # Максимум записей на пользователя
EMOTION_KEYWORDS = {
    "радость": ["рад", "счастлив", "ура", "класс", "круто"],
    "грусть": ["грустно", "печаль", "плохо", "обидно"],
    "гнев": ["злой", "бесит", "ненавижу", "разозлился"]
}

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
        "provider": "Websim"
    }
}

DEFAULT_MODEL = "GPT-4 (Blackbox)"

# blacklist: Liaobots GigaChat