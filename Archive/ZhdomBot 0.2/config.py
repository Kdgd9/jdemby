# config.py
TELEGRAM_TOKEN = "8178994454:AAG4y3LbgM3UteO2WO-nViyElmUAaPSDv5E"
MAX_CONTEXT_LENGTH = 50  # Максимальное количество пар вопрос-ответ
MAX_QUESTION_LENGTH = 5000  # Максимальная длина вопроса


MAX_CONCURRENT_IMAGE_REQUESTS = 5  # Максимум одновременных запросов
IMAGE_REQUEST_TIMEOUT = 90  # Таймаут генерации в секундах
MAX_IMAGE_PROMPT_LENGTH = 500  # Максимальная длина промпта
RATE_LIMIT_PER_USER = 3  # Максимум запросов в минуту от одного пользователя

# Настройки моделей
PROVIDERS = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "provider": "Blackbox"
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "provider": "Blackbox"
    },
    "deepseek-v3": {
        "model_name": "deepseek-v3",
        "provider": "Blackbox"
    },
    "claude-3.7-sonnet": {
        "model_name": "claude-3.7-sonnet",
        "provider": "Blackbox"
    }
}

DEFAULT_MODEL = "deepseek-v3"