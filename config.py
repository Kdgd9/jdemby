# config.py
MAX_QUESTION_LENGTH = 5000  # Максимальная длина вопроса
MAX_CONTEXT_LENGTH = 150  # Обычный контекст (для краткосрочной памяти)

MAX_IMAGE_PROMPT_LENGTH = 1500  # Максимальная длина промпта
SAVE_INTERVAL_SECONDS = 360

MAX_PROVIDER_RETRIES = 3 # Максимальное количество попыток смены провайдера ИИ при ошибке
DEFAULT_REPLY_MODE = "original" # Режим ответа по умолчанию: "original" (на исходное сообщение) или "direct" (напрямую на команду)
DEFAULT_ULTRA_SHORT_CONTEXT_MESSAGES_COUNT = 5 # Количество последних сообщений из чата для добавления в ультра-короткий контекст при командах /ask, /long и их алиасах в группах. Установите 0, чтобы отключить по умолчанию.

# Импортируем BOT_LEGEND из отдельного файла
try:
    from bot_legend import BOT_LEGEND
except ImportError:
    print("Ошибка: Не найден файл bot_legend.py или он не содержит переменную BOT_LEGEND.")
    print("Пожалуйста, создайте файл bot_legend.py и определите в нем BOT_LEGEND.")
    BOT_LEGEND = [] # Устанавливаем пустой список в случае ошибки, чтобы избежать краха

THINKING_MESSAGES = [
    "🧠 Думаю...",
    "💭 Хммм...",
    "⚙️ Бип-боп...",
    "🔍 А почему...",
    "🤔 Хммммммм...",
    "📝 Ждёмби печатает...",
    "🌐 Ну и ну...",
    "⏳ Секунду..."
]

ERROR_RESPONSES_1 = [
    "🤔 Упс, что-то пошло не так с первого раза. Дайте мне секунду, я попробую ещё раз...",
    "🤔 Хммм, нейросеть не отвечает, попробую ешё раз через несколько секунд...",
    "🤔 LLM выдала ошибку. Одну секунду, и я попробую снова!",
    "🤔 Кажется, модель молчит. Позвольте мне повторить попытку.",
    "🤔 Языковая модель не дала ответ, но я не сдаюсь! Повторяю запрос..."
]

ERROR_RESPONSES_FINAL = [
    "🥺 Ох, кажется, нейросеть не справилась. Такое случается. Можно попробовать спросить ещё раз, либо сменить модель (/model)",
    "🥺 К сожалению, мои попытки были безуспешны. LLM выдал ошибку или промолчал. Подождёшь или сменишь модель? (/model)",
    "🥺 Я очень старался, но ничего не вышло. Приношу извинения! Пожалуйста, спросите через какое-то время или смените модель. (/model)",
    "🥺 Возможно, модель-нейросеть перегружена или вообще отвалилась. Нужно дать отдохнуть или поменять модель (/model)",
    "🥺 Ээх, не получилось. Нужно снова задать запрос модели через пару секунд или сменить её (/model)"
]

RETRY_DELAY_SECONDS = 7 # Задержка между попытками в секундах

RANDOM_MESSAGE_INTERVAL = 3600  # 1 час по умолчанию
MIN_INTERVAL = 60 # Минимальный интервал для случайных сообщений в секундах

# Настройки моделей
PROVIDERS = {
    # Gemini API
        "Gemini 2.5 Flash (Official API)": {
        "model_name": "gemini-2.5-flash",
        "provider": "OfficialGoogle", # Специальный флаг для нашего кода
        "developer_only": False,
    },
        "Gemini 2.5 Flash-Lite Preview (Official API)": {
        "model_name": "gemini-2.5-flash-lite-preview-06-17",
        "provider": "OfficialGoogle", # Специальный флаг для нашего кода
        "developer_only": False,
    },
        "Gemini 1.5 Flash-8B (Official API)": {
        "model_name": "gemini-1.5-flash-8b",
        "provider": "OfficialGoogle", # Специальный флаг для нашего кода
        "developer_only": False,
    },


    # G4F
        "ChatGPT (Chatai)": {
        "model_name": "	gpt-4o-mini",
        "provider": "Chatai",
        "developer_only": False, 
    },
        "Qwen (Cloudflare)": {
        "model_name": "qwen-1.5-7b", 
        "provider": "Cloudflare", 
        "developer_only": True, 
    },
        "LLaMA (Cloudflare)": {
        "model_name": "llama-3-8b", 
        "provider": "Cloudflare", 
        "developer_only": False, 
    },
        "Gemini (Websim)": {
        "model_name": "gemini-1.5-pro",
        "provider": "Websim",
        "developer_only": True, 
    },
        "ChatGPT (PollinationsAI)": {
        "model_name": "llama-3.3-70b",
        "provider": "PollinationsAI",
        "developer_only": False, 
    },
        "ChatGPT (Yqcloud) НЕ РЕКОМЕНДУЮ": {
        "model_name": "gpt-4",
        "provider": "Yqcloud",
        "developer_only": True, 
    },
        "Claude??? (Glider) НЕ РЕКОМЕНДУЮ": { # Убедитесь, что модель llama-3.1-8b действительно доступна через Glider
        "model_name": "llama-3.1-8b",
        "provider": "Glider",
        "developer_only": True, 
    },
        "claude-3.7-sonnet (Blackbox) ТЕСТ": {
        "model_name": "claude-3-sonnet", # Модель может называться иначе в g4f, например 'claude-3-sonnet'
        "provider": "Blackbox",
        "developer_only": True, 
    },
        "Gemini (Google) ТЕСТ": { # Убедитесь, что модель gemini-2.0 доступна. Часто это 'gemini' или 'gemini-pro'
        "model_name": "gemini-flash", # Может быть 'gemini-pro'
        "provider": "Gemini", # Или 'Google' в зависимости от g4f
        "developer_only": True, 
    },
        "GPT-4o (OpenaiChat) ТЕСТ": {
        "model_name": "gpt-4o", # Это имя провайдера, а не модели. Модель обычно 'gpt-4o' или 'gpt-4-turbo'
        "provider": "OpenaiChat",   # Фактическая модель определяется HAR файлом или настройками g4f для этого провайдера
        "developer_only": True, 
    },
        "o1 (Copilot)": {
        "model_name": "o1",
        "provider": "Copilot",
        "developer_only": True, 
    },
        "ChatGPT (Copilot)": {
        "model_name": "gpt-4",
        "provider": "Copilot",
        "developer_only": True, 
    },
        "Gemini (Free2GPT)": {
        "model_name": "gemini-1.5-pro",
        "provider": "Free2GPT",
        "developer_only": True, 
    },
        "Gemini (FreeGpt)": {
        "model_name": "gemini-1.5-pro",
        "provider": "FreeGpt",
        "developer_only": True, 
    },
        "gemini-2.5-flash (LegacyLMArena)": {
        "model_name": "gemini-2.5-flash",
        "provider": "LegacyLMArena",
        "developer_only": True, 
    },
        "grok-3 (LegacyLMArena)": {
        "model_name": "grok-3",
        "provider": "LegacyLMArena",
        "developer_only": True, 
    },
        "gpt-4o-mini (OIVSCodeSer2)": {
        "model_name": "gpt-4o-mini",
        "provider": "OIVSCodeSer2",
        "developer_only": True, 
    },
        "gpt-4.1-mini (OIVSCodeSer5)": {
        "model_name": "gpt-4.1-mini",
        "provider": "OIVSCodeSer5",
        "developer_only": True, 
    },
        "gpt-4.1-mini (OIVSCodeSer0501)": {
        "model_name": "gpt-4.1-mini",
        "provider": "OIVSCodeSer0501",
        "developer_only": True, 
    },
        "sonar (PerplexityLabs)": {
        "model_name": "sonar",
        "provider": "PerplexityLabs",
        "developer_only": True, 
    },
        "Google Gemini (TeachAnything)": {
        "model_name": "gemini-1.5-pro",
        "provider": "TeachAnything",
        "developer_only": False, 
    },
        "llama-3.2-3b (Together)": {
        "model_name": "llama-3.2-3b",
        "provider": "Together",
        "developer_only": True, 
    },
        "deepseek-r1 (Together)": {
        "model_name": "deepseek-r1",
        "provider": "Together",
        "developer_only": True, 
    },
        "qwen-2.5-vl-72b (Together)": {
        "model_name": "qwen-2.5-vl-72b",
        "provider": "Together",
        "developer_only": True, 
    },
        "gpt-4 (WeWordle)": {
        "model_name": "gpt-4",
        "provider": "WeWordle",
        "developer_only": True, 
    },
        "DuckDuckGo (DuckDuckGo)": {
        "model_name": "DuckDuckGo",
        "provider": "DuckDuckGo",
        "developer_only": True, 
    },
        "Qwen (HuggingSpace)": {
        "model_name": "qwen-2-72b",
        "provider": "HuggingSpace",
        "developer_only": False, 
    },
}


DEFAULT_MODEL = "Gemini 1.5 Flash-8B (Official API)"
# blacklist: Liaobots GigaChat Pizzagpt DDG TypeGPT ChatGptEs


IMAGE_PROVIDERS_CONFIG = {
        "Imagen 3 (Official API)": {
        "model_name": "gemini-2.0-flash-preview-image-generation", # Это гипотетическое имя, нужно будет уточнить в документации Vertex AI
        "provider": "OfficialGoogle", # Тот же флаг
        "developer_only": True, # Или True, если хотите сделать его доступным только для разработчиков
    },
        "Flux (ARTA)": {
        "model_name": "flux", 
        "provider": "ARTA",
    },
        "GPT Image (ARTA)": {
        "model_name": "gpt-image", 
        "provider": "ARTA",
    },
        "sdxl (ARTA)": {
        "model_name": "sdxl-l", 
        "provider": "ARTA",
    },
        "sdxl (ImageLabs)": {
        "model_name": "sdxl-turbo",
        "provider": "ImageLabs",
    },
        "dall-e-3 (PollinationsAI)": { # Переименовано из PollinationsImage для консистентности с текстовыми провайдерами
        "model_name": "dall-e-3",
        "provider": "PollinationsAI", # Используем тот же провайдер, если он поддерживает оба типа
    },
        "Flux (Websim)": { # Websim может не поддерживать генерацию изображений или модель 'flux'
        "model_name": "flux",
        "provider": "Websim", # Проверьте, предоставляет ли Websim генерацию изображений через g4f
    },
}

DEFAULT_IMAGE_MODEL = "Flux (ARTA)"
