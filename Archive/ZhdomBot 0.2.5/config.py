# config.py
TELEGRAM_TOKEN = "8178994454:AAG4y3LbgM3UteO2WO-nViyElmUAaPSDv5E"
MAX_CONTEXT_LENGTH = 50  # ������������ ���������� ��� ������-�����
MAX_QUESTION_LENGTH = 5000  # ������������ ����� �������


MAX_CONCURRENT_IMAGE_REQUESTS = 5  # �������� ������������� ��������
IMAGE_REQUEST_TIMEOUT = 90  # ������� ��������� � ��������
MAX_IMAGE_PROMPT_LENGTH = 500  # ������������ ����� �������
RATE_LIMIT_PER_USER = 3  # �������� �������� � ������ �� ������ ������������

# ��������� �������
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
    "gemini-1.5-flash": {
        "model_name": "gemini-1.5-flash",
        "provider": "Websim"
    }
}

DEFAULT_MODEL = "deepseek-v3"

# blacklist: Liaobots GigaChat