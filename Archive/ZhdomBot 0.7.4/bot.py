# -*- coding: utf-8 -*-
# COPYRIGHT ZhdomDev
# Optimized version

__version__ = "0.7.4" # Версия бота

import logging
import re
import asyncio
import random
import time
import html
from typing import Dict, List, Optional, Tuple, Any

# Aiogram imports
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, CommandObject
from aiogram.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, KeyboardButton, Message
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.enums import ChatType, ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter, TelegramBadRequest, TelegramAPIError
from aiogram.client.default import DefaultBotProperties # Для установки parse_mode по умолчанию

# g4f imports
import g4f
from g4f.Provider import BaseProvider

# Configuration import
# --- Важно: Убедитесь, что файлы config.py и secrets.py существуют и содержат необходимые переменные ---
try:
    from config import (
        PROVIDERS, # Словарь с конфигурацией провайдеров и моделей
        DEFAULT_MODEL, # Модель по умолчанию
        MAX_CONTEXT_LENGTH, # Макс. длина контекста (в парах сообщений)
        MAX_QUESTION_LENGTH, # Макс. длина вопроса пользователя
        MAX_IMAGE_PROMPT_LENGTH, # Макс. длина промпта для изображения
        THINKING_MESSAGES, # Сообщения о том, что бот "думает"
        RANDOM_MESSAGE_INTERVAL, # Интервал случайных сообщений по умолчанию (в секундах)
        MIN_INTERVAL # Минимальный интервал для случайных сообщений (в секундах)
        # RANDOM_MESSAGES - больше не используется, генерируется LLM
    )
except ImportError as e:
    print(f"Ошибка импорта из config.py: {e}")
    print("Пожалуйста, убедитесь, что файл config.py существует и содержит необходимые переменные (кроме TELEGRAM_TOKEN).")
    exit(1)

try:
    from secrets import TELEGRAM_TOKEN
except ImportError:
    print("Ошибка: Не найден файл secrets.py или он не содержит переменную TELEGRAM_TOKEN.")
    print("Пожалуйста, создайте файл secrets.py в той же директории, что и bot.py, и определите в нем TELEGRAM_TOKEN.")
    exit(1)

# --- Конец блока импорта и проверки config.py / secrets.py ---


# --- Logging Setup ---
# Настройка логирования для отслеживания работы бота
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Provider Mapping ---
# Сопоставление имен провайдеров из конфига с классами g4f
# Используем getattr для динамического получения классов провайдеров
PROVIDER_MAP: Dict[str, type[BaseProvider]] = {}
if PROVIDERS: # Проверяем, что словарь PROVIDERS не пустой
    try:
        PROVIDER_MAP = {
            name: getattr(g4f.Provider, name)
            for name in {prov_data["provider"] for prov_data in PROVIDERS.values()}
            if hasattr(g4f.Provider, name)
        }
        logger.info(f"Loaded providers: {list(PROVIDER_MAP.keys())}")

        # Проверка наличия всех необходимых провайдеров
        missing_providers = {
            prov_data["provider"] for prov_data in PROVIDERS.values()
            if prov_data["provider"] not in PROVIDER_MAP
        }
        if missing_providers:
            logger.error(f"Missing provider classes in g4f library: {missing_providers}")
            # Можно добавить exit(1) если критично
    except AttributeError as e:
        logger.error(f"Error accessing provider attributes in g4f: {e}. Please check g4f installation and provider names in config.py.")
    except Exception as e:
         logger.error(f"An unexpected error occurred during provider mapping: {e}")

else:
    logger.warning("PROVIDERS dictionary in config.py is empty or not defined. No providers loaded.")


# --- Utility Functions ---

def is_group_chat(message: types.Message) -> bool:
    """Проверяет, является ли чат групповым."""
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}

def escape_html(text: str) -> str:
    """Экранирует основные HTML-спецсимволы."""
    if not isinstance(text, str): # Добавим проверку типа
        logger.warning(f"escape_html received non-string input: {type(text)}. Converting to string.")
        text = str(text)
    return html.escape(text, quote=False) # quote=False чтобы не экранировать кавычки

def convert_markdown_to_html(text: str) -> str:
    """
    Преобразует ограниченный Markdown-подобный синтаксис в HTML для Telegram.
    Поддерживает: **жирный**, __курсив__, ~~зачеркнутый~~, `код`, ```блок кода```, [ссылки](url).
    Важно: Эта функция вызывается *после* экранирования HTML в safe_send_message.
    """
    if not isinstance(text, str): # Добавим проверку типа
        logger.warning(f"convert_markdown_to_html received non-string input: {type(text)}. Returning as is.")
        return str(text) # Возвращаем как строку

    # 1. Экранируем HTML символы СНАЧАЛА, чтобы теги внутри Markdown не ломали разметку
    text = escape_html(text)

    # 2. Обрабатываем блоки кода (```...```) -> <pre><code>...</code></pre>
    # Оборачиваем в <pre> и <code> для лучшей семантики и возможного рендеринга
    # Используем нежадный поиск (.*?) и флаг re.DOTALL для многострочности
    # Добавляем экранирование содержимого блока кода еще раз на всякий случай
    text = re.sub(r'```(.*?)```', lambda m: f'<pre><code>{escape_html(m.group(1))}</code></pre>', text, flags=re.DOTALL)

    # 3. Обрабатываем инлайн код (`...`) -> <code>...</code>
    # Нежадный поиск ([^`]+) - один или более символов, не являющихся `
    # Экранируем содержимое на всякий случай
    text = re.sub(r'`([^`]+?)`', lambda m: f'<code>{escape_html(m.group(1))}</code>', text)

    # 4. Обрабатываем жирный (**...**) -> <b>...</b>
    # Нежадный поиск (.*?)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    # 5. Обрабатываем курсив (__...__) -> <i>...</i>
    # Нежадный поиск (.*?)
    text = re.sub(r'__(.+?)__', r'<i>\1</i>', text)

    # 6. Обрабатываем зачеркнутый (~~...~~) -> <s>...</s>
    # Нежадный поиск (.*?)
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)

    # 7. Обрабатываем ссылки [text](url) -> <a href="url">text</a>
    # Экранируем URL перед вставкой в href
    def replace_link(match):
        link_text = match.group(1)
        url = match.group(2)
        # Дополнительно экранируем кавычки и сам URL для безопасности
        safe_url = escape_html(url.replace('"', '&quot;'))
        # Текст ссылки уже экранирован на шаге 1
        return f'<a href="{safe_url}">{link_text}</a>'
    text = re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)

    return text


async def safe_send_message(
    target: Message, # Используем Message напрямую для ответа/отправки
    text: str,
    parse_mode: Optional[str] = ParseMode.HTML, # По умолчанию HTML
    reply_to_message: bool = True, # Отвечать ли на сообщение по умолчанию
    **kwargs: Any
) -> Optional[Message]:
    """
    Безопасная отправка сообщений с обработкой ошибок форматирования и длины.
    """
    max_length = 4096 # Максимальная длина сообщения в Telegram
    sent_message = None

    if not isinstance(text, str):
        logger.warning(f"safe_send_message received non-string: {type(text)}. Converting.")
        text = str(text)

    # Определяем метод отправки: ответить или просто отправить в чат
    # Проверяем наличие метода reply, на случай если target - неполный объект
    send_method = target.reply if hasattr(target, 'reply') and reply_to_message else target.answer

    try:
        # 1. Попытка отправить с указанным parse_mode (обычно HTML)
        # Конвертация Markdown в HTML происходит здесь, перед отправкой
        processed_text = convert_markdown_to_html(text) if parse_mode == ParseMode.HTML else text

        # Разбиваем на части, если текст слишком длинный
        for i in range(0, len(processed_text), max_length):
            chunk = processed_text[i:i + max_length]
            # Убираем disable_web_page_preview=True если не нужно
            sent_message = await send_method(
                chunk,
                parse_mode=parse_mode,
                **kwargs # Передаем остальные аргументы (например, reply_markup)
            )
            # Небольшая задержка между частями, чтобы избежать флуда
            if len(processed_text) > max_length:
                await asyncio.sleep(0.2) # Чуть увеличим задержку
        return sent_message # Возвращаем последнее отправленное сообщение

    except (TelegramBadRequest, ValueError) as e:
        # 2. Ошибка форматирования или другая BadRequest ошибка
        logger.warning(f"Formatting/Request error with parse_mode={parse_mode}: {e}. Text: {text[:100]}...")
        # Пробуем отправить без parse_mode
        try:
            # Используем исходный текст, НЕ processed_text
            plain_text = text
            logger.info("Retrying to send message without parse_mode.")
            for i in range(0, len(plain_text), max_length):
                chunk = plain_text[i:i + max_length]
                sent_message = await send_method(
                    chunk,
                    parse_mode=None, # Отправляем без форматирования
                    **kwargs
                )
                if len(plain_text) > max_length:
                    await asyncio.sleep(0.2)
            return sent_message
        except Exception as e_final:
            logger.exception(f"Failed to send message even without parse_mode: {e_final}")
            # Попытка отправить сообщение об ошибке (может не сработать)
            try:
                error_report = f"⚠️ Не удалось отправить отформатированный ответ.\nОшибка: {escape_html(str(e))}\nПопытка без форматирования тоже не удалась: {escape_html(str(e_final))}"
                await send_method(
                    error_report[:max_length], # Обрезаем сообщение об ошибке
                    parse_mode=None
                )
            except Exception:
                logger.error("Failed even to send the error message.")
            return None # Не удалось отправить

    except TelegramRetryAfter as e:
        logger.warning(f"Rate limit hit for chat {target.chat.id}. Retrying after {e.retry_after}s.")
        await asyncio.sleep(e.retry_after)
        # Повторяем попытку после ожидания
        return await safe_send_message(target, text, parse_mode, reply_to_message, **kwargs)

    except TelegramForbiddenError:
        logger.error(f"Bot is forbidden to perform action in chat {target.chat.id}. Maybe kicked or blocked.")
        # Здесь можно добавить логику удаления чата из активных, если нужно
        if target.chat.id in storage.active_chats:
             del storage.active_chats[target.chat.id]
             logger.info(f"Removed chat {target.chat.id} from active random messaging.")
        return None

    except TelegramAPIError as e:
        logger.exception(f"Unhandled Telegram API Error in safe_send_message: {e}")
        try:
            await send_method(f"⚠️ Произошла ошибка Telegram API: {escape_html(str(e))}", parse_mode=None)
        except Exception:
            logger.error("Failed to send Telegram API error message.")
        return None

    except Exception as e:
        logger.exception(f"Unexpected error in safe_send_message: {e}")
        try:
            await send_method(f"⚠️ Произошла непредвиденная ошибка: {escape_html(str(e))}", parse_mode=None)
        except Exception:
            logger.error("Failed to send unexpected error message.")
        return None


# --- Context Management ---

class ChatContext:
    """Управляет контекстом диалога для одного чата/пользователя."""
    def __init__(self, max_short_term: int = MAX_CONTEXT_LENGTH):
        # Убедимся, что max_short_term - это число и не отрицательное
        self.max_short_term = max(0, int(max_short_term)) if isinstance(max_short_term, (int, float, str)) and str(max_short_term).isdigit() else 10 # Значение по умолчанию, если некорректно
        if self.max_short_term == 0:
             logger.warning("MAX_CONTEXT_LENGTH is 0, short-term memory will be disabled.")

        # Системные инструкции и "легенда" бота
        self.legend: List[Dict[str, str]] = [
            {"role": "system", "content": "Your name is Ждёмби."},
            {"role": "system", "content": "You cannot mention or depict harm to ducks, geese, or swans in your responses."},
            {"role": "system", "content": (
                "Format your response using ONLY the following markdown-style tags:\n"
                "**bold text** for bold\n"
                "__italic text__ for italic\n"
                "~~strikethrough~~ for strikethrough\n"
                "[hyperlink text](URL) for links\n"
                "`inline code` for inline code\n"
                "```\ncode block\n``` for code blocks (use language identifier if known, e.g., ```python)\n"
                "Do not use any other formatting like lists, headers, blockquotes etc.\n"
                "Escape special characters like _*[]()~`>#+-=|{}.! within the text if they are not part of the formatting tags."
            )}
        ]
        self.short_term: List[Dict[str, str]] = [] # Краткосрочная память (очищается /clear)
        self.long_term: List[Dict[str, str]] = []  # Долгосрочная память (очищается /fullreset)

    def add_message(self, role: str, content: str, memory_type: str = "short"):
        """Добавляет сообщение в указанный тип памяти."""
        if not isinstance(content, str): # Проверка типа контента
             logger.warning(f"Attempted to add non-string content to context: {type(content)}. Converting.")
             content = str(content)
        if not content: # Не добавляем пустые сообщения
             logger.warning(f"Attempted to add empty message to {memory_type} memory for role {role}.")
             return

        message = {"role": role, "content": content}
        if memory_type == "short":
            if self.max_short_term > 0: # Добавляем только если память включена
                self.short_term.append(message)
                # Обрезаем старые сообщения, если превышен лимит КОРОТКОСРОЧНОЙ памяти
                # Умножаем на 2, т.к. лимит считается в парах (вопрос-ответ)
                overflow = len(self.short_term) - self.max_short_term * 2
                if overflow > 0:
                    self.short_term = self.short_term[overflow:]
        elif memory_type == "long":
            self.long_term.append(message)
            # Долгосрочную память не обрезаем автоматически

    def get_full_context(self) -> List[Dict[str, str]]:
        """Возвращает полный контекст для передачи модели."""
        return self.legend + self.long_term + self.short_term

    def get_long_context(self) -> List[Dict[str, str]]:
        """Возвращает только легенду и долгосрочную память."""
        return self.legend + self.long_term

    def clear_short(self):
        """Очищает краткосрочную память."""
        self.short_term = []
        logger.info("Short-term context cleared.")

    def clear_all(self):
        """Очищает всю память (краткосрочную и долгосрочную)."""
        self.short_term = []
        self.long_term = []
        logger.info("Full context cleared (short-term and long-term).")


# --- Global Storage ---

class Storage:
    """Хранит контексты чатов, настройки моделей и активные чаты для случайных сообщений."""
    def __init__(self):
        # Используем chat_id как ключ для групп и user_id для личных чатов
        self.contexts: Dict[int, ChatContext] = {}
        self.models: Dict[int, str] = {} # chat_id -> model_key (e.g., "Default")
        # chat_id -> {'active': bool, 'interval': int, 'last_sent': float}
        self.active_chats: Dict[int, Dict[str, Any]] = {}

    def _get_key(self, chat_id: int, user_id: int, is_group: bool) -> int:
        """Возвращает ключ для хранения данных (chat_id для групп, user_id для ЛС)."""
        # Важно: Используем user_id для личных чатов, чтобы контекст был персональным
        return chat_id if is_group else user_id

    def get_context(self, chat_id: int, user_id: int, is_group: bool) -> ChatContext:
        """Получает или создает контекст для чата/пользователя."""
        key = self._get_key(chat_id, user_id, is_group)
        if key not in self.contexts:
            self.contexts[key] = ChatContext()
            logger.info(f"Created new context for key {key} (chat={chat_id}, user={user_id}, group={is_group})")
        return self.contexts[key]

    def clear_short(self, chat_id: int, user_id: int, is_group: bool):
        """Очищает краткосрочную память для чата/пользователя."""
        key = self._get_key(chat_id, user_id, is_group)
        if key in self.contexts:
            self.contexts[key].clear_short()
        else:
             logger.warning(f"Attempted to clear short context for non-existent key {key}")


    def clear_all(self, chat_id: int, user_id: int, is_group: bool):
        """Очищает всю память для чата/пользователя."""
        key = self._get_key(chat_id, user_id, is_group)
        if key in self.contexts:
            self.contexts[key].clear_all()
        else:
             logger.warning(f"Attempted to clear full context for non-existent key {key}")

    def get_available_models(self) -> List[str]:
        """Возвращает список доступных имен моделей из конфига."""
        return list(PROVIDERS.keys()) if PROVIDERS else []

    def get_model_key(self, chat_id: int) -> str:
        """Возвращает ключ текущей модели для чата (или дефолтную)."""
        # Используем chat_id как ключ для настроек модели, даже в ЛС,
        # чтобы настройки модели были общими для чата/ЛС, а не для пользователя.
        # Если нужна модель на пользователя, нужно изменить ключ здесь и в set_model.
        return self.models.get(chat_id, DEFAULT_MODEL)

    def set_model(self, chat_id: int, model_key: str):
        """Устанавливает модель для чата."""
        if PROVIDERS and model_key in PROVIDERS:
            self.models[chat_id] = model_key
            logger.info(f"Model for chat {chat_id} set to '{model_key}'")
        else:
            logger.warning(f"Attempted to set invalid model '{model_key}' for chat {chat_id}. Available: {list(PROVIDERS.keys())}")

    def init_chat_settings(self, chat_id: int) -> Dict[str, Any]:
        """Инициализирует или возвращает настройки для случайных сообщений в чате."""
        if chat_id not in self.active_chats:
             # Убедимся, что RANDOM_MESSAGE_INTERVAL - число
            interval = RANDOM_MESSAGE_INTERVAL if isinstance(RANDOM_MESSAGE_INTERVAL, (int, float)) and RANDOM_MESSAGE_INTERVAL > 0 else 3600 # Дефолт - 1 час
            self.active_chats[chat_id] = {
                'active': False,
                'interval': interval,
                'last_sent': 0.0
            }
            logger.info(f"Initialized random message settings for chat {chat_id} with interval {interval}s")
        return self.active_chats[chat_id]

# --- Bot Initialization ---
# Устанавливаем HTML как режим парсинга по умолчанию для всего бота
default_properties = DefaultBotProperties(parse_mode=ParseMode.HTML)
try:
    # Проверяем наличие токена (импортированного из secrets.py)
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is not defined in secrets.py")
    bot = Bot(token=TELEGRAM_TOKEN, default=default_properties)
except ValueError as e: # Более конкретная ошибка для токена
    logger.critical(f"Invalid or missing Telegram token: {e}")
    exit(1)
except Exception as e:
     logger.critical(f"Failed to initialize Bot: {e}")
     exit(1)


dp = Dispatcher()
storage = Storage() # Создаем экземпляр хранилища
router = Router() # Создаем роутер для команд и сообщений
BOT_USERNAME: Optional[str] = None # Имя пользователя бота будет получено при запуске

# --- Finite State Machine (FSM) ---
class Form(StatesGroup):
    """Состояния для процесса выбора модели."""
    model_selection = State()


# --- Background Tasks ---

async def send_random_messages():
    """Периодически отправляет случайные сообщения в активные чаты."""
    await asyncio.sleep(15) # Небольшая задержка перед первым запуском
    logger.info("Starting random message task...")
    while True:
        await asyncio.sleep(30)  # Проверяем каждые 30 секунд
        current_time = time.time()

        # Создаем копию ключей, чтобы избежать проблем при удалении во время итерации
        active_chat_ids = list(storage.active_chats.keys())

        for chat_id in active_chat_ids:
            # Проверяем, существует ли чат еще в словаре (мог быть удален из-за ошибки)
            if chat_id not in storage.active_chats:
                continue

            settings = storage.active_chats[chat_id]

            if not settings['active']:
                continue

            # Проверяем интервал
            interval = settings.get('interval', RANDOM_MESSAGE_INTERVAL)
            if not isinstance(interval, (int, float)) or interval <= 0:
                logger.warning(f"Invalid interval {interval} for chat {chat_id}. Using default {RANDOM_MESSAGE_INTERVAL}s.")
                interval = RANDOM_MESSAGE_INTERVAL
                settings['interval'] = interval # Исправляем в настройках

            time_since_last = current_time - settings.get('last_sent', 0.0)

            if time_since_last >= interval:
                logger.info(f"Attempting to send random message to chat {chat_id}")
                try:
                    # Получаем контекст группы (случайные сообщения только для групп)
                    # Передаем user_id=0 т.к. это не от конкретного пользователя
                    context = storage.get_context(chat_id=chat_id, user_id=0, is_group=True)
                    model_key = storage.get_model_key(chat_id) # Получаем ключ модели для чата

                    if not PROVIDERS:
                        logger.error("PROVIDERS dictionary is empty. Cannot select model for random message.")
                        continue # Пропускаем чат, если нет провайдеров

                    if model_key not in PROVIDERS:
                        logger.warning(f"Chat {chat_id} has invalid model '{model_key}', using default '{DEFAULT_MODEL}'")
                        model_key = DEFAULT_MODEL
                        # Проверяем, существует ли дефолтная модель
                        if model_key not in PROVIDERS:
                            logger.error(f"Default model '{DEFAULT_MODEL}' not found in PROVIDERS. Skipping chat {chat_id}.")
                            continue

                    provider_config = PROVIDERS[model_key]
                    provider_name = provider_config.get("provider")
                    model_identifier = provider_config.get("model_name") # Имя/ID модели

                    if not provider_name or not model_identifier:
                         logger.error(f"Incomplete provider config for model '{model_key}' in chat {chat_id}. Missing 'provider' or 'model_name'. Skipping.")
                         continue

                    if provider_name not in PROVIDER_MAP:
                         logger.error(f"Provider class '{provider_name}' configured for model '{model_key}' not found in PROVIDER_MAP. Skipping chat {chat_id}.")
                         continue

                    provider_class = PROVIDER_MAP[provider_name]

                    # Формируем промпт с базовым (долгосрочным) контекстом
                    messages = context.get_long_context() + [{
                        "role": "user",
                        "content": "Придумай случайное, короткое, интересное или забавное сообщение для поддержания беседы в групповом чате. Будь креативным и естественным, как если бы ты был участником чата. Не представляйся и не говори, что ты бот. Твой ответ должен быть только текстом сообщения, без каких-либо предисловий или объяснений."
                    }]

                    logger.debug(f"Sending request to {provider_name} ({model_identifier}) for random message in chat {chat_id}")

                    response = await g4f.ChatCompletion.create_async(
                        model=model_identifier,
                        messages=messages,
                        provider=provider_class,
                        timeout=30 # Устанавливаем таймаут для запроса
                    )

                    response_text = str(response).strip() # Преобразуем в строку на всякий случай
                    if response_text:
                        # Используем chat_id напрямую для отправки, так как у нас нет объекта Message
                        # Создаем временный объект Message для использования safe_send_message
                        # Важно: нужен chat.id и chat.type
                        temp_message_stub = types.Message(
                            message_id=0, # Не важно
                            date=int(time.time()), # Текущее время
                            chat=types.Chat(id=chat_id, type=ChatType.GROUP), # Указываем ID и тип чата
                            from_user=None # Сообщение от бота, нет пользователя
                        )
                        await safe_send_message(
                            target=temp_message_stub,
                            text=response_text,
                            parse_mode=ParseMode.HTML, # Ответ модели должен быть уже в нужном формате
                            reply_to_message=False # Не отвечаем ни на какое сообщение
                        )
                        settings['last_sent'] = current_time
                        logger.info(f"Sent random message to {chat_id} via {model_key}: {response_text[:50]}...")
                    else:
                        logger.warning(f"Received empty random message response from {model_key} for chat {chat_id}")

                except TelegramForbiddenError:
                    # Ошибка уже логируется и чат удаляется в safe_send_message
                    logger.warning(f"Bot forbidden in chat {chat_id}, removing from active list.")
                    if chat_id in storage.active_chats:
                        del storage.active_chats[chat_id]
                except TelegramRetryAfter as e:
                    logger.warning(f"Rate limit for chat {chat_id} during random message, retry after {e.retry_after}s")
                    await asyncio.sleep(e.retry_after) # Ждем и попробуем в след. цикле
                except asyncio.TimeoutError:
                    logger.error(f"Timeout error getting random message from {model_key} for chat {chat_id}")
                    settings['last_sent'] = current_time + 60 # Попробуем снова через минуту
                except Exception as e:
                    logger.exception(f"Error sending random message to chat {chat_id}: {e}")
                    # Добавляем небольшую паузу после ошибки для этого чата
                    settings['last_sent'] = current_time + 60 # Попробуем снова через минуту
                    await asyncio.sleep(5) # Небольшая общая пауза

# --- Core Logic ---

async def process_question(
    message: types.Message,
    question: str,
    memory_type: str = "short" # "short" или "long"
):
    """Обрабатывает вопрос пользователя, взаимодействует с LLM и отправляет ответ."""
    chat_id = message.chat.id
    # Проверяем наличие from_user
    if not message.from_user:
        logger.error(f"Received message without from_user in chat {chat_id}. Cannot process.")
        return
    user_id = message.from_user.id
    is_group = is_group_chat(message)

    # Получаем контекст и настройки модели
    context = storage.get_context(chat_id, user_id, is_group)
    model_key = storage.get_model_key(chat_id) # Ключ модели для этого чата

    if not PROVIDERS:
        logger.error("PROVIDERS dictionary is empty. Cannot process question.")
        await safe_send_message(message, "⚠️ Ошибка конфигурации: Список моделей пуст.", reply_to_message=True, parse_mode=None)
        return

    if model_key not in PROVIDERS:
        logger.error(f"Invalid model key '{model_key}' found for chat {chat_id}. Reverting to default '{DEFAULT_MODEL}'.")
        model_key = DEFAULT_MODEL
        if model_key not in PROVIDERS:
             logger.error(f"Default model '{DEFAULT_MODEL}' not found in PROVIDERS. Cannot process question.")
             await safe_send_message(message, f"⚠️ Ошибка конфигурации: Модель по умолчанию '{DEFAULT_MODEL}' не найдена.", reply_to_message=True, parse_mode=None)
             return
        storage.set_model(chat_id, model_key) # Сохраняем дефолтную модель для чата

    provider_config = PROVIDERS[model_key]
    provider_name = provider_config.get("provider")
    model_identifier = provider_config.get("model_name")

    if not provider_name or not model_identifier:
         logger.error(f"Incomplete provider config for model '{model_key}' in chat {chat_id}. Missing 'provider' or 'model_name'.")
         await safe_send_message(message, f"⚠️ Ошибка конфигурации для модели '{model_key}'.", reply_to_message=True, parse_mode=None)
         return

    if provider_name not in PROVIDER_MAP:
         logger.error(f"Provider class '{provider_name}' configured for model '{model_key}' not found in PROVIDER_MAP.")
         await safe_send_message(message, f"⚠️ Ошибка конфигурации: провайдер {provider_name} не найден.", reply_to_message=True, parse_mode=None)
         return

    provider_class = PROVIDER_MAP[provider_name]

    # Добавляем вопрос пользователя в контекст
    context.add_message("user", question, memory_type)

    # Отправляем сообщение "Думаю..."
    status_msg = await safe_send_message(
        message,
        random.choice(THINKING_MESSAGES) if THINKING_MESSAGES else "🤔 Думаю...",
        reply_to_message=True,
        parse_mode=None # Просто текст
    )

    try:
        logger.info(f"Processing question from user {user_id} in chat {chat_id} using {model_key} ({provider_name}/{model_identifier})")
        logger.debug(f"Context size: short={len(context.short_term)}, long={len(context.long_term)}")

        # Запрос к LLM
        response = await g4f.ChatCompletion.create_async(
            model=model_identifier,
            messages=context.get_full_context(),
            provider=provider_class,
            timeout=120 # Таймаут для ответа модели (в секундах)
        )

        full_response = str(response).strip() # Преобразуем в строку

        if not full_response:
            logger.warning(f"Received empty response from {model_key} for user {user_id} in chat {chat_id}")
            full_response = "⚠️ Получен пустой ответ от модели."
            # Не добавляем пустой ответ ассистента в контекст

        else:
            # Добавляем ответ ассистента в тот же тип памяти, что и вопрос
            # Ограничиваем длину сохраняемого ответа
            context.add_message("assistant", full_response[:8192], memory_type)
            logger.info(f"Received response from {model_key} for user {user_id} in chat {chat_id}. Length: {len(full_response)}")

        # Удаляем сообщение "Думаю..." (если оно было успешно отправлено)
        if status_msg:
            try:
                await status_msg.delete()
            except TelegramAPIError as e:
                 logger.warning(f"Could not delete status message {status_msg.message_id}: {e}")


        # Отправляем ответ пользователю (safe_send_message сама разобьет на части)
        await safe_send_message(
            target=message,
            text=full_response,
            parse_mode=ParseMode.HTML, # Ожидаем, что модель вернет Markdown, который конвертируется в HTML
            reply_to_message=True
        )

    except asyncio.TimeoutError:
         logger.error(f"Timeout error processing question with {model_key} for user {user_id} in chat {chat_id}")
         if status_msg: # Попытаемся удалить "Думаю" и здесь
             try: await status_msg.delete()
             except TelegramAPIError as e_del: logger.warning(f"Could not delete status message {status_msg.message_id} after timeout: {e_del}")
         await safe_send_message(message, f"⚠️ Модель ({model_key}) не ответила вовремя.", reply_to_message=True, parse_mode=None)
         # Удаляем последний вопрос пользователя из контекста, так как ответа не было
         if memory_type == "short" and context.short_term and context.short_term[-1]["role"] == "user":
              context.short_term.pop()
              logger.info("Removed user question from short-term context after timeout.")
         elif memory_type == "long" and context.long_term and context.long_term[-1]["role"] == "user":
              context.long_term.pop()
              logger.info("Removed user question from long-term context after timeout.")


    except Exception as e:
        logger.exception(f"Error processing question with {model_key} for user {user_id} in chat {chat_id}")
        if status_msg: # Попытаемся удалить "Думаю" и при других ошибках
             try: await status_msg.delete()
             except TelegramAPIError as e_del: logger.warning(f"Could not delete status message {status_msg.message_id} after error: {e_del}")
        await safe_send_message(message, f"⚠️ Произошла ошибка при обработке вашего запроса: {escape_html(str(e))}", reply_to_message=True, parse_mode=None)
        # Удаляем последний вопрос пользователя из контекста при ошибке
        if memory_type == "short" and context.short_term and context.short_term[-1]["role"] == "user":
             context.short_term.pop()
             logger.info("Removed user question from short-term context after error.")
        elif memory_type == "long" and context.long_term and context.long_term[-1]["role"] == "user":
             context.long_term.pop()
             logger.info("Removed user question from long-term context after error.")


async def handle_command_with_question(
    message: types.Message,
    command: CommandObject,
    allow_reply: bool, # Разрешено ли отвечать на другое сообщение?
    memory_type: str   # "short" или "long"
):
    """
    Общий обработчик для команд /ask и /long, которые принимают текст вопроса.
    Формирует текст запроса к LLM, включая информацию об авторах при ответе.
    """
    # Проверяем наличие from_user
    if not message.from_user:
        logger.error(f"Received command message without from_user in chat {message.chat.id}. Cannot process.")
        return

    current_user = message.from_user
    # Используем get_full_name() для более полного имени, если доступно, иначе first_name
    current_user_name = escape_html(current_user.full_name if hasattr(current_user, 'full_name') and current_user.full_name else current_user.first_name)
    full_question_text: Optional[str] = None

    # 1. Обработка ответа на сообщение
    if allow_reply and message.reply_to_message:
        original_msg = message.reply_to_message
        # Проверяем наличие from_user в исходном сообщении
        if not original_msg.from_user:
            logger.warning(f"Reply target message {original_msg.message_id} has no from_user. Using 'Unknown'.")
            original_author_name = "Unknown"
        else:
            original_author = original_msg.from_user
            original_author_name = escape_html(original_author.full_name if hasattr(original_author, 'full_name') and original_author.full_name else original_author.first_name)

        # Проверка, что есть текст в оригинальном сообщении
        original_text_content = original_msg.text or original_msg.caption
        if not original_text_content:
            await safe_send_message(message, "❌ Исходное сообщение не содержит текста.", reply_to_message=True, parse_mode=None)
            return

        # Экранируем и обрезаем текст исходного сообщения
        original_text_escaped = escape_html(original_text_content.strip())[:MAX_QUESTION_LENGTH]

        # Текст из текущей команды (может быть пустым)
        # Экранируем и обрезаем
        command_text = escape_html(command.args.strip())[:MAX_QUESTION_LENGTH] if command.args else ""

        # Формируем структурированный запрос
        full_question_text = (
            f"[Сообщение, на которое отвечают] (Автор: {original_author_name}): {original_text_escaped}\n"
            f"[Ответ/комментарий] (Автор: {current_user_name}): {command_text}"
        )
        # Добавляем пояснение для LLM
        full_question_text += "\n[Инструкция для LLM]: Ответь на комментарий пользователя, учитывая контекст сообщения, на которое он отвечает."


    # 2. Обработка обычного вопроса (без ответа на другое сообщение)
    else:
        raw_question = command.args.strip() if command.args else None
        if not raw_question:
            # Если команда /long без аргументов, это не ошибка, просто информируем
            if memory_type == "long":
                 await safe_send_message(message, "ℹ️ Укажите текст, который нужно добавить в долгосрочную память.", reply_to_message=True, parse_mode=None)
                 return
            # А для /ask - это ошибка
            else:
                await safe_send_message(message, "❌ Пожалуйста, укажите ваш вопрос после команды.", reply_to_message=True, parse_mode=None)
                return

        # Экранируем и обрезаем вопрос
        question_text_escaped = escape_html(raw_question[:MAX_QUESTION_LENGTH])
        full_question_text = f"[Вопрос] (Автор: {current_user_name}): {question_text_escaped}"
        # Добавляем пояснение для LLM
        full_question_text += "\n[Инструкция для LLM]: Ответь на вопрос пользователя."


    # 3. Валидация общей длины сформированного запроса (увеличим лимит немного)
    # Умножаем на 1.5 т.к. добавили имена авторов и инструкции
    max_combined_length = int(MAX_QUESTION_LENGTH * 1.5)
    if len(full_question_text) > max_combined_length:
        await safe_send_message(
            message,
            f"❌ Ваш запрос слишком длинный (>{max_combined_length} символов с учетом контекста). Пожалуйста, сократите его.",
            reply_to_message=True,
            parse_mode=None
        )
        return

    # 4. Вызов основного обработчика запросов
    await process_question(message, full_question_text, memory_type)


# --- Command Handlers ---

@router.message(Command("start", "help", "старт", "помощь"))
async def cmd_start(message: types.Message):
    """Обработчик команд /start и /help."""
    # Используем f-string для подстановки версии
    help_text = (
        f"👋 Привет! Я <b>Ждёмби-бот</b> v{__version__}\n\n"
        "Я умею отвечать на вопросы, генерировать изображения и просто болтать в чате.\n\n"
        "<b>Основные команды:</b>\n"
        "• <code>/ask [ваш вопрос]</code> - Задать вопрос боту (учитывает краткосрочную память).\n"
        "   <i>(Можно также ответить на сообщение командой /ask)</i>\n"
        "• <code>/long [текст]</code> - Добавить информацию в долгосрочную память бота (не очищается /clear).\n"
        "• <code>/image [описание]</code> - Сгенерировать изображение по текстовому описанию.\n"
        "• <code>/clear</code> - Очистить краткосрочную память (историю диалога).\n"
        "• <code>/fullreset</code> - Полностью очистить всю память (краткосрочную и долгосрочную).\n"
        "• <code>/model</code> - Выбрать другую модель ИИ для ответов.\n\n"
        "<b>Для групповых чатов:</b>\n"
        "• <code>/sglypa [минуты]</code> - Включить/выключить режим случайных сообщений с заданным интервалом (например, <code>/sglypa 30</code> для 30 минут). Без аргумента - переключает режим.\n"
        "• Просто <b>упомяните меня</b> (@{BOT_USERNAME if BOT_USERNAME else 'ИмяБота'}) в начале сообщения, чтобы задать вопрос.\n\n"
        "<i>Подсказка: Долгосрочная память полезна для задания общего контекста или инструкций боту.</i>"
    )
    # Отправляем с parse_mode=HTML, так как используем HTML теги
    await safe_send_message(message, help_text, reply_to_message=False, parse_mode=ParseMode.HTML)

# Используем общий обработчик для /ask и его псевдонимов
@router.message(Command("ask", "аск", "bot", "бот", "ждёмби", "zhdomby", "ждемби", "zhdyomby"))
async def cmd_ask(message: types.Message, command: CommandObject):
    """Обработчик команды /ask и её псевдонимов."""
    await handle_command_with_question(
        message=message,
        command=command,
        allow_reply=True,    # /ask может отвечать на сообщения
        memory_type="short"  # Использует краткосрочную память
    )

@router.message(Command("long", "лонг"))
async def cmd_long(message: types.Message, command: CommandObject):
    """Обработчик команды /long."""
    await handle_command_with_question(
        message=message,
        command=command,
        allow_reply=False,   # /long не отвечает на сообщения, только добавляет контекст
        memory_type="long"   # Использует долгосрочную память
    )

@router.message(Command("clear"))
async def cmd_clear(message: types.Message):
    """Очищает краткосрочную память."""
    if not message.from_user: return # Игнорируем если нет пользователя
    is_group = is_group_chat(message)
    storage.clear_short(message.chat.id, message.from_user.id, is_group)
    await safe_send_message(message, "🧽 Краткосрочный контекст очищен!", reply_to_message=False, parse_mode=None)

@router.message(Command("fullreset"))
async def cmd_fullreset(message: types.Message):
    """Очищает всю память (краткосрочную и долгосрочную)."""
    if not message.from_user: return # Игнорируем если нет пользователя
    is_group = is_group_chat(message)
    storage.clear_all(message.chat.id, message.from_user.id, is_group)
    await safe_send_message(message, "♻️ Вся память (краткосрочная и долгосрочная) очищена!", reply_to_message=False, parse_mode=None)

@router.message(Command("model"))
async def cmd_model(message: types.Message, state: FSMContext):
    """Начинает процесс выбора модели ИИ."""
    available_models = storage.get_available_models()
    if not available_models:
         await safe_send_message(message, "❌ Нет доступных моделей для выбора.", reply_to_message=False, parse_mode=None)
         return

    builder = ReplyKeyboardBuilder()
    current_model = storage.get_model_key(message.chat.id)

    for model_key in available_models:
        # Добавляем маркер к текущей выбранной модели
        text = f"✅ {model_key}" if model_key == current_model else model_key
        builder.add(KeyboardButton(text=text))

    builder.adjust(2) # Располагаем кнопки по 2 в ряд

    await safe_send_message(
        message,
        f"🤖 Выберите модель ИИ.\nТекущая модель: <b>{current_model}</b>",
        reply_to_message=False,
        reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True),
        parse_mode=ParseMode.HTML
    )
    # Устанавливаем состояние ожидания выбора модели
    await state.set_state(Form.model_selection)

@router.message(Form.model_selection) # Обработчик для состояния выбора модели
async def process_model_selection(message: types.Message, state: FSMContext):
    """Завершает процесс выбора модели ИИ."""
    # Проверяем, что сообщение текстовое
    if not message.text:
        await safe_send_message(message, "❌ Пожалуйста, используйте кнопки для выбора.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None)
        # Не очищаем состояние, даем еще попытку
        return

    # Убираем маркер ✅ из текста кнопки, если он есть
    selected_model_key = message.text.replace("✅ ", "").strip()

    if selected_model_key in storage.get_available_models():
        storage.set_model(message.chat.id, selected_model_key)
        await safe_send_message(
            message,
            f"✅ Модель успешно изменена на <b>{selected_model_key}</b>!",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(), # Убираем клавиатуру
            parse_mode=ParseMode.HTML
        )
    else:
        await safe_send_message(
            message,
            "❌ Некорректный выбор модели. Пожалуйста, используйте кнопки.",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=None
        )

    # Очищаем состояние FSM в любом случае
    await state.clear()

@router.message(Command("image", "img", "имейдж", "имж", "имг"))
async def cmd_image(message: types.Message, command: CommandObject):
    """Генерирует изображение по текстовому промпту."""
    prompt = None
    if command.args:
        prompt = command.args.strip()
    elif message.reply_to_message and (message.reply_to_message.text or message.reply_to_message.caption):
        prompt = (message.reply_to_message.text or message.reply_to_message.caption).strip()

    if not prompt:
        await safe_send_message(message, "❌ Пожалуйста, укажите описание для изображения после команды или ответьте на сообщение с текстом.", reply_to_message=True, parse_mode=None)
        return

    # Проверяем длину промпта
    max_len = MAX_IMAGE_PROMPT_LENGTH if isinstance(MAX_IMAGE_PROMPT_LENGTH, int) and MAX_IMAGE_PROMPT_LENGTH > 0 else 1000 # Дефолт 1000
    if len(prompt) > max_len:
        await safe_send_message(message, f"❌ Описание слишком длинное (макс. {max_len} символов).", reply_to_message=True, parse_mode=None)
        return

    status_msg = await safe_send_message(message, "🎨 Генерирую изображение...", reply_to_message=True, parse_mode=None)

    try:
        user_id = message.from_user.id if message.from_user else "unknown"
        logger.info(f"Generating image for user {user_id} in chat {message.chat.id} with prompt: {prompt[:50]}...")

        # --- Выбор провайдера и модели для изображений ---
        # Можно вынести в config.py или выбрать здесь
        # Важно: Убедитесь, что выбранный провайдер и модель поддерживают генерацию изображений
        # и возвращают URL или байты изображения. Логика ниже ожидает URL.
        # Пример: Используем модель из конфига, если она подходит, или дефолтную
        image_model_key = storage.get_model_key(message.chat.id) # Попробуем текущую модель
        image_provider_config = PROVIDERS.get(image_model_key)

        # Проверяем, подходит ли текущая модель/провайдер (нужно определить критерии)
        # Например, если провайдер ARTA или Bing (условно)
        is_suitable_provider = image_provider_config and image_provider_config.get("provider") in ["ARTA", "Bing", "OpenaiChat"] # Добавьте подходящие

        if not is_suitable_provider:
             # Если текущая не подходит, ищем первую подходящую в конфиге
             found_suitable = False
             for key, config in PROVIDERS.items():
                 if config.get("provider") in ["ARTA", "Bing", "OpenaiChat"]: # Ищем подходящий
                     image_model_key = key
                     image_provider_config = config
                     logger.info(f"Current model not suitable for images. Using '{key}' instead.")
                     found_suitable = True
                     break
             if not found_suitable:
                 logger.error("No suitable image generation provider found in config.")
                 if status_msg: await status_msg.delete()
                 await safe_send_message(message, "❌ Не найдена подходящая модель для генерации изображений в конфигурации.", reply_to_message=True, parse_mode=None)
                 return

        image_provider_name = image_provider_config.get("provider")
        image_model_identifier = image_provider_config.get("model_name") # Может отличаться от текстовой модели

        if not image_provider_name or not image_model_identifier or image_provider_name not in PROVIDER_MAP:
             logger.error(f"Invalid config for image model '{image_model_key}'. Provider: {image_provider_name}, Model ID: {image_model_identifier}")
             if status_msg: await status_msg.delete()
             await safe_send_message(message, f"❌ Ошибка конфигурации для модели изображений '{image_model_key}'.", reply_to_message=True, parse_mode=None)
             return

        image_provider_class = PROVIDER_MAP[image_provider_name]
        # --- Конец выбора провайдера ---


        logger.info(f"Using {image_provider_name} ({image_model_identifier}) for image generation.")

        # Формируем промпт для модели изображений
        image_generation_prompt = f"Generate an image based on this description: {prompt}"
        # Некоторые модели могут требовать специфичного формата промпта

        response = await g4f.ChatCompletion.create_async(
            model=image_model_identifier,
            provider=image_provider_class,
            messages=[{"role": "user", "content": image_generation_prompt}],
            timeout=180 # Увеличенный таймаут для генерации изображений
        )

        response_str = str(response) # Преобразуем ответ в строку

        # Ищем URL изображения в ответе. Эта логика может потребовать адаптации
        # в зависимости от формата ответа конкретного провайдера.
        # Улучшенный регэкс для поиска URL изображений (http/https, без пробелов и скобок в URL)
        image_urls = re.findall(r'https?://[^\s()\'"]+\.(?:png|jpe?g|webp|gif)\b', response_str, re.IGNORECASE)

        if status_msg:
            try:
                await status_msg.delete()
            except TelegramAPIError: pass # Игнорируем ошибку удаления

        if image_urls:
            # Берем первый найденный URL
            image_url = image_urls[0]
            logger.info(f"Found image URL: {image_url}")
            try:
                # Отправляем фото по URL
                await message.reply_photo(
                    photo=image_url,
                    caption=f"🖼 {escape_html(prompt[:1000])}" # Ограничиваем длину подписи (1024 макс)
                )
            except TelegramBadRequest as e:
                 # Частая ошибка - URL невалиден или недоступен
                 logger.error(f"Failed to send photo by URL {image_url}: {e}. URL might be invalid or inaccessible.")
                 await safe_send_message(message, f"⚠️ Не удалось загрузить или отправить изображение по полученной ссылке. Возможно, ссылка некорректна или недоступна.\n({escape_html(str(e))})", reply_to_message=True, parse_mode=None)
            except Exception as e:
                 logger.exception(f"Error sending photo from URL {image_url}: {e}")
                 await safe_send_message(message, f"⚠️ Ошибка при отправке изображения: {escape_html(str(e))}", reply_to_message=True, parse_mode=None)

        else:
            logger.warning(f"No image URL found in response for prompt: {prompt[:50]}. Response: {response_str[:200]}")
            await safe_send_message(message, "❌ Не удалось получить URL изображения от модели. Попробуйте другой запрос или модель.", reply_to_message=True, parse_mode=None)

    except asyncio.TimeoutError:
        logger.error(f"Timeout error during image generation for prompt: {prompt[:50]}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, "⚠️ Время ожидания генерации изображения истекло.", reply_to_message=True, parse_mode=None)

    except Exception as e:
        logger.exception(f"Error during image generation for prompt: {prompt[:50]}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, f"⚠️ Ошибка генерации изображения: {escape_html(str(e))}", reply_to_message=True, parse_mode=None)


@router.message(Command("sglypa", "сглыпа", "рандом"))
async def cmd_sglypa(message: types.Message, command: CommandObject):
    """Управляет режимом случайных сообщений в группе."""
    if not is_group_chat(message):
        await safe_send_message(message, "ℹ️ Эта команда доступна только в групповых чатах.", reply_to_message=False, parse_mode=None)
        return

    chat_id = message.chat.id
    args = command.args
    settings = storage.init_chat_settings(chat_id) # Получаем или инициализируем настройки

    # Проверяем MIN_INTERVAL
    min_interval = MIN_INTERVAL if isinstance(MIN_INTERVAL, int) and MIN_INTERVAL > 0 else 60 # Дефолт 60 сек

    if args:
        try:
            minutes = int(args)
            if minutes <= 0:
                 await safe_send_message(message, "❌ Интервал должен быть положительным числом минут.", reply_to_message=True, parse_mode=None)
                 return

            interval_seconds = minutes * 60

            if interval_seconds < min_interval:
                await safe_send_message(
                    message,
                    f"❌ Минимальный интервал для случайных сообщений: <b>{min_interval // 60}</b> минут. Вы указали: {minutes} мин.",
                    reply_to_message=True,
                    parse_mode=ParseMode.HTML # Используем HTML для жирного шрифта
                )
                return

            settings['interval'] = interval_seconds
            settings['active'] = True # Активируем режим при установке интервала
            # Сбрасываем таймер при изменении интервала, чтобы следующее сообщение было через новый интервал
            settings['last_sent'] = time.time()

            logger.info(f"Random messages activated for chat {chat_id} with interval {minutes} minutes.")
            await safe_send_message(
                message,
                f"✅ Режим случайных сообщений <b>активирован</b>!\n"
                f"Интервал: <b>{minutes}</b> минут.\n"
                f"Следующее сообщение примерно через {minutes} мин.",
                reply_to_message=False,
                parse_mode=ParseMode.HTML
            )

        except ValueError:
            await safe_send_message(message, "❌ Неверный формат интервала. Укажите целое число минут (например, <code>/sglypa 30</code>).", reply_to_message=True, parse_mode=ParseMode.HTML)
            return
        except Exception as e:
             logger.exception(f"Error processing sglypa command with args in chat {chat_id}: {e}")
             await safe_send_message(message, f"⚠️ Произошла ошибка: {escape_html(str(e))}", reply_to_message=True, parse_mode=None)

    else:
        # Переключаем режим, если аргументы не указаны
        settings['active'] = not settings['active']
        status = "активирован" if settings['active'] else "выключен"
        interval_minutes = settings.get('interval', RANDOM_MESSAGE_INTERVAL) // 60
        interval_info = ""

        if settings['active']:
            # Если активировали, сбрасываем таймер
            settings['last_sent'] = time.time()
            interval_info = (
                f"\nТекущий интервал: <b>{interval_minutes}</b> мин.\n"
                f"Следующее сообщение примерно через {interval_minutes} мин."
            )
            logger.info(f"Random messages toggled to {status} for chat {chat_id}.")
        else:
            logger.info(f"Random messages toggled to {status} for chat {chat_id}.")


        await safe_send_message(
            message,
            f"🔔 Режим случайных сообщений теперь <b>{status}</b>!{interval_info}",
            reply_to_message=False,
            parse_mode=ParseMode.HTML
        )

# --- Message Handlers ---

# Обработка упоминаний бота в группах
# Важно: Этот хэндлер должен идти ПОСЛЕ обработчиков команд,
# чтобы команды типа /ask @botname обрабатывались как команды, а не упоминания.
@router.message(F.text, F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def handle_mention(message: types.Message):
    """Обрабатывает сообщения, начинающиеся с упоминания бота."""
    global BOT_USERNAME
    if not BOT_USERNAME:
        logger.warning("BOT_USERNAME not set, cannot process mentions yet.")
        return # Не можем обработать упоминание, если имя пользователя неизвестно

    # Проверяем наличие текста сообщения
    if not message.text:
        return # Не обрабатываем сообщения без текста (например, стикеры с упоминанием)

    # Паттерн для поиска упоминания в начале строки (с @ или без)
    # re.escape экранирует специальные символы в имени пользователя
    # \b - граница слова, чтобы не срабатывало на часть ника
    # re.IGNORECASE - не учитывать регистр
    mention_pattern = re.compile(rf'^(?:@{re.escape(BOT_USERNAME)}\b\s*)+', re.IGNORECASE)

    if mention_pattern.match(message.text):
        # Проверяем, что сообщение не является командой (начинается с /)
        # Это предотвратит обработку команд типа "/start @botname" как упоминания
        command_pattern = re.compile(r'^\s*/')
        if command_pattern.match(mention_pattern.sub('', message.text).strip()):
             logger.debug(f"Ignoring message starting with command after mention: {message.text[:50]}")
             return


        # Удаляем упоминание из начала текста
        question = mention_pattern.sub('', message.text).strip()
        user_id = message.from_user.id if message.from_user else "unknown"
        logger.info(f"Bot mentioned in chat {message.chat.id} by user {user_id}")

        # Если после упоминания нет текста, но есть ответ на другое сообщение, берем текст оттуда
        if not question and message.reply_to_message:
            original_text = message.reply_to_message.text or message.reply_to_message.caption
            if original_text:
                question = original_text.strip()
                logger.info("Using text from replied message as question for mention.")

        # Проверяем, есть ли вопрос и не слишком ли он длинный
        if not question:
            logger.debug("Mention without question text and no usable reply.")
            # Можно отправить сообщение типа "Да?", но лучше проигнорировать
            # await safe_send_message(message, "Да?", reply_to_message=True, parse_mode=None)
            return

        max_len = MAX_QUESTION_LENGTH if isinstance(MAX_QUESTION_LENGTH, int) and MAX_QUESTION_LENGTH > 0 else 2000 # Дефолт 2000
        if len(question) > max_len:
            await safe_send_message(message, f"❌ Ваш вопрос слишком длинный (макс. {max_len} символов).", reply_to_message=True, parse_mode=None)
            return

        # Формируем текст запроса, как в handle_command_with_question (вариант без ответа)
        if not message.from_user: # Доп. проверка
             logger.error("Mention handler: message.from_user is None.")
             return
        current_user = message.from_user
        current_user_name = escape_html(current_user.full_name if hasattr(current_user, 'full_name') and current_user.full_name else current_user.first_name)
        question_text_escaped = escape_html(question) # Вопрос уже обрезан по длине выше
        full_question_text = f"[Вопрос через упоминание] (Автор: {current_user_name}): {question_text_escaped}"
        full_question_text += "\n[Инструкция для LLM]: Ответь на вопрос пользователя, заданный через упоминание."

        # Обрабатываем вопрос через стандартную функцию
        await process_question(message, full_question_text, memory_type="short")


# --- Startup and Shutdown ---

# Исправленная сигнатура: принимаем bot: Bot
async def on_startup(bot: Bot):
    """Выполняется при запуске бота."""
    global BOT_USERNAME
    try:
        # Используем переданный экземпляр bot
        me = await bot.get_me()
        BOT_USERNAME = me.username
        # Проверяем, что имя пользователя получено
        if not BOT_USERNAME:
             logger.warning("Bot username is empty or None after get_me(). Mentions might not work.")
             BOT_USERNAME = "UnknownBot" # Запасной вариант

        logger.info(f"Bot @{BOT_USERNAME} (ID: {me.id}) started successfully!")

        # Запускаем фоновые задачи
        # Проверяем наличие PROVIDERS перед запуском фоновых задач, которые их используют
        if PROVIDERS and PROVIDER_MAP:
             asyncio.create_task(send_random_messages(), name="RandomMessageSender")
             logger.info("Random message task scheduled.")
        else:
             logger.warning("PROVIDERS or PROVIDER_MAP is empty. Random message task NOT scheduled.")

        # asyncio.create_task(save_state_periodically(), name="StateSaver") # Если нужна периодическая сериализация состояния

    except Exception as e:
        logger.critical(f"Failed to get bot info or schedule tasks on startup: {e}", exc_info=True)
        # Возможно, стоит завершить работу, если не удалось получить имя пользователя
        # exit(1)

# Исправленная сигнатура: принимаем bot: Bot
async def on_shutdown(bot: Bot):
    """Выполняется при остановке бота."""
    logger.info("Bot is shutting down...")
    try:
        # Корректное закрытие сессии бота
        # Используем переданный экземпляр bot
        if bot.session:
             await bot.session.close()
             logger.info("Bot session closed.")
        else:
             logger.info("Bot session was not active or already closed.")
    except Exception as e:
         logger.error(f"Error closing bot session: {e}", exc_info=True)
    # Здесь можно добавить сохранение состояния перед выходом, если необходимо
    logger.info("Shutdown process finished.")


if __name__ == "__main__":
    # Регистрируем роутер в диспетчере *перед* регистрацией хуков
    dp.include_router(router)

    # Регистрируем обработчики запуска и остановки
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    # Запуск polling'а
    try:
        logger.info("Starting bot polling...")
        # Передаем экземпляр бота в run_polling
        # allowed_updates можно настроить для получения только нужных типов обновлений
        # dp.resolve_used_update_types() автоматически определяет используемые типы
        dp.run_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        logger.info("Bot stopped by KeyboardInterrupt.")
    except Exception as e:
        # Логируем критическую ошибку, которая привела к остановке поллинга
        logger.critical(f"Critical error during polling: {e}", exc_info=True)
    finally:
        logger.info("Polling finished.")
        # Дополнительные действия по очистке, если нужны (хотя on_shutdown должен вызываться)