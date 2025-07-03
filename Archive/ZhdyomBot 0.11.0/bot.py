# -*- coding: utf-8 -*-
# COPYRIGHT ZhdomDev
# Optimized version

__version__ = "0.11" # Версия бота (изменена на 0.11, опция чтения общих сообщений чата по умолчанию выключена)

import logging
import re
import asyncio
import random
import time
import html
import json # Импортируем модуль json
import os # Импортируем модуль os для работы с файлами
from typing import Dict, List, Optional, Tuple, Any, Callable, Coroutine

# Aiogram imports
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, CommandObject
from aiogram.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, KeyboardButton, Message, ChatPermissions
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.enums import ChatType, ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter, TelegramBadRequest, TelegramAPIError
from aiogram.client.default import DefaultBotProperties # Для установки parse_mode по умолчанию

# g4f imports
import g4f
from g4f.Provider import BaseProvider
import aiohttp # Import aiohttp to catch its exceptions
import g4f.errors # Import g4f.errors to catch specific g4f exceptions

# Configuration import
# --- Важно: Убедитесь, что файлы config.py и secrets.py существуют и содержат необходимые переменные ---
try:
    from config import (
        PROVIDERS, # Словарь с конфигурацией провайдеров и моделей для текста
        IMAGE_PROVIDERS_CONFIG, # Словарь с конфигурацией провайдеров для изображений
        DEFAULT_MODEL, # Модель по умолчанию для текста
        DEFAULT_IMAGE_MODEL, # Модель по умолчанию для изображений
        MAX_CONTEXT_LENGTH, # Макс. длина контекста (в парах сообщений) - теперь используется как default *для пар*
        MAX_QUESTION_LENGTH, # Макс. длина вопроса пользователя
        MAX_IMAGE_PROMPT_LENGTH, # Макс. длина промпта для изображения
        THINKING_MESSAGES, # Сообщения о том, что бот "думает"
        RANDOM_MESSAGE_INTERVAL, # Интервал случайных сообщений по умолчанию (в секундах)
        MIN_INTERVAL, # Минимальный интервал для случайных сообщений (в секундах)
        SAVE_INTERVAL_SECONDS, # Интервал для периодического сохранения памяти
        BOT_LEGEND, # Импортируем BOT_LEGEND из config.py
        MAX_PROVIDER_RETRIES, # Максимальное количество попыток смены провайдера
        DEFAULT_REPLY_MODE, # Режим ответа по умолчанию
        ERROR_RESPONSES_1, # Ответы на первую ошибку
        ERROR_RESPONSES_FINAL, # Ответы на финальную ошибку
        RETRY_DELAY_SECONDS # Задержка между попытками
    )
except ImportError as e:
    print(f"Ошибка импорта из config.py: {e}")
    print("Пожалуйста, убедитесь, что файл config.py существует и содержит необходимые переменные (включая IMAGE_PROVIDERS_CONFIG, DEFAULT_IMAGE_MODEL, SAVE_INTERVAL_SECONDS, BOT_LEGEND, MAX_PROVIDER_RETRIES, DEFAULT_REPLY_MODE, ERROR_RESPONSES_1, ERROR_RESPONSES_FINAL, RETRY_DELAY_SECONDS), кроме TELEGRAM_TOKEN.")
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
# Собираем все уникальные имена провайдеров из текстовых и графических конфигов
all_provider_names = set()
if PROVIDERS:
    all_provider_names.update({prov_data["provider"] for prov_data in PROVIDERS.values()})
if IMAGE_PROVIDERS_CONFIG:
     all_provider_names.update({prov_data["provider"] for prov_data in IMAGE_PROVIDERS_CONFIG.values()})


if all_provider_names: # Проверяем, что есть имена провайдеров для маппинга
    try:
        PROVIDER_MAP = {
            name: getattr(g4f.Provider, name)
            for name in all_provider_names
            if hasattr(g4f.Provider, name)
        }
        logger.info(f"Loaded providers: {list(PROVIDER_MAP.keys())}")

        # Проверка наличия всех необходимых провайдеров
        missing_providers = {
            name for name in all_provider_names
            if name not in PROVIDER_MAP
        }
        if missing_providers:
            logger.error(f"Missing provider classes in g4f library: {missing_providers}")
            # Можно добавить exit(1) если критично
    except AttributeError as e:
        logger.error(f"Error accessing provider attributes in g4f: {e}. Please check g4f installation and provider names in config.py.")
    except Exception as e:
         logger.error(f"An unexpected error occurred during provider mapping: {e}")

else:
    logger.warning("PROVIDERS and IMAGE_PROVIDERS_CONFIG dictionaries in config.py are empty or not defined. No providers loaded.")


# --- Utility Functions ---

def is_group_chat(message: types.Message) -> bool:
    """Проверяет, является ли чат групповым."""
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}

def escape_html(text: str) -> str:
    """Экранирует основные HTML-спецсимволы."""
    if not isinstance(text, str): # Добавим проверку типа
        logger.warning(f"escape_html received non-string input: {type(text)}. Converting to string.")
        text = str(text)
    # Заменяем основные символы
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # quote=False в html.escape не экранирует кавычки, так что делаем это вручную, если нужно
    # text = text.replace('"', '&quot;')
    return text

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
    # Важно: Экранирование происходит *перед* заменой Markdown на HTML теги
    # text_escaped = escape_html(text) # Экранируем ВЕСЬ текст

    # Временное решение: Не экранировать весь текст сразу, а экранировать *содержимое* тегов при замене
    # Это может быть менее надежно, если Markdown содержит незакрытые теги или сложную вложенность

    # 2. Обрабатываем блоки кода (```...```) -> <pre><code>...</code></pre>
    # Экранируем содержимое блока кода перед вставкой
    def replace_code_block(match):
        lang = match.group(1) # Язык (может быть пустым)
        code = match.group(2)
        escaped_code = html.escape(code) # Используем html.escape для содержимого кода
        if lang:
            # Добавляем класс языка для возможной подсветки на клиенте
            return f'<pre><code class="language-{html.escape(lang)}">{escaped_code}</code></pre>'
        else:
            return f'<pre><code>{escaped_code}</code></pre>'
    # Паттерн для ``` optional_lang \n code ```
    text = re.sub(r'```(\w*)\s*\n(.*?)\n```', replace_code_block, text, flags=re.DOTALL)
    # Паттерн для ```code``` (без языка и переносов строк)
    text = re.sub(r'```(.*?)```', lambda m: f'<pre><code>{html.escape(m.group(1))}</code></pre>', text, flags=re.DOTALL)


    # 3. Обрабатываем инлайн код (`...`) -> <code>...</code>
    # Экранируем содержимое
    text = re.sub(r'`([^`]+?)`', lambda m: f'<code>{html.escape(m.group(1))}</code>', text)

    # 4. Обрабатываем жирный (**...**) -> <b>...</b>
    # Заменяем только если внутри не только пробелы
    text = re.sub(r'\*\*(?=\S)(.+?)(?<=\S)\*\*', r'<b>\1</b>', text)

    # 5. Обрабатываем курсив (__...__) -> <i>...</i>
    # Заменяем только если внутри не только пробелы
    text = re.sub(r'__(?=\S)(.+?)(?<=\S)__', r'<i>\1</i>', text)

    # 6. Обрабатываем зачеркнутый (~~...~~) -> <s>...</s>
    # Заменяем только если внутри не только пробелы
    text = re.sub(r'~~(?=\S)(.+?)(?<=\S)~~', r'<s>\1</s>', text)

    # 7. Обрабатываем ссылки [text](url) -> <a href="url">text</a>
    # Экранируем URL и текст ссылки перед вставкой
    def replace_link(match):
        link_text = match.group(1)
        url = match.group(2)
        # Экранируем URL для атрибута href
        safe_url = html.escape(url).replace('"', '&quot;')
        # Экранируем текст ссылки
        safe_text = html.escape(link_text)
        return f'<a href="{safe_url}">{safe_text}</a>'
    text = re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)

    return text


async def safe_send_message(
    target: types.Message,
    text: str,
    parse_mode: Optional[str] = ParseMode.HTML,
    reply_to_message: bool = True,
    bot_instance: Optional[Bot] = None, # Renamed from bot to bot_instance for clarity in this function
    **kwargs: Any
) -> Optional[types.Message]:
    """
    Безопасная отправка сообщений с обработкой ошибок форматирования и длины.
    `target` - это сообщение, НА КОТОРОЕ нужно ответить, или сообщение, из чата которого нужно отправить новое.
    `reply_to_message` - если True, будет использован метод target.reply(). Если False, будет использован bot.send_message(chat_id=target.chat.id, ...).
    """
    max_length = 4096
    sent_message = None
    chat_id = target.chat.id

    if not isinstance(text, str):
        logger.warning(f"safe_send_message received non-string: {type(text)}. Converting.")
        text = str(text)

    # Use the passed bot_instance if available, otherwise use target.bot
    actual_bot = bot_instance if bot_instance else target.bot
    if not actual_bot:
        logger.error(f"safe_send_message: Cannot determine Bot instance for chat {chat_id}")
        return None

    send_method_callable: Optional[Callable[..., Coroutine[Any, Any, types.Message]]] = None
    send_args = kwargs.copy()

    # Определяем, как будем отправлять: ответом на target или новым сообщением в чат target
    if reply_to_message and hasattr(target, 'reply'):
        send_method_callable = target.reply
        # target.reply уже знает message_id для ответа
    else:
        send_method_callable = actual_bot.send_message
        # Для send_message нужно будет передать chat_id явно
        # Убираем reply_to_message_id, если он там случайно оказался и мы не отвечаем
        send_args.pop('reply_to_message_id', None)


    if not send_method_callable:
        logger.error(f"Could not determine send method for chat {chat_id}")
        return None

    try:
        if parse_mode == ParseMode.HTML:
            processed_text = convert_markdown_to_html(text)
        else:
            processed_text = text

        for i in range(0, len(processed_text), max_length):
            chunk = processed_text[i:i + max_length]

            current_send_args = send_args.copy() # Копия для каждой части
            current_send_args['text'] = chunk
            current_send_args['parse_mode'] = parse_mode

            if send_method_callable == actual_bot.send_message:
                # Если отправляем новым сообщением, добавляем chat_id
                current_send_args['chat_id'] = chat_id
                # Убедимся, что reply_to_message_id не передается, если это не ответ
                if not reply_to_message : # Дополнительная проверка на случай, если reply_to_message был False
                    current_send_args.pop('reply_to_message_id', None)

            sent_message = await send_method_callable(**current_send_args)

            if len(processed_text) > max_length:
                await asyncio.sleep(0.3)
        return sent_message

    except (TelegramBadRequest, ValueError) as e:
        error_text = str(e)
        logger.warning(f"Formatting/Request error with parse_mode={parse_mode}: {error_text}. Text: {text[:100]}...")
        if "parse" in error_text or "tag" in error_text or "entity" in error_text or parse_mode == ParseMode.HTML:
            logger.info("Retrying to send message without parse_mode due to formatting error.")
            try:
                plain_text = text
                for i in range(0, len(plain_text), max_length):
                    chunk = plain_text[i:i + max_length]

                    current_send_args = send_args.copy()
                    current_send_args['text'] = chunk
                    current_send_args['parse_mode'] = None # No parse mode

                    if send_method_callable == actual_bot.send_message:
                        current_send_args['chat_id'] = chat_id
                        if not reply_to_message:
                             current_send_args.pop('reply_to_message_id', None)

                    sent_message = await send_method_callable(**current_send_args)

                    if len(plain_text) > max_length:
                        await asyncio.sleep(0.3)
                return sent_message
            except Exception as e_final:
                logger.exception(f"Failed to send message even without parse_mode: {e_final}")
                try:
                    error_report = f"⚠️ Не удалось отправить отформатированный ответ.\nОшибка: {escape_html(str(e))}\nПопытка без форматирования тоже не удалась: {escape_html(str(e_final))}"

                    error_send_args = send_args.copy()
                    error_send_args['text'] = error_report[:max_length]
                    error_send_args['parse_mode'] = None

                    if send_method_callable == actual_bot.send_message:
                        error_send_args['chat_id'] = chat_id
                        if not reply_to_message:
                             error_send_args.pop('reply_to_message_id', None)

                    await send_method_callable(**error_send_args)

                except Exception:
                    logger.error("Failed even to send the error message.")
                return None
        else:
            logger.error(f"Unhandled BadRequest error: {e}")
            return None

    except TelegramRetryAfter as e:
        logger.warning(f"Rate limit hit for chat {chat_id}. Retrying after {e.retry_after}s.")
        await asyncio.sleep(e.retry_after)
        return await safe_send_message(target, text, parse_mode, reply_to_message, bot_instance=actual_bot, **kwargs)

    except TelegramForbiddenError:
        logger.error(f"Bot is forbidden to perform action in chat {chat_id}. Maybe kicked or blocked.")
        if chat_id in storage.active_chats:
             del storage.active_chats[chat_id]
             logger.info(f"Removed chat {chat_id} from active random messaging.")
        return None

    except TelegramAPIError as e:
        logger.exception(f"Unhandled Telegram API Error in safe_send_message: {e}")
        try:
            error_report = f"⚠️ Произошла ошибка Telegram API: {escape_html(str(e))}"
            error_send_args = send_args.copy()
            error_send_args['text'] = error_report
            error_send_args['parse_mode'] = None

            if send_method_callable == actual_bot.send_message:
                error_send_args['chat_id'] = chat_id
                if not reply_to_message:
                     error_send_args.pop('reply_to_message_id', None)

            await send_method_callable(**error_send_args)
        except Exception:
            logger.error("Failed to send Telegram API error message.")
        return None

    except Exception as e:
        logger.exception(f"Unexpected error in safe_send_message: {e}")
        try:
            error_report = f"⚠️ Произошла непредвиденная ошибка: {escape_html(str(e))}"
            error_send_args = send_args.copy()
            error_send_args['text'] = error_report
            error_send_args['parse_mode'] = None

            if send_method_callable == actual_bot.send_message:
                error_send_args['chat_id'] = chat_id
                if not reply_to_message:
                    error_send_args.pop('reply_to_message_id', None)

            await send_method_callable(**error_send_args)
        except Exception:
            logger.error("Failed to send unexpected error message.")
        return None


# --- Context Management ---

class ChatContext:
    """Управляет контекстом диалога для одного чата/пользователя."""
    def __init__(self, key: int, max_short_term: int):
        self.key = key # Сохраняем ключ для сохранения/загрузки

        if isinstance(BOT_LEGEND, list) and all(isinstance(item, dict) for item in BOT_LEGEND):
             self.legend = list(BOT_LEGEND)
        else:
             logger.error("BOT_LEGEND in config.py is not a list of dictionaries. Using empty legend.")
             self.legend = []

        self.short_term: List[Dict[str, str]] = [] # Инициализируем short_term перед вызовом set_max_short_term
        self.long_term: List[Dict[str, str]] = [] # Инициализируем long_term

        self._max_short_term = 0 # Инициализируем для type hinting
        self.set_max_short_term(max_short_term) # Используем сеттер для валидации

        self.load_long_term()

    def set_max_short_term(self, new_limit: int):
        """Устанавливает новый лимит для краткосрочной памяти, обрезая её при необходимости."""
        validated_limit = max(0, int(new_limit)) if isinstance(new_limit, (int, float, str)) and str(new_limit).isdigit() else (_DEFAULT_SHORT_TERM_MESSAGE_LIMIT)
        if validated_limit == 0:
            logger.warning(f"Short-term memory limit set to 0 for key {self.key}. Short-term memory will be disabled.")
        
        self._max_short_term = validated_limit
        
        # Обрезаем short_term, если новый лимит меньше текущего размера
        if len(self.short_term) > self._max_short_term:
            self.short_term = self.short_term[-self._max_short_term:]
            logger.info(f"Short-term memory for key {self.key} trimmed to {self._max_short_term} messages.")
        logger.debug(f"Short-term memory limit for key {self.key} set to {self._max_short_term} messages.")


    @property
    def max_short_term(self) -> int:
        return self._max_short_term


    def add_message(self, role: str, content: str, memory_type: str = "short"):
        """Добавляет сообщение в указанный тип памяти."""
        if not isinstance(content, str):
             logger.warning(f"Attempted to add non-string content to context: {type(content)}. Converting.")
             content = str(content)
        if not content:
             logger.warning(f"Attempted to add empty message to {memory_type} memory for role {role}.")
             return

        message = {"role": role, "content": content}
        if memory_type == "short":
            if self.max_short_term > 0:
                self.short_term.append(message)
                # Теперь обрезаем по установленному _max_short_term
                if len(self.short_term) > self.max_short_term:
                    self.short_term = self.short_term[-self.max_short_term:]
        elif memory_type == "long":
            self.long_term.append(message)


    def get_full_context(self) -> List[Dict[str, str]]:
        """Возвращает полный контекст для передачи модели."""
        # Этот метод больше не используется напрямую для формирования контекста LLM,
        # так как логика перенесена в process_question для гибкой настройки.
        # Оставлен для совместимости, если где-то еще используется.
        return self.legend + self.long_term + self.short_term

    def get_long_context(self) -> List[Dict[str, str]]:
        """Возвращает только легенду и долгосрочную память."""
        return self.legend + self.long_term

    def clear_short(self):
        """Очищает краткосрочную память."""
        self.short_term = []
        logger.info(f"Short-term context cleared for key {self.key}.")

    def clear_all(self):
        """Очищает всю память (краткосрочную и долгосрочную)."""
        self.short_term = []
        self.long_term = []
        logger.info(f"Full context cleared for key {self.key} (short-term and long-term).")
        filename = f"long_term_memory_{self.key}.json"
        if os.path.exists(filename):
            try:
                os.remove(filename)
                logger.info(f"Deleted long-term memory file: {filename}")
            except OSError as e:
                logger.error(f"Error deleting long-term memory file {filename}: {e}")


    def save_long_term(self):
        """Сохраняет долгосрочную память в JSON файл."""
        if not self.long_term:
            filename = f"long_term_memory_{self.key}.json"
            if os.path.exists(filename):
                 try:
                    os.remove(filename)
                    logger.info(f"Deleted empty long-term memory file: {filename}")
                 except OSError as e:
                    logger.error(f"Error deleting empty long-term memory file {filename}: {e}")
            return

        filename = f"long_term_memory_{self.key}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.long_term, f, ensure_ascii=False, indent=4)
        except IOError as e:
            logger.error(f"Error saving long-term memory for key {self.key} to {filename}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during saving long-term memory for key {self.key}: {e}")


    def load_long_term(self):
        """Загружает долгосрочную память из JSON файла."""
        filename = f"long_term_memory_{self.key}.json"
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        self.long_term = loaded_data
                        logger.info(f"Long-term memory loaded for key {self.key} from {filename}. Items: {len(self.long_term)}")
                    else:
                        logger.warning(f"Loaded data from {filename} is not a list. Ignoring.")
                        self.long_term = []
            except FileNotFoundError:
                self.long_term = []
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from long-term memory file {filename}. File might be corrupted. Starting with empty memory.")
                self.long_term = []
            except IOError as e:
                logger.error(f"Error loading long-term memory for key {self.key} from {filename}: {e}")
                self.long_term = []
            except Exception as e:
                logger.exception(f"Unexpected error during loading long-term memory for key {self.key}: {e}")
                self.long_term = []
        else:
            self.long_term = []


# --- Global Storage ---

# Эти дефолты теперь независимы от _DEFAULT_CONTEXT_TYPES_ENABLED
_DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED = False # Изменено на False по умолчанию
_DEFAULT_SHORT_TERM_MESSAGE_LIMIT = MAX_CONTEXT_LENGTH * 2 # 2 сообщения на пару (пользователь + ассистент)

# Дефолтные настройки для включения типов контекста в промпт LLM
_DEFAULT_CONTEXT_TYPES_ENABLED = {
    'legend': True,
    'long': True,
    'short': True,
}


class Storage:
    """Хранит контексты чатов, настройки моделей и активные чаты для случайных сообщений."""
    def __init__(self):
        self.contexts: Dict[int, ChatContext] = {}
        # Эти словари теперь будут хранить текущие настройки для каждого ключа (chat_id/user_id)
        self.models: Dict[int, str] = {}
        self.image_models: Dict[int, str] = {}
        self.reply_modes: Dict[int, str] = {} # chat_id -> "original" | "direct"
        self.context_types_enabled: Dict[int, Dict[str, bool]] = {} # chat_id -> {'legend': True, 'long': True, 'short': True}
        self.read_general_chat_messages_enabled: Dict[int, bool] = {} # chat_id -> True/False for general group text to short term
        self.short_term_limits: Dict[int, int] = {} # chat_id -> int for max messages in short term memory
        self.active_chats: Dict[int, Dict[str, Any]] = {} # Настройки для случайных сообщений
        self.developer_modes: Dict[int, bool] = {} # chat_id -> True/False для режима разработчика


    def _get_key(self, chat_id: int, user_id: int, is_group: bool) -> int:
        """Возвращает ключ для хранения данных (chat_id для групп, user_id для ЛС)."""
        return chat_id if is_group else user_id

    def _get_settings_filepath(self, key: int) -> str:
        """Генерирует путь к файлу настроек для конкретного чата/пользователя."""
        return f"chat_settings_{key}.json"

    def _load_settings_for_key(self, key: int):
        """Загружает настройки для конкретного ключа (chat_id/user_id) из файла."""
        filepath = self._get_settings_filepath(key)
        
        # Убедимся, что дефолтные значения всегда есть, прежде чем пытаться загрузить
        if key not in self.models:
            self.models[key] = DEFAULT_MODEL
        if key not in self.image_models:
            self.image_models[key] = DEFAULT_IMAGE_MODEL
        if key not in self.reply_modes:
            self.reply_modes[key] = DEFAULT_REPLY_MODE
        if key not in self.context_types_enabled:
            self.context_types_enabled[key] = _DEFAULT_CONTEXT_TYPES_ENABLED.copy()
        if key not in self.read_general_chat_messages_enabled: # New default
            self.read_general_chat_messages_enabled[key] = _DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED
        if key not in self.short_term_limits: # New default
            self.short_term_limits[key] = _DEFAULT_SHORT_TERM_MESSAGE_LIMIT
        if key not in self.developer_modes: # Default to False if not in file
            self.developer_modes[key] = False


        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                    
                    # Обновляем словари storage загруженными данными
                    # Используем .get() с дефолтами, чтобы избежать KeyError, если какая-то настройка отсутствует в файле
                    self.models[key] = settings_data.get('model', DEFAULT_MODEL)
                    self.image_models[key] = settings_data.get('image_model', DEFAULT_IMAGE_MODEL)
                    self.reply_modes[key] = settings_data.get('reply_mode', DEFAULT_REPLY_MODE)
                    self.developer_modes[key] = settings_data.get('developer_mode', False) # Load developer_mode
                    
                    # Для context_types_enabled, сливаем с дефолтными, чтобы сохранить структуру и добавить новые ключи
                    loaded_context_types = settings_data.get('context_types_enabled', {})
                    if isinstance(loaded_context_types, dict):
                        self.context_types_enabled[key] = {**_DEFAULT_CONTEXT_TYPES_ENABLED, **loaded_context_types}
                        # Удаляем ultra_short, если он был в загруженных настройках
                        if 'ultra_short' in self.context_types_enabled[key]:
                            del self.context_types_enabled[key]['ultra_short']
                        # Для обратной совместимости: если старая 'group_text_to_short_term' есть, переносим ее значение
                        if 'group_text_to_short_term' in loaded_context_types:
                             self.read_general_chat_messages_enabled[key] = loaded_context_types['group_text_to_short_term']
                             del self.context_types_enabled[key]['group_text_to_short_term'] # Удаляем из старого места
                    else:
                        logger.warning(f"Invalid 'context_types_enabled' format for key {key}. Resetting to default.")
                        self.context_types_enabled[key] = _DEFAULT_CONTEXT_TYPES_ENABLED.copy()

                    # Загружаем новые отдельные настройки
                    self.read_general_chat_messages_enabled[key] = settings_data.get('read_general_chat_messages_enabled', _DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED)
                    self.short_term_limits[key] = settings_data.get('short_term_limit', _DEFAULT_SHORT_TERM_MESSAGE_LIMIT)


                    logger.info(f"Loaded settings for key {key} from {filepath}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading settings for key {key} from {filepath}: {e}. Using current defaults.")
                # Если загрузка не удалась, оставляем те дефолтные значения, которые были установлены в начале функции
            except Exception as e:
                logger.exception(f"Unexpected error loading settings for key {key}: {e}. Using current defaults.")
        else:
            logger.info(f"Settings file not found for key {key}. Using default settings.")
            # Если файл не существует, уже инициализированы дефолтные значения

    def _save_settings_for_key(self, key: int):
        """Сохраняет настройки для конкретного ключа (chat_id/user_id) в файл."""
        filepath = self._get_settings_filepath(key)
        
        # Получаем актуальные настройки (с дефолтами, если их нет)
        context_types_to_save = self.context_types_enabled.get(key, _DEFAULT_CONTEXT_TYPES_ENABLED.copy())
        if 'ultra_short' in context_types_to_save: # Убедимся, что ultra_short не сохраняется
            del context_types_to_save['ultra_short']

        settings_to_save = {
            'model': self.models.get(key, DEFAULT_MODEL),
            'image_model': self.image_models.get(key, DEFAULT_IMAGE_MODEL),
            'reply_mode': self.reply_modes.get(key, DEFAULT_REPLY_MODE),
            'developer_mode': self.developer_modes.get(key, False),
            'context_types_enabled': context_types_to_save,
            'read_general_chat_messages_enabled': self.read_general_chat_messages_enabled.get(key, _DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED),
            'short_term_limit': self.short_term_limits.get(key, _DEFAULT_SHORT_TERM_MESSAGE_LIMIT)
        }
            
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved settings for key {key} to {filepath}")
        except IOError as e:
            logger.error(f"Error saving settings for key {key} to {filepath}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error saving settings for key {key}: {e}")

    def get_context(self, chat_id: int, user_id: int, is_group: bool) -> ChatContext:
        """Получает или создает контекст для чата/пользователя."""
        key = self._get_key(chat_id, user_id, is_group)
        
        # Убедимся, что настройки для этого ключа загружены при первом доступе
        self._load_settings_for_key(key)
        
        # Получаем актуальный лимит для краткосрочной памяти
        current_short_term_limit = self.short_term_limits.get(key, _DEFAULT_SHORT_TERM_MESSAGE_LIMIT)

        if key not in self.contexts:
            self.contexts[key] = ChatContext(key=key, max_short_term=current_short_term_limit)
            logger.info(f"Created new context for key {key} (chat={chat_id}, user={user_id}, group={is_group}) with short-term limit {current_short_term_limit}.")
        else:
            # Обновляем max_short_term для существующего контекста, если он изменился
            if self.contexts[key].max_short_term != current_short_term_limit:
                self.contexts[key].set_max_short_term(current_short_term_limit)
                logger.info(f"Updated existing context's short-term limit for key {key} to {current_short_term_limit}.")

        return self.contexts[key]

    def clear_short(self, chat_id: int, user_id: int, is_group: bool):
        """Очищает краткосрочную память для чата/пользователя."""
        key = self._get_key(chat_id, user_id, is_group)
        if key in self.contexts:
            self.contexts[key].clear_short()
        else:
             logger.warning(f"Attempted to clear short context for non-existent key {key}")


    async def clear_all(self, chat_id: int, user_id: int, is_group: bool, bot_instance: Bot): # Added bot_instance parameter
        """Очищает всю память (краткосрочную и долгосрочную), включая файл настроек."""
        key = self._get_key(chat_id, user_id, is_group)
        if key in storage.contexts:
            storage.contexts[key].clear_all()
            del storage.contexts[key]
            # Также удаляем файл настроек
            filepath = self._get_settings_filepath(key)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Deleted settings file: {filepath}")
                except OSError as e:
                    logger.error(f"Error deleting settings file {filepath}: {e}")
            # Удаляем записи из внутренних словарей Storage
            self.models.pop(key, None)
            self.image_models.pop(key, None)
            self.reply_modes.pop(key, None)
            self.context_types_enabled.pop(key, None)
            self.read_general_chat_messages_enabled.pop(key, None) # Clear new setting
            self.short_term_limits.pop(key, None) # Clear new setting
            self.developer_modes.pop(key, None) # Clear developer mode setting

            await safe_send_message(target=types.Message(chat=types.Chat(id=chat_id, type=ChatType.PRIVATE), from_user=types.User(id=user_id, is_bot=False)), # Временно создаем stub
                                    text="♻️ Вся память (краткосрочная и долгосрочная) и настройки очищены!", 
                                    reply_to_message=False, parse_mode=None, bot_instance=bot_instance) # Используем переданный bot_instance
        else:
             await safe_send_message(target=types.Message(chat=types.Chat(id=chat_id, type=ChatType.PRIVATE), from_user=types.User(id=user_id, is_bot=False)), # Временно создаем stub
                                    text="ℹ️ Контекст для этого чата не найден.", 
                                    reply_to_message=False, parse_mode=None, bot_instance=bot_instance) # Используем переданный bot_instance

    def get_available_models(self, for_dev_mode: bool = False) -> List[str]:
        """
        Возвращает список доступных имен моделей для текста из конфига.
        Если for_dev_mode=True, включает модели с developer_only=True.
        """
        if not PROVIDERS:
            return []
        
        filtered_models = []
        for model_key, config in PROVIDERS.items():
            is_developer_only = config.get("developer_only", False)
            if not is_developer_only or for_dev_mode:
                filtered_models.append(model_key)
        return filtered_models

    def get_available_image_models(self, for_dev_mode: bool = False) -> List[str]:
        """
        Возвращает список доступных имен моделей для изображений из конфига.
        Если for_dev_mode=True, включает модели с developer_only=True.
        """
        if not IMAGE_PROVIDERS_CONFIG:
            return []

        filtered_models = []
        for model_key, config in IMAGE_PROVIDERS_CONFIG.items():
            is_developer_only = config.get("developer_only", False)
            if not is_developer_only or for_dev_mode:
                filtered_models.append(model_key)
        return filtered_models


    def get_model_key(self, chat_id: int) -> str:
        """Возвращает ключ текущей модели для текста для чата (или дефолтную)."""
        # Убедимся, что настройки для этого chat_id загружены
        if chat_id not in self.models:
            self._load_settings_for_key(chat_id)
        return self.models.get(chat_id, DEFAULT_MODEL)

    def set_model(self, chat_id: int, model_key: str):
        """Устанавливает модель для текста для чата."""
        # Для установки модели нужно проверить, доступна ли она в текущем режиме или она не developer_only
        is_dev_mode = self.get_developer_mode(chat_id)
        if model_key in self.get_available_models(for_dev_mode=is_dev_mode):
            self.models[chat_id] = model_key
            self._save_settings_for_key(chat_id) # Сохраняем настройки немедленно
            logger.info(f"Text model for chat {chat_id} set to '{model_key}'")
        else:
            logger.warning(f"Attempted to set invalid text model '{model_key}' for chat {chat_id} (not available or developer_only in non-dev mode).")

    def get_image_model_key(self, chat_id: int) -> str:
        """Возвращает ключ текущей модели для изображений для чата (или дефолтную)."""
        # Убедимся, что настройки для этого chat_id загружены
        if chat_id not in self.image_models:
            self._load_settings_for_key(chat_id)
        return self.image_models.get(chat_id, DEFAULT_IMAGE_MODEL)

    def set_image_model(self, chat_id: int, model_key: str):
        """Устанавливает модель для изображений для чата."""
        is_dev_mode = self.get_developer_mode(chat_id)
        if model_key in self.get_available_image_models(for_dev_mode=is_dev_mode):
            self.image_models[chat_id] = model_key
            self._save_settings_for_key(chat_id) # Сохраняем настройки немедленно
            logger.info(f"Image model for chat {chat_id} set to '{model_key}'")
        else:
            logger.warning(f"Attempted to set invalid image model '{model_key}' for chat {chat_id} (not available or developer_only in non-dev mode).")

    def get_reply_mode(self, chat_id: int) -> str:
        """Возвращает режим ответа для чата ('original' или 'direct')."""
        # Убедимся, что настройки для этого chat_id загружены
        if chat_id not in self.reply_modes:
            self._load_settings_for_key(chat_id)
        return self.reply_modes.get(chat_id, DEFAULT_REPLY_MODE)

    def set_reply_mode(self, chat_id: int, mode: str):
        """Устанавливает режим ответа для чата."""
        if mode in ["original", "direct"]:
            self.reply_modes[chat_id] = mode
            self._save_settings_for_key(chat_id) # Сохраняем настройки немедленно
            logger.info(f"Reply mode for chat {chat_id} set to '{mode}'")
        else:
            logger.warning(f"Attempted to set invalid reply mode '{mode}' for chat {chat_id}.")

    def get_chat_context_settings(self, chat_id: int) -> Dict[str, bool]:
        """Возвращает настройки включения/выключения типов контекста для чата."""
        # Убедимся, что настройки для этого chat_id загружены
        if chat_id not in self.context_types_enabled:
            self._load_settings_for_key(chat_id)
        # Возвращаем копию, чтобы избежать случайных изменений извне
        return self.context_types_enabled.get(chat_id, _DEFAULT_CONTEXT_TYPES_ENABLED).copy()

    def set_chat_context_setting(self, chat_id: int, setting_name: str, enabled: bool):
        """Устанавливает настройку для конкретного типа контекста в чате."""
        # Убедимся, что настройки для этого chat_id загружены
        if chat_id not in self.context_types_enabled:
            self._load_settings_for_key(chat_id) # Это инициализирует self.context_types_enabled[chat_id]

        if setting_name in self.context_types_enabled[chat_id]:
            self.context_types_enabled[chat_id][setting_name] = enabled
            self._save_settings_for_key(chat_id) # Сохраняем настройки немедленно
            logger.info(f"Context setting '{setting_name}' for chat {chat_id} set to {enabled}")
        else:
            logger.warning(f"Attempted to set invalid context setting '{setting_name}' for chat {chat_id}.")

    def get_read_general_chat_messages_enabled(self, chat_id: int) -> bool:
        """Возвращает состояние опции чтения общих сообщений чата."""
        if chat_id not in self.read_general_chat_messages_enabled:
            self._load_settings_for_key(chat_id)
        return self.read_general_chat_messages_enabled.get(chat_id, _DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED)

    def set_read_general_chat_messages_enabled(self, chat_id: int, enabled: bool):
        """Устанавливает состояние опции чтения общих сообщений чата."""
        self.read_general_chat_messages_enabled[chat_id] = enabled
        self._save_settings_for_key(chat_id)
        logger.info(f"Read general chat messages for chat {chat_id} set to {enabled}")


    def get_short_term_limit(self, chat_id: int) -> int:
        """Возвращает текущий лимит краткосрочной памяти для чата."""
        if chat_id not in self.short_term_limits:
            self._load_settings_for_key(chat_id)
        return self.short_term_limits.get(chat_id, _DEFAULT_SHORT_TERM_MESSAGE_LIMIT)

    def set_short_term_limit(self, chat_id: int, limit: int):
        """Устанавливает лимит краткосрочной памяти для чата и обновляет контекст."""
        validated_limit = max(0, int(limit)) if isinstance(limit, (int, float, str)) and str(limit).isdigit() else _DEFAULT_SHORT_TERM_MESSAGE_LIMIT
        self.short_term_limits[chat_id] = validated_limit
        self._save_settings_for_key(chat_id)
        logger.info(f"Short-term message limit for chat {chat_id} set to {validated_limit}")

        # Если контекст уже существует, обновим его лимит
        # Мы не можем получить ChatContext здесь через _get_key, потому что нам нужен user_id
        # Но при следующем обращении через get_context он будет обновлен.
        # Для немедленного обновления для группы, где user_id=0:
        group_key = self._get_key(chat_id, 0, True)
        if group_key in self.contexts:
            self.contexts[group_key].set_max_short_term(validated_limit)
            logger.info(f"Updated ChatContext for group key {group_key} with new short-term limit {validated_limit}.")
        
        # Для приватных чатов, обновление произойдет при следующем get_context для конкретного user_id
        # Нет простого способа найти все user_id в этой группе, если это групповой чат
        # Но для приватных чатов ключ chat_id равен user_id, так что это тоже сработает.


    def get_developer_mode(self, chat_id: int) -> bool:
        """Возвращает состояние режима разработчика для данного чата."""
        if chat_id not in self.developer_modes:
            self._load_settings_for_key(chat_id)
        return self.developer_modes.get(chat_id, False) # Default to False

    def set_developer_mode(self, chat_id: int, enabled: bool):
        """Устанавливает состояние режима разработчика для данного чата."""
        self.developer_modes[chat_id] = enabled
        self._save_settings_for_key(chat_id)
        logger.info(f"Developer mode for chat {chat_id} set to {enabled}")


    def init_chat_settings(self, chat_id: int) -> Dict[str, Any]:
        """Инициализирует или возвращает настройки для случайных сообщений в чате."""
        if chat_id not in self.active_chats:
            interval = RANDOM_MESSAGE_INTERVAL if isinstance(RANDOM_MESSAGE_INTERVAL, (int, float)) and RANDOM_MESSAGE_INTERVAL > 0 else 3600 # Дефолт - 1 час
            self.active_chats[chat_id] = {
                'active': False,
                'interval': interval,
                'last_sent': 0.0
            }
            logger.info(f"Initialized random message settings for chat {chat_id} with interval {interval}s")
        return self.active_chats[chat_id]

# --- Bot Initialization ---
default_properties = DefaultBotProperties(parse_mode=ParseMode.HTML)
bot: Optional[Bot] = None
try:
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is not defined in secrets.py")
    bot_instance = Bot(token=TELEGRAM_TOKEN, default=default_properties)
    bot = bot_instance
except ValueError as e:
    logger.critical(f"Invalid or missing Telegram token: {e}")
    exit(1)
except Exception as e:
     logger.critical(f"Failed to initialize Bot: {e}")
     exit(1)


dp = Dispatcher()
storage = Storage()
router = Router()
BOT_USERNAME: Optional[str] = None

# --- Finite State Machine (FSM) ---
class Form(StatesGroup):
    """Состояния для процесса выбора модели и настроек контекста."""
    model_selection = State()
    image_model_selection = State()
    context_settings_menu = State()
    set_short_term_limit = State() # New state for setting short-term message limit

# --- User Activity Logging ---
USER_ACTIVITY_LOG_FILE = "user_activity.log"

async def log_user_activity(
    chat_id: int,
    chat_title: str,
    user_name: str, # Swapped order for consistency
    user_id: int,
    chat_type: str
):
    """Логирует активность пользователя в чате в отдельный файл."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_entry = f"{timestamp} | Chat ID: {chat_id} | Chat Type: {chat_type} | Chat Title: {chat_title} | User ID: {user_id} | User Name: {user_name}\n"

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: append_to_file(USER_ACTIVITY_LOG_FILE, log_entry)
        )
    except Exception as e:
        logger.error(f"Error writing to user activity log file {USER_ACTIVITY_LOG_FILE}: {e}")

def append_to_file(filename, data):
    """Синхронная функция для добавления данных в файл."""
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(data)
    except IOError as e:
        logger.error(f"IOError appending to file {filename}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error appending to file {filename}: {e}")


# --- Background Tasks ---

async def send_random_messages(bot: Bot): # Parameter name changed to 'bot'
    """Периодически отправляет случайные сообщения в активные чаты."""
    await asyncio.sleep(15)
    logger.info("Starting random message task...")
    while True:
        await asyncio.sleep(30)
        current_time = time.time()

        active_chat_ids = list(storage.active_chats.keys())

        for chat_id in active_chat_ids:
            if chat_id not in storage.active_chats:
                continue

            settings = storage.active_chats[chat_id]

            if not settings.get('active'):
                continue

            interval = settings.get('interval', RANDOM_MESSAGE_INTERVAL)
            if not isinstance(interval, (int, float)) or interval <= 0:
                logger.warning(f"Invalid interval {interval} for chat {chat_id}. Using default {RANDOM_MESSAGE_INTERVAL}s.")
                interval = RANDOM_MESSAGE_INTERVAL
                settings['interval'] = interval

            last_sent = settings.get('last_sent', 0.0)
            time_since_last = current_time - last_sent

            if time_since_last >= interval:
                logger.info(f"Attempting to send random message to chat {chat_id} (Interval passed)")
                try:
                    # Для случайных сообщений всегда используем общий контекст группы (user_id=0)
                    context = storage.get_context(chat_id=chat_id, user_id=0, is_group=True)

                    # --- Provider Retry Logic for Random Messages ---
                    initial_model_key = storage.get_model_key(chat_id)
                    # For random messages, we don't enable developer models by default
                    available_text_models = storage.get_available_models(for_dev_mode=False)

                    # Construct messages for LLM for random messages (always full context for now)
                    # Note: For random messages, we don't apply chat-specific context settings
                    # as they are meant to be general chat interactions.
                    messages_for_random_llm = context.legend + context.long_term + context.short_term

                    valid_messages_for_random = []
                    for msg_item in messages_for_random_llm:
                        if isinstance(msg_item, dict) and 'role' in msg_item and 'content' in msg_item:
                            valid_messages_for_random.append(msg_item)
                        else:
                            logger.warning(f"Chat {chat_id}: Skipping invalid message format in context for random: {msg_item}")

                    random_prompt_messages = valid_messages_for_random + [{
                        "role": "user",
                        "content": (
                            "Ты участник этого группового чата. "
                            "Используя контекст (включая последние сообщения), "
                            "придумай СЛУЧАЙНОЕ, КОРОТКОЕ, НЕЙТРАЛЬНОЕ или ЗАБАВНОЕ сообщение для поддержания беседы. "
                            "Твой ответ должен быть ТОЛЬКО текстом самого сообщения. "
                            "НЕ представляйся, НЕ упоминай, что ты бот, НЕ используй префиксы типа 'Сообщение:', НЕ ссылайся на этот промпт. "
                            "Просто напиши сообщение так, как будто ты обычный участник чата. "
                            "Соблюдай правила форматирования из системных сообщений."
                        )}]

                    response_text = ""
                    try:
                        # Random messages should not use developer mode error handling
                        response_text = await attempt_llm_request(
                            messages_to_send=random_prompt_messages,
                            initial_model_key=initial_model_key,
                            model_config_dict=PROVIDERS,
                            available_model_keys=available_text_models,
                            g4f_function=g4f.ChatCompletion.create_async,
                            is_image_generation=False,
                            max_attempts=MAX_PROVIDER_RETRIES, # Always use retries for random messages
                            chat_id_for_log=chat_id,
                            user_id_for_log=0, # No specific user for random message
                            status_message_to_update=None, # No status message for random
                            bot_for_status_update=bot, # Use the bot instance
                            is_developer_mode=False # Force False for random messages
                        )
                    except Exception as e:
                        logger.error(f"Chat {chat_id}: Failed to get random message from LLM after all attempts: {e}")
                        # If all attempts fail, we just log and skip sending this random message
                        settings['last_sent'] = current_time + 60 # Retry sooner if all failed
                        continue # Skip sending message and go to next chat/wait


                    if response_text:
                        temp_message_stub = types.Message(
                            message_id=0,
                            date=int(time.time()),
                            chat=types.Chat(id=chat_id, type=ChatType.GROUP), # Assuming group chat for random messages
                            from_user=None
                        )
                        logger.info(f"Chat {chat_id}: Attempting to send generated random message via safe_send_message...")
                        sent_msg = await safe_send_message(
                            target=temp_message_stub,
                            text=response_text,
                            parse_mode=ParseMode.HTML,
                            reply_to_message=False,
                            bot_instance=bot # Use the 'bot' parameter from send_random_messages
                        )
                        if sent_msg:
                            settings['last_sent'] = current_time
                            logger.info(f"Chat {chat_id}: Successfully sent random message.")
                            context.add_message("assistant", response_text, "short")
                            logger.info(f"Chat {chat_id}: Added assistant random message to short-term context.")
                        else:
                            logger.error(f"Chat {chat_id}: Failed to send random message using safe_send_message.")
                    else:
                        logger.warning(f"Chat {chat_id}: Received empty random message response after all attempts.")

                except TelegramForbiddenError:
                    logger.warning(f"Chat {chat_id}: Bot forbidden for random message, removing from active list.")
                    if chat_id in storage.active_chats:
                        del storage.active_chats[chat_id]
                except TelegramRetryAfter as e:
                    logger.warning(f"Chat {chat_id}: Rate limit hit during random message, retry after {e.retry_after}s")
                    await asyncio.sleep(e.retry_after)
                except Exception as e: # Catch-all for any other unexpected errors in the random message loop
                    logger.exception(f"Chat {chat_id}: Unhandled error processing random message: {e}")
                    settings['last_sent'] = current_time + 60
                    await asyncio.sleep(1)

# Фоновая задача для периодического сохранения состояния
async def save_state_periodically():
    """Периодически сохраняет долгосрочную память всех активных контекстов."""
    save_interval = SAVE_INTERVAL_SECONDS if isinstance(SAVE_INTERVAL_SECONDS, (int, float)) and SAVE_INTERVAL_SECONDS > 0 else 300
    logger.info(f"Starting periodic state saving task with interval {save_interval} seconds.")
    while True:
        await asyncio.sleep(save_interval)
        logger.info("Initiating periodic state saving...")
        try:
            for key, context_obj in storage.contexts.items(): # Renamed to context_obj to avoid conflict
                context_obj.save_long_term()
            logger.info("Periodic state saving completed.")
        except Exception as e:
            logger.error(f"Error during periodic state saving: {e}", exc_info=True)


# --- Core Logic for LLM interaction with Provider Retry ---

async def attempt_llm_request(
    messages_to_send: List[Dict[str, str]],
    initial_model_key: str,
    model_config_dict: Dict[str, Dict[str, Any]], # PROVIDERS или IMAGE_PROVIDERS_CONFIG
    available_model_keys: List[str], # Full list of keys for this type (text/image)
    g4f_function: Callable, # e.g. g4f.ChatCompletion.create_async or image_generation_orchestrator
    is_image_generation: bool = False,
    image_generation_prompt: Optional[str] = None, # Only for image orchestrator
    max_attempts: int = 3,
    chat_id_for_log: int = 0,
    user_id_for_log: Optional[int] = 0, # Optional user ID for logging
    status_message_to_update: Optional[types.Message] = None, # For "Trying another model..."
    bot_for_status_update: Optional[Bot] = None,
    is_developer_mode: bool = False # New: Developer mode flag
) -> Any: # Returns response text (str) или image URLs (List[str]) или raises exception
    """
    Attempts to get a response from LLM, trying different providers on failure.
    Includes retry logic for common errors and specific handling for auth/requirements.
    In developer mode, retries are disabled, and errors are re-raised immediately.
    """
    if not available_model_keys:
        logger.error(f"[Chat {chat_id_for_log}] No available models to try.")
        raise ValueError("No available models configured.")

    # In developer mode, only one attempt is made per model, no retries.
    effective_max_attempts = 1 if is_developer_mode else max_attempts

    # Create a prioritized list of model keys to try
    ordered_keys_to_try = [initial_model_key] + [m for m in available_model_keys if m != initial_model_key]
    # If initial_model_key was not in available_model_keys (e.g., misconfigured default), start with available_model_keys
    if initial_model_key not in ordered_keys_to_try and available_model_keys:
        ordered_keys_to_try = available_model_keys[:]

    if not ordered_keys_to_try:
        logger.error(f"[Chat {chat_id_for_log}] No models in ordered_keys_to_try after initial setup.")
        raise ValueError("Could not determine order of models to try.")

    last_exception_seen: Optional[Exception] = None

    actual_attempts_done = 0
    current_model_key_index = 0

    while actual_attempts_done < effective_max_attempts and current_model_key_index < len(ordered_keys_to_try):
        current_model_key = ordered_keys_to_try[current_model_key_index]
        actual_attempts_done += 1

        current_provider_config = model_config_dict.get(current_model_key)
        if not current_provider_config:
            logger.warning(f"[Chat {chat_id_for_log}] [Attempt {actual_attempts_done}/{effective_max_attempts}] Model key '{current_model_key}' not in config. Skipping.")
            last_exception_seen = KeyError(f"Model key '{current_model_key}' not found in config.")
            current_model_key_index +=1 # Try next in ordered list
            continue

        provider_name = current_provider_config.get("provider")
        model_id_key = "image_model_name" if is_image_generation else "model_name"
        model_identifier = current_provider_config.get(model_id_key, current_provider_config.get("model_name"))


        if not provider_name or not model_identifier:
            logger.error(f"[Chat {chat_id_for_log}] [Attempt {actual_attempts_done}/{effective_max_attempts}] Incomplete provider config for '{current_model_key}'. Skipping.")
            last_exception_seen = ValueError(f"Incomplete provider config for '{current_model_key}'.")
            current_model_key_index +=1
            continue

        if provider_name not in PROVIDER_MAP:
            logger.error(f"[Chat {chat_id_for_log}] [Attempt {actual_attempts_done}/{effective_max_attempts}] Provider class '{provider_name}' for '{current_model_key}' not found. Skipping.")
            last_exception_seen = KeyError(f"Provider class '{provider_name}' not found.")
            current_model_key_index +=1
            continue

        provider_class_to_use = PROVIDER_MAP[provider_name]

        logger.info(f"[Chat {chat_id_for_log}] [User {user_id_for_log}] Attempt {actual_attempts_done}/{effective_max_attempts}: Using model '{current_model_key}' (Provider: {provider_name}, ID: {model_identifier})")

        if actual_attempts_done > 1 and status_message_to_update and bot_for_status_update and not is_developer_mode:
            # Only edit status message for retries if NOT in developer mode
            try:
                await bot_for_status_update.edit_message_text(
                    text=f"{random.choice(THINKING_MESSAGES)} (Попытка {actual_attempts_done} с моделью {escape_html(current_model_key)}...)",
                    chat_id=status_message_to_update.chat.id,
                    message_id=status_message_to_update.message_id,
                    parse_mode=ParseMode.HTML
                )
            except Exception as e_edit:
                logger.warning(f"Could not edit status message: {e_edit}")

        try:
            response: Any
            if is_image_generation:
                response = await g4f_function(
                    prompt_text=image_generation_prompt,
                    provider_class=provider_class_to_use,
                    model_id=model_identifier,
                    timeout_val=180
                )
                if isinstance(response, list):
                    return response
                else:
                    logger.error(f"Image orchestrator returned non-list: {type(response)} for {current_model_key}")
                    raise ValueError(f"Image generation with {current_model_key} returned unexpected type.")

            else: # Text generation
                response = await g4f_function(
                    model=model_identifier,
                    messages=messages_to_send,
                    provider=provider_class_to_use,
                    timeout=120
                )
                response_content = str(response).strip()
                if not response_content:
                    logger.warning(f"[Chat {chat_id_for_log}] Empty text response from {current_model_key}.")
                    raise ValueError(f"Empty response from {current_model_key}")
                return response_content


        except (asyncio.TimeoutError, aiohttp.client_exceptions.ClientError, g4f.errors.RateLimitError, ValueError, TypeError) as e: # Removed g4f.errors.RetryError
            last_exception_seen = e
            logger.warning(f"[Chat {chat_id_for_log}] [Attempt {actual_attempts_done}/{effective_max_attempts}] Failed with model '{current_model_key}': {type(e).__name__} - {e}")
            if is_developer_mode: # In developer mode, re-raise immediately on first failure
                raise e
            current_model_key_index += 1 # Try next model
            if actual_attempts_done < effective_max_attempts:
                await asyncio.sleep(1) # Small delay
        except (g4f.errors.MissingAuthError, g4f.errors.MissingRequirementsError, g4f.errors.NoValidHarFileError) as e:
            last_exception_seen = e
            logger.error(f"[Chat {chat_id_for_log}] Provider-specific setup error with '{current_model_key}': {e}. This provider will likely not work.")
            raise e # Always re-raise fundamental setup errors immediately
        except Exception as e:
            last_exception_seen = e
            logger.exception(f"[Chat {chat_id_for_log}] [Attempt {actual_attempts_done}/{effective_max_attempts}] Unexpected error with model '{current_model_key}': {e}")
            if is_developer_mode: # In developer mode, re-raise immediately on first failure
                raise e
            current_model_key_index += 1 # Try next model
            if actual_attempts_done < effective_max_attempts:
                await asyncio.sleep(1)

    # All attempts failed or no models to try
    if last_exception_seen:
        failed_model_key_index = current_model_key_index -1 if current_model_key_index > 0 else 0
        failed_model_key = ordered_keys_to_try[failed_model_key_index] if failed_model_key_index < len(ordered_keys_to_try) else "unknown model"

        logger.error(f"[Chat {chat_id_for_log}] All {actual_attempts_done} attempts failed. Last error with model '{failed_model_key}': {last_exception_seen}")
        raise last_exception_seen
    else:
        msg = "All provider attempts failed without a specific operational error (e.g., all models skipped due to config issues or no models available)."
        logger.error(f"[Chat {chat_id_for_log}] {msg}")
        raise Exception(msg)


async def process_question(
    message: types.Message,
    question: str, # 'question' here is the already formatted string from the handler
    memory_type: str = "short",
    command_used: Optional[str] = None # e.g. "ask", "long" for UST logic
):
    """
    Обрабатывает вопрос пользователя, взаимодействует с LLM и отправляет ответ.
    Включает логику повторных попыток на уровне запроса к боту.
    """
    chat_id = message.chat.id
    if not message.from_user:
        logger.error(f"Received message without from_user in chat {chat_id}. Cannot process.")
        return
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    is_developer_mode = storage.get_developer_mode(chat_id)

    chat_title = message.chat.title if is_group else "Private Chat"
    user_name = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name, user_id, message.chat.type)

    # Получаем контекст. get_context теперь автоматически устанавливает max_short_term
    context = storage.get_context(chat_id, user_id, is_group) 
    initial_model_key = storage.get_model_key(chat_id)

    if not PROVIDERS:
        logger.error("PROVIDERS dictionary is empty. Cannot process question.")
        await safe_send_message(message, "⚠️ Ошибка конфигурации: Список моделей для текста пуст.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return

    # Создаем ОДНО статусное сообщение, которое будет обновляться
    status_msg = await safe_send_message(
        message,
        random.choice(THINKING_MESSAGES) if THINKING_MESSAGES else "🤔 Думаю...",
        reply_to_message=True, # Default reply to the command message
        parse_mode=None,
        bot_instance=message.bot
    )

    full_response = ""
    # In developer mode, we only make one external attempt
    max_external_attempts = 1 if is_developer_mode else 2
    attempt_count = 0

    while attempt_count < max_external_attempts:
        attempt_count += 1
        logger.info(f"Processing question from user {user_id} in chat {chat_id}. External attempt {attempt_count}/{max_external_attempts}. DevMode: {is_developer_mode}")

        try:
            # --- Construct messages list for LLM for this turn ---
            llm_messages_list: List[Dict[str, str]] = []
            chat_context_settings = storage.get_chat_context_settings(chat_id)

            # 1. Add Legend (if enabled)
            if chat_context_settings.get('legend', True):
                llm_messages_list.extend(context.legend)

            # 2. Add Long-Term Memory (if enabled)
            if chat_context_settings.get('long', True):
                llm_messages_list.extend(context.long_term)

            # 3. Add Short-Term Memory (conditionally)
            if chat_context_settings.get('short', True):
                # Если short-term включен, используем сообщения из short_term, обрезанные по лимиту.
                # ChatContext уже управляет своим размером при добавлении сообщений.
                llm_messages_list.extend(context.short_term)
                logger.debug(f"Chat {chat_id}: Short-term context enabled. Added {len(context.short_term)} existing messages. Max: {context.max_short_term}.")
            else:
                logger.debug(f"Chat {chat_id}: Short-term context disabled. Past short-term messages will not be included.")
            
            # 4. Always add the CURRENT user message (the 'question' parameter) as the last message for the LLM
            # This 'question' parameter already has the improved timestamp and username prefix.
            llm_messages_list.append({"role": "user", "content": question})
            logger.info(f"Chat {chat_id}: Current user message added to LLM prompt. Total messages for LLM: {len(llm_messages_list)}")

            # Final validation of messages format
            valid_llm_messages = []
            for msg_item in llm_messages_list:
                if isinstance(msg_item, dict) and 'role' in msg_item and 'content' in msg_item:
                    valid_llm_messages.append(msg_item)
                else:
                    logger.warning(f"Chat {chat_id}: Skipping invalid message format in final LLM prompt: {msg_item}")

            if not valid_llm_messages:
                logger.error(f"Chat {chat_id}: No valid messages to send to LLM after construction.")
                raise ValueError("No valid messages for LLM prompt.")

            # --- Attempt LLM request with internal retries (MAX_PROVIDER_RETRIES) ---
            all_available_text_models = storage.get_available_models(for_dev_mode=is_developer_mode)

            full_response = await attempt_llm_request(
                messages_to_send=valid_llm_messages,
                initial_model_key=initial_model_key,
                model_config_dict=PROVIDERS,
                available_model_keys=all_available_text_models,
                g4f_function=g4f.ChatCompletion.create_async,
                is_image_generation=False,
                max_attempts=MAX_PROVIDER_RETRIES,
                chat_id_for_log=chat_id,
                user_id_for_log=user_id,
                status_message_to_update=status_msg, # Передаем статусное сообщение для обновления
                bot_for_status_update=message.bot,
                is_developer_mode=is_developer_mode # Pass developer mode flag
            )

            if full_response: # If we got a successful response, break the loop
                break
            else: # This path might not be taken if attempt_llm_request raises on empty response
                logger.warning(f"Attempt {attempt_count} got empty response. Retrying...")
                if not is_developer_mode and attempt_count < max_external_attempts:
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    continue # Try again
                raise ValueError("Empty response from LLM (internal attempt).") # Force exit/error for dev mode or final attempt

        except Exception as e:
            logger.error(f"External attempt {attempt_count}/{max_external_attempts} failed for user {user_id} in chat {chat_id}. Error: {type(e).__name__} - {e}")

            if is_developer_mode:
                # In developer mode, report the specific error immediately and stop.
                full_response = f"❌ Ошибка в режиме разработчика (модель: <code>{escape_html(initial_model_key)}</code>):\n" \
                                f"<code>{escape_html(str(e))}</code>\n\n" \
                                f"Тип ошибки: <code>{type(e).__name__}</code>"
                break # Exit loop, we have the error message
            elif attempt_count == 1:
                # Первая попытка провалилась, редактируем существующее статусное сообщение
                if status_msg:
                    try:
                        error_text_for_status = random.choice(ERROR_RESPONSES_1) if ERROR_RESPONSES_1 else "⚠️ Произошла ошибка. Пробую ещё раз..."
                        await message.bot.edit_message_text( # Редактируем статусное сообщение
                            text=error_text_for_status,
                            chat_id=status_msg.chat.id,
                            message_id=status_msg.message_id,
                            parse_mode=None # Сообщения об ошибках обычно без форматирования
                        )
                        logger.info(f"Chat {chat_id}: Edited status message to show first retry error.")
                    except Exception as e_edit:
                        logger.warning(f"Could not edit status message to show first retry error: {e_edit}. Falling back to new message if possible.")
                        # Если редактирование не удалось, отправляем новое сообщение
                        await safe_send_message(
                            message,
                            random.choice(ERROR_RESPONSES_1) if ERROR_RESPONSES_1 else "⚠️ Произошла ошибка. Пробую ещё раз...",
                            reply_to_message=True,
                            parse_mode=None,
                            bot_instance=message.bot
                        )
                else: # Если статусное сообщение не было установлено изначально (крайний случай)
                    await safe_send_message(
                        message,
                        random.choice(ERROR_RESPONSES_1) if ERROR_RESPONSES_1 else "⚠️ Произошла ошибка. Пробую ещё раз...",
                        reply_to_message=True,
                        parse_mode=None,
                        bot_instance=message.bot
                    )

                await asyncio.sleep(RETRY_DELAY_SECONDS)
            elif attempt_count == max_external_attempts:
                # Вторая (финальная) попытка провалилась
                full_response = random.choice(ERROR_RESPONSES_FINAL) if ERROR_RESPONSES_FINAL else "⚠️ Произошла критическая ошибка. Пожалуйста, попробуйте позже."
                logger.error(f"Final attempt failed for user {user_id} in chat {chat_id}.")
                break # Выходим из цикла, так как достигнуто максимальное количество попыток

    # --- After loop: Send final response or error message ---
    # Удаляем статусное сообщение в конце обработки запроса
    if status_msg:
        try:
            await status_msg.delete()
        except Exception as e_del:
             logger.warning(f"Could not delete status message {status_msg.message_id}: {e_del}")

    # Add messages to persistent context ONLY if the LLM request was successful (not an error message from us)
    if full_response and full_response not in ERROR_RESPONSES_1 and full_response not in ERROR_RESPONSES_FINAL and not (is_developer_mode and "Ошибка в режиме разработчика" in full_response):
        # Add the original formatted user question to the context
        context.add_message("user", question, memory_type)
        logger.info(f"Added user question to context for user {user_id} in chat {chat_id}. Length: {len(question)}")

        # Add assistant response to context
        context.add_message("assistant", full_response[:8192], memory_type)
        logger.info(f"Added assistant response to context for user {user_id} in chat {chat_id}. Length: {len(full_response)}")
    else:
        logger.warning(f"Not adding messages to context due to empty response or error in chat {chat_id}. DevMode: {is_developer_mode}")
        if is_developer_mode and "Ошибка в режиме разработчика" in full_response:
             # In dev mode, we don't pop the user message from context if an error occurs.
             logger.info(f"Developer mode active. User message for chat {chat_id} remains in context.")
        elif not is_developer_mode and (full_response in ERROR_RESPONSES_FINAL or not full_response):
            # If not dev mode and final error/empty response, remove the last user message from short_term
            # This is important to avoid sending the same failing query again in next context.
            if context.short_term and context.short_term[-1]["role"] == "user":
                context.short_term.pop()
                logger.info(f"Removed user question from short-term context for chat {chat_id} after final attempt failure (not dev mode).")
            elif context.long_term and context.long_term[-1]["role"] == "user": # Also check long-term if that was the target
                context.long_term.pop()
                logger.info(f"Removed user question from long-term context for chat {chat_id} after final attempt failure (not dev mode).")


    # Determine reply target based on chat settings
    reply_target_message = message # Default: reply to the command message
    if message.reply_to_message: # If the command was a reply to something
        current_reply_mode = storage.get_reply_mode(chat_id)
        if current_reply_mode == "original":
            reply_target_message = message.reply_to_message # Switch target to the original message
            logger.info(f"Replying to original message {message.reply_to_message.message_id} due to 'original' reply mode.")
        else:
            logger.info(f"Replying to command message {message.message_id} due to 'direct' reply mode.")

    await safe_send_message(
        target=reply_target_message,
        text=full_response,
        parse_mode=ParseMode.HTML, # Always HTML for dev errors too, as they include <code>
        reply_to_message=True,
        bot_instance=message.bot
    )


async def handle_command_with_question(
    message: types.Message,
    command: CommandObject,
    allow_reply: bool,
    memory_type: str
):
    """
    Общий обработчик для команд /ask и /long, которые принимают текст вопроса.
    Формирует текст запроса к LLM, включая информацию об авторах при ответе.
    """
    command_name = command.command.lower() if command.command else "unknown_command"
    logger.info(f"Handling command {command_name} from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user:
        logger.error(f"Received command message without from_user in chat {message.chat.id}. Cannot process.")
        return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title if is_group else "Private Chat"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    current_user = message.from_user
    current_user_name = escape_html(current_user.full_name if hasattr(current_user, 'full_name') and current_user.full_name else current_user.first_name)
    full_question_text_for_llm: Optional[str] = None # This will be the formatted text for LLM
    raw_question_content: Optional[str] = None
    llm_instruction = ""

    # Generate explicit timestamp for LLM
    timestamp_llm_format = time.strftime('[Текущее время: %H:%M:%S, Дата: %d.%m.%Y]', time.localtime())

    # 1. Обработка текста из аргументов команды или из отвеченного сообщения (если аргументы пусты)
    if command.args:
        raw_question_content = command.args.strip()
        
        if allow_reply and message.reply_to_message: # Command has args AND is a reply
            original_msg = message.reply_to_message
            original_author_name = "Неизвестный автор"
            if original_msg.from_user:
                 original_author_name = escape_html(original_msg.from_user.full_name if hasattr(original_msg.from_user, 'full_name') and original_msg.from_user.full_name else original_msg.from_user.first_name)

            original_text_content = original_msg.text or original_msg.caption
            if original_text_content:
                original_text_escaped = escape_html(original_text_content.strip())[:MAX_QUESTION_LENGTH]
                command_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])

                full_question_text_for_llm = (
                    f"{timestamp_llm_format}\n"
                    f"[Пользователь: {current_user_name}]\n"
                    f"[Ответ на сообщение (Автор: {original_author_name})]\n"
                    f"[Текст оригинального сообщения: {original_text_escaped}]\n"
                    f"[Текст ответа/комментария: {command_text_escaped}]"
                )
                llm_instruction = "Ответь на комментарий пользователя, учитывая контекст сообщения, на которое он отвечает."
            else: # Command has args, is a reply, but replied-to message has no text. Use command args only.
                question_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])
                full_question_text_for_llm = (
                    f"{timestamp_llm_format}\n"
                    f"[Пользователь: {current_user_name}]\n"
                    f"[Сообщение: {question_text_escaped}]"
                )
        else: # Command has args, not a reply (or reply not allowed for this construction)
            question_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])
            full_question_text_for_llm = (
                f"{timestamp_llm_format}\n"
                f"[Пользователь: {current_user_name}]\n"
                f"[Сообщение: {question_text_escaped}]"
            )

    elif allow_reply and message.reply_to_message: # No command args, but it's a reply
        original_msg = message.reply_to_message
        raw_question_content = original_msg.text or original_msg.caption
        if not raw_question_content:
            # Если команда без аргументов является ответом на сообщение БЕЗ ТЕКСТА, то игнорируем.
            logger.info(f"Command /{command_name} была использована в ответ на сообщение без текста. Чтобы задать вопрос по тексту, убедитесь, что исходное сообщение содержит текст.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
            await safe_send_message(message, f"ℹ️ Команда /{command_name} была использована в ответ на сообщение без текста. Чтобы задать вопрос по тексту, убедитесь, что исходное сообщение содержит текст.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
            return

        raw_question_content = raw_question_content.strip() # Store for length check
        original_author_name = "Неизвестный автор"
        if original_msg.from_user:
            original_author_name = escape_html(original_msg.from_user.full_name if hasattr(original_msg.from_user, 'full_name') and original_msg.from_user.full_name else original_msg.from_user.first_name)

        original_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])
        # Формируем вопрос так, будто пользователь {current_user_name} задает вопрос к тексту {original_author_name}
        full_question_text_for_llm = (
            f"{timestamp_llm_format}\n"
            f"[Пользователь: {current_user_name}]\n"
            f"[Контекст из предыдущего сообщения (Автор: {original_author_name})]: {original_text_escaped}\n"
            f"[Подразумеваемый вопрос: (относится к тексту выше)]"
        )
        llm_instruction = f"Пользователь {current_user_name} ответил командой /{command_name} на предыдущее сообщение от {original_author_name}. Сформулируй релевантный ответ или задай уточняющий вопрос к этому предыдущему сообщению, как если бы это делал {current_user_name}."

    else: # No command args and not a usable reply
        if memory_type == "long":
            await safe_send_message(message, "ℹ️ Укажите текст, который нужно добавить в долгосрочную память.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
            return
        await safe_send_message(message, "❌ Пожалуйста, укажите ваш вопрос после команды или ответьте командой на сообщение с текстом.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return

    # Validate raw question content length (if it was set)
    if raw_question_content and len(raw_question_content) > MAX_QUESTION_LENGTH:
        await safe_send_message(message, f"❌ Текст вашего вопроса слишком длинный (>{MAX_QUESTION_LENGTH} симв.). Сократите его.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return

    # Add LLM instruction to the full_question_text_for_llm
    if full_question_text_for_llm and llm_instruction:
         full_question_text_for_llm += f"\n[Инструкция для LLM]: {llm_instruction}"
    elif not full_question_text_for_llm: # Should not happen if logic above is correct
        logger.error("full_question_text_for_llm is None before processing, indicates logic error in handle_command_with_question.")
        await safe_send_message(message, "Internal error: could not formulate question.", reply_to_message=True, bot_instance=message.bot)
        return


    # 3. Валидация общей длины сформированного запроса (увеличим лимит немного)
    max_combined_length = int(MAX_QUESTION_LENGTH * 2.5) # Increased buffer for detailed prompt format
    if len(full_question_text_for_llm) > max_combined_length:
        await safe_send_message(
            message,
            f"❌ Ваш запрос (с учетом контекста ответа и форматирования) слишком длинный (>{max_combined_length} символов). Пожалуйста, сократите его.",
            reply_to_message=True,
            parse_mode=None,
            bot_instance=message.bot
        )
        return

    # 4. Вызов основного обработчика запросов
    await process_question(message, full_question_text_for_llm, memory_type, command_used=command_name)


# --- Command Handlers ---

@router.message(Command("start", "help", "старт", "помощь"))
async def cmd_start(message: types.Message, bot: Bot): # Parameter name changed to 'bot'
    """Обработчик команд /start и /help."""
    logger.info(f"Handling command start/help from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title if is_group else "Private Chat"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    global BOT_USERNAME
    if not BOT_USERNAME:
        try:
            me = await bot.get_me() # Use passed bot instance
            BOT_USERNAME = me.username or "UnknownBot"
            logger.info(f"Got bot username in cmd_start: @{BOT_USERNAME}")
        except Exception as e:
            logger.error(f"Failed to get bot username in cmd_start: {e}")
            BOT_USERNAME = "UnknownBot"

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
        "• <code>/model</code> - Выбрать другую модель ИИ для ответов (текст).\n"
        "• <code>/imagemodel</code> - Выбрать другую модель ИИ для генерации изображений.\n"
        "• <code>/memstats</code> - Показать статистику по памяти (количество сообщений).\n"
        "• <code>/replymode</code> - Изменить режим ответа бота (на исходное сообщение или на команду).\n"
        "• <code>/context</code> - Настроить, какие типы контекста использовать (легенда, долгосрочная, краткосрочная).\n"
        "• <code>/devmode</code> - Включить/выключить режим разработчика (отключает повторные попытки LLM, показывает подробные ошибки).\n\n" # Added devmode command
        "<b>Для групповых чатов:</b>\n"
        f"• <code>/sglypa [минуты]</code> - Включить/выключить режим случайных сообщений с заданным интервалом (например, <code>/sglypa 30</code> для 30 минут). Без аргумента - переключает режим.\n"
        f"• Просто <b>упомяните меня</b> (@{BOT_USERNAME}) в начале сообщения, чтобы задать вопрос.\n\n"
        "<i>Подсказка: Долгосрочная память полезная для задания общего контекста или инструкций боту.</i>"
    )
    await safe_send_message(message, help_text, reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=bot)

# Используем общий обработчик для /ask и его псевдонимов
@router.message(Command("ask", "аск", "bot", "бот", "ждёмби", "zhdomby", "ждемби", "zhdyomby", "брат"))
async def cmd_ask(message: types.Message, command: CommandObject):
    """Обработчик команды /ask и её псевдонимов."""
    logger.info(f"Handling command ask/alias from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    await handle_command_with_question(
        message=message,
        command=command,
        allow_reply=True,
        memory_type="short"
    )

@router.message(Command("long", "лонг"))
async def cmd_long(message: types.Message, command: CommandObject):
    """Обработчик команды /long."""
    logger.info(f"Handling command long from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    await handle_command_with_question(
        message=message,
        command=command,
        allow_reply=False, # /long не отвечает на сообщения, только добавляет контекст
        memory_type="long"
    )

@router.message(Command("clear", "клир", "клиар", "отчистить", "сброс")) # Added aliases
async def cmd_clear(message: types.Message):
    """Очищает краткосрочную память."""
    logger.info(f"Handling command clear/alias from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title if is_group else "Private Chat"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    storage_key = storage._get_key(message.chat.id, message.from_user.id, is_group)
    if storage_key in storage.contexts:
        storage.contexts[storage_key].clear_short()
        await safe_send_message(message, "🧽 Краткосрочный контекст очищен!", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
    else:
        await safe_send_message(message, "ℹ️ Контекст для этого чата не найден.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)


@router.message(Command("fullreset"))
async def cmd_fullreset(message: types.Message):
    """Очищает всю память (краткосрочную и долгосрочную)."""
    logger.info(f"Handling command fullreset from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title if is_group else "Private Chat"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    # Вызываем метод storage.clear_all, который теперь сам отправляет сообщение
    await storage.clear_all(chat_id, user_id, is_group, message.bot) # Передаем bot_instance


@router.message(Command("model", "модель")) # Added alias
async def cmd_model(message: types.Message, state: FSMContext):
    """Начинает процесс выбора модели ИИ для текста."""
    logger.info(f"Handling command model/alias from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title if is_group else "Private Chat"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    is_developer_mode = storage.get_developer_mode(chat_id)
    available_models = storage.get_available_models(for_dev_mode=is_developer_mode)
    
    if not available_models:
         await safe_send_message(message, "❌ Нет доступных моделей для текста в конфигурации.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
         return

    builder = ReplyKeyboardBuilder()
    current_model = storage.get_model_key(message.chat.id)

    for model_key in available_models:
        text = f"✅ {model_key}" if model_key == current_model else model_key
        builder.add(KeyboardButton(text=text))

    builder.adjust(2)

    dev_mode_hint = " (Включен режим разработчика)" if is_developer_mode else ""
    await safe_send_message(
        message,
        f"🤖 Выберите модель ИИ для <b>текста</b>.{dev_mode_hint}\nТекущая модель: <b>{escape_html(current_model)}</b>",
        reply_to_message=False,
        reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True),
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot
    )
    await state.set_state(Form.model_selection)

@router.message(Form.model_selection)
async def process_model_selection(message: types.Message, state: FSMContext):
    """Завершает процесс выбора модели ИИ для текста."""
    logger.info(f"Handling text model selection from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title if is_group else "Private Chat"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    if not message.text:
        await safe_send_message(message, "❌ Пожалуйста, используйте кнопки для выбора модели текста.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None, bot_instance=message.bot)
        return

    selected_model_key = message.text.replace("✅ ", "").strip()

    storage.set_model(message.chat.id, selected_model_key) # set_model now handles validity check
    current_model_after_set = storage.get_model_key(message.chat.id) # Get the actual current model after attempting to set
    
    if current_model_after_set == selected_model_key: # Check if setting was successful
        await safe_send_message(
            message,
            f"✅ Модель для текста успешно изменена на <b>{escape_html(selected_model_key)}</b>!",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot
        )
    else:
        # This means set_model failed, likely because it's a developer-only model in non-dev mode
        await safe_send_message(
            message,
            f"❌ Некорректный выбор модели для текста или модель '<b>{escape_html(selected_model_key)}</b>' доступна только в режиме разработчика.",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot
        )
    await state.clear()

# Обработчик команды /imagemodel
@router.message(Command("imagemodel"))
async def cmd_imagemodel(message: types.Message, state: FSMContext):
    """Начинает процесс выбора модели ИИ для изображений."""
    logger.info(f"Handling command imagemodel from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    is_developer_mode = storage.get_developer_mode(chat_id)
    available_image_models = storage.get_available_image_models(for_dev_mode=is_developer_mode)

    if not available_image_models:
         await safe_send_message(message, "❌ Нет доступных моделей для изображений в конфигурации.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
         return

    builder = ReplyKeyboardBuilder()
    current_image_model = storage.get_image_model_key(message.chat.id)

    for model_key in available_image_models:
        text = f"✅ {model_key}" if model_key == current_image_model else model_key
        builder.add(KeyboardButton(text=text))

    builder.adjust(2)

    dev_mode_hint = " (Включен режим разработчика)" if is_developer_mode else ""
    await safe_send_message(
        message,
        f"🖼 Выберите модель ИИ для <b>изображений</b>.{dev_mode_hint}\nТекущая модель: <b>{escape_html(current_image_model)}</b>",
        reply_to_message=False,
        reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True),
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot
    )
    await state.set_state(Form.image_model_selection)

# Обработчик для состояния выбора графической модели
@router.message(Form.image_model_selection)
async def process_image_model_selection(message: types.Message, state: FSMContext):
    """Завершает процесс выбора модели ИИ для изображений."""
    logger.info(f"Handling image model selection from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    if not message.text:
        await safe_send_message(message, "❌ Пожалуйста, используйте кнопки для выбора модели изображений.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None, bot_instance=message.bot)
        return

    selected_model_key = message.text.replace("✅ ", "").strip()

    storage.set_image_model(message.chat.id, selected_model_key) # set_image_model now handles validity
    current_image_model_after_set = storage.get_image_model_key(message.chat.id)

    if current_image_model_after_set == selected_model_key: # Check if setting was successful
        await safe_send_message(
            message,
            f"✅ Модель для изображений успешно изменена на <b>{escape_html(selected_model_key)}</b>!",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot
        )
    else:
        # This means set_image_model failed, likely because it's a developer-only model in non-dev mode
        await safe_send_message(
            message,
            f"❌ Некорректный выбор модели для изображений или модель '<b>{escape_html(selected_model_key)}</b>' доступна только в режиме разработчика.", # Corrected line
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot
        )
    await state.clear()

@router.message(Command("replymode"))
async def cmd_replymode(message: types.Message):
    """Переключает режим ответа бота."""
    logger.info(f"Handling command replymode from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id # For logging
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)


    current_mode = storage.get_reply_mode(chat_id)
    new_mode = "direct" if current_mode == "original" else "original"
    storage.set_reply_mode(chat_id, new_mode)

    mode_description = "на исходное сообщение (если команда была ответом)" if new_mode == "original" \
                       else "напрямую на сообщение с командой"

    await safe_send_message(
        message,
        f"⚙️ Режим ответа изменен. Теперь бот будет отвечать: <b>{mode_description}</b>.",
        reply_to_message=False,
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot
    )


@router.message(Command("context", "контекст"))
async def cmd_context_settings(message: types.Message, state: FSMContext):
    """Настраивает, какие типы контекста использовать."""
    logger.info(f"Handling command context from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    await _send_context_settings_menu(message, state)


async def _send_context_settings_menu(message: types.Message, state: FSMContext):
    """Вспомогательная функция для отправки меню настроек контекста."""
    chat_id = message.chat.id
    current_settings_prompt_types = storage.get_chat_context_settings(chat_id)
    current_read_general_chat = storage.get_read_general_chat_messages_enabled(chat_id)
    current_short_term_limit = storage.get_short_term_limit(chat_id)

    builder = ReplyKeyboardBuilder()

    # Define button texts and their corresponding setting names for prompt types
    buttons_info_prompt_types = {
        "Легенда": "legend",
        "Долгосрочная память": "long",
        "Краткосрочная память": "short",
    }

    for display_name, setting_key in buttons_info_prompt_types.items():
        is_enabled = current_settings_prompt_types.get(setting_key, _DEFAULT_CONTEXT_TYPES_ENABLED.get(setting_key, True))
        status = "Вкл" if is_enabled else "Выкл"
        status_text = f"{display_name}: {status}"
        builder.add(KeyboardButton(text=status_text))

    # Add the new specific settings for chat reading and short-term limit
    read_general_chat_status = "Вкл" if current_read_general_chat else "Выкл"
    builder.add(KeyboardButton(text=f"Чтение общих сообщений чата: {read_general_chat_status}"))
    builder.add(KeyboardButton(text=f"Лимит краткосрочной памяти: {current_short_term_limit} сообщений"))
    
    builder.add(KeyboardButton(text="Готово")) # Button to exit the menu
    builder.adjust(1) # One button per row for clarity

    await safe_send_message(
        message,
        "⚙️ Настройте, какие типы контекста будут использоваться:\n"
        "<i>(Нажмите на кнопку, чтобы переключить состояние)</i>",
        reply_to_message=False,
        reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True),
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot
    )
    await state.set_state(Form.context_settings_menu)


@router.message(Form.context_settings_menu)
async def process_context_settings_selection(message: types.Message, state: FSMContext):
    """Обрабатывает выбор пользователя в меню настроек контекста."""
    logger.info(f"Handling context settings selection from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    if not message.text:
        await safe_send_message(message, "❌ Пожалуйста, используйте кнопки для выбора настройки контекста.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None, bot_instance=message.bot)
        return

    selected_text = message.text.strip()

    if selected_text == "Готово":
        await safe_send_message(message, "✅ Настройки контекста сохранены.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None, bot_instance=message.bot)
        await state.clear()
        return
    
    # Map button text back to internal setting names for prompt types
    setting_map_prompt_types = {
        "Легенда": "legend",
        "Долгосрочная память": "long",
        "Краткосрочная память": "short",
    }

    # Handle toggle buttons for prompt types
    found_prompt_type_key: Optional[str] = None
    for display_name_prefix, setting_key in setting_map_prompt_types.items():
        if selected_text.startswith(display_name_prefix):
            found_prompt_type_key = setting_key
            break

    if found_prompt_type_key:
        current_settings = storage.get_chat_context_settings(chat_id)
        current_state = current_settings.get(found_prompt_type_key, _DEFAULT_CONTEXT_TYPES_ENABLED.get(found_prompt_type_key, True))
        new_state = not current_state
        storage.set_chat_context_setting(chat_id, found_prompt_type_key, new_state)
        
        await safe_send_message(message, f"⚙️ Настройка '{display_name_prefix}' изменена на {'ВКЛ' if new_state else 'ВЫКЛ'}.", reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=message.bot)
        await _send_context_settings_menu(message, state) # Re-send menu with updated state
        return

    # Handle "Чтение общих сообщений чата" toggle
    if selected_text.startswith("Чтение общих сообщений чата:"):
        current_read_general = storage.get_read_general_chat_messages_enabled(chat_id)
        new_read_general = not current_read_general
        storage.set_read_general_chat_messages_enabled(chat_id, new_read_general)
        
        await safe_send_message(message, f"⚙️ Настройка 'Чтение общих сообщений чата' изменена на {'ВКЛ' if new_read_general else 'ВЫКЛ'}.", reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=message.bot)
        await _send_context_settings_menu(message, state)
        return

    # Handle "Лимит краткосрочной памяти" button to enter input state
    if selected_text.startswith("Лимит краткосрочной памяти:"):
        await safe_send_message(
            message,
            "🔢 Введите новый лимит для краткосрочной памяти (целое число сообщений).\n"
            f"Текущий лимит: <b>{storage.get_short_term_limit(chat_id)}</b>.",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot
        )
        await state.set_state(Form.set_short_term_limit)
        return

    await safe_send_message(message, "❌ Неизвестная настройка. Пожалуйста, выберите из предложенных кнопок.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
    # Stay in the state, allow another attempt


@router.message(Form.set_short_term_limit)
async def process_short_term_limit_input(message: types.Message, state: FSMContext):
    """Обрабатывает ввод нового лимита для краткосрочной памяти."""
    logger.info(f"Handling short-term limit input from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)


    if not message.text:
        await safe_send_message(message, "❌ Пожалуйста, введите числовое значение.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
        return
    
    try:
        new_limit = int(message.text.strip())
        if new_limit < 0:
            await safe_send_message(message, "❌ Лимит должен быть неотрицательным числом.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
            return

        storage.set_short_term_limit(chat_id, new_limit) # This also updates ChatContext
        
        await safe_send_message(
            message,
            f"✅ Лимит краткосрочной памяти установлен на <b>{new_limit}</b> сообщений.",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot
        )
        # Возвращаемся в главное меню настроек контекста
        await _send_context_settings_menu(message, state)

    except ValueError:
        await safe_send_message(message, "❌ Неверный формат. Пожалуйста, введите целое число.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
        # Остаемся в этом состоянии, чтобы пользователь мог ввести корректное число
        await state.set_state(Form.set_short_term_limit)


@router.message(Command("image", "img", "имейдж", "имж", "имг"))
async def cmd_image(message: types.Message, command: CommandObject, bot: Bot): # Parameter name changed to 'bot'
    """Генерирует изображение по текстовому промпту."""
    logger.info(f"Handling command image from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    is_developer_mode = storage.get_developer_mode(chat_id)

    prompt = None
    if command.args:
        prompt = command.args.strip()
    elif message.reply_to_message and (message.reply_to_message.text or message.reply_to_message.caption):
        prompt = (message.reply_to_message.text or message.reply_to_message.caption).strip()

    if not prompt:
        await safe_send_message(message, "❌ Пожалуйста, укажите описание для изображения после команды или ответьте на сообщение с текстом.", reply_to_message=True, parse_mode=None, bot_instance=bot)
        return

    max_len = MAX_IMAGE_PROMPT_LENGTH if isinstance(MAX_IMAGE_PROMPT_LENGTH, int) and MAX_IMAGE_PROMPT_LENGTH > 0 else 1000
    if len(prompt) > max_len:
        await safe_send_message(message, f"❌ Описание слишком длинное (макс. {max_len} символов).", reply_to_message=True, parse_mode=None, bot_instance=bot)
        return

    initial_image_model_key = storage.get_image_model_key(message.chat.id)

    if not IMAGE_PROVIDERS_CONFIG:
         logger.error("IMAGE_PROVIDERS_CONFIG dictionary is empty. Cannot generate image.")
         await safe_send_message(message, "❌ Ошибка конфигурации: Список моделей для изображений пуст.", reply_to_message=True, parse_mode=None, bot_instance=bot)
         return

    status_msg = await safe_send_message(message, "🎨 Генерирую изображение...", reply_to_message=True, parse_mode=None, bot_instance=bot)

    image_urls: List[str] = [] # Explicitly type as list of strings

    # This is the orchestrator function that attempt_llm_request will call for each provider attempt
    async def image_generation_orchestrator(prompt_text:str, provider_class:type[BaseProvider], model_id:str, timeout_val:int) -> List[str]:
        _image_urls_internal: List[str] = []
        _generation_successful = False

        # Try dedicated g4f.image.create_async first
        if hasattr(g4f, 'image') and hasattr(g4f.image, 'create_async') and callable(g4f.image.create_async):
            logger.info(f"Attempting direct image generation via g4f.image.create_async for provider {provider_class.__name__}.")
            try:
                response_list = await g4f.image.create_async(
                    prompt=prompt_text,
                    provider=provider_class,
                    model=model_id,
                    timeout=timeout_val
                )
                if isinstance(response_list, list) and response_list and all(isinstance(url, str) for url in response_list):
                    _image_urls_internal = response_list
                    _generation_successful = True
                    logger.info(f"Direct image generation successful. URLs: {_image_urls_internal}")
                else:
                    logger.warning(f"Direct image generation with {provider_class.__name__} returned non-list of strings or empty: {response_list}")
            except NotImplementedError:
                logger.warning(f"g4f.image.create_async not implemented for {provider_class.__name__}. Will try ChatCompletion fallback.")
            except Exception as e_direct:
                logger.exception(f"Error during direct g4f.image.create_async with {provider_class.__name__}: {e_direct}. Will try ChatCompletion fallback.")

        if not _generation_successful:
            logger.info(f"Attempting image URL extraction from ChatCompletion fallback for provider {provider_class.__name__}.")
            image_gen_prompt_for_chat = f"Generate an image based on this description: {prompt_text}. Provide ONLY the direct image URL in your response."
            try:
                chat_response_obj = await g4f.ChatCompletion.create_async(
                    model=model_id,
                    provider=provider_class,
                    messages=[{"role": "user", "content": image_gen_prompt_for_chat}],
                    timeout=timeout_val
                )
                _response_str_for_fallback = str(chat_response_obj) # Ensure it's a string
                direct_urls = re.findall(r'https?://[^\s()\'"]+\.(?:png|jpe?g|webp|gif)\b', _response_str_for_fallback, re.IGNORECASE)
                markdown_urls = re.findall(r'\[.*?\]\((https?://[^\s()\'"]+)\)', _response_str_for_fallback)
                _image_urls_internal = direct_urls + markdown_urls
                if _image_urls_internal:
                    logger.info(f"Fallback ChatCompletion found URL(s): {_image_urls_internal}")
                else: # Fallback did not find URLs, raise ValueError to signal attempt_llm_request to retry
                    logger.warning(f"Fallback ChatCompletion did not return a valid image URL for provider {provider_class.__name__}. Response: {_response_str_for_fallback[:200]}")
                    raise ValueError(f"Image generation fallback with {provider_class.__name__} did not yield a usable URL.")
            except Exception as e_fallback_chat: # Catch errors from ChatCompletion itself
                 logger.error(f"Error during ChatCompletion fallback for image with {provider_class.__name__}: {e_fallback_chat}")
                 raise e_fallback_chat # Re-raise to be handled by attempt_llm_request

        if _image_urls_internal: # If we have URLs from either method
            return _image_urls_internal
        # If no URLs and no specific error was raised by fallback to signal retry, this means direct method failed quietly or fallback didn't find URLs but didn't error.
        # We need to ensure an error is raised if no URLs are found by the orchestrator, so attempt_llm_request retries.
        raise ValueError(f"Image generation failed for {provider_class.__name__} after all methods within orchestrator: No URLs found.")


    try:
        all_available_image_models = storage.get_available_image_models(for_dev_mode=is_developer_mode)

        # attempt_llm_request will call image_generation_orchestrator for each provider
        # The orchestrator itself handles the two-stage (direct then fallback) logic.
        image_urls = await attempt_llm_request(
            messages_to_send=[], # Not used directly by orchestrator, but required by attempt_llm_request signature
            initial_model_key=initial_image_model_key,
            model_config_dict=IMAGE_PROVIDERS_CONFIG,
            available_model_keys=all_available_image_models,
            g4f_function=image_generation_orchestrator, # Pass the orchestrator
            is_image_generation=True,
            image_generation_prompt=prompt, # Pass the original user prompt
            max_attempts=MAX_PROVIDER_RETRIES,
            chat_id_for_log=chat_id,
            user_id_for_log=user_id,
            status_message_to_update=status_msg,
            bot_for_status_update=bot, # Use 'bot' here
            is_developer_mode=is_developer_mode # Pass developer mode flag
        ) # This should now return List[str] или raise


        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass

        if image_urls and isinstance(image_urls, list) and image_urls[0]: # Check if list and has content
            image_url_to_send = image_urls[0] # Send the first one
            logger.info(f"Successfully generated image. URL: {image_url_to_send}")
            try:
                await message.reply_photo(
                    photo=image_url_to_send,
                    caption=f"🖼 {escape_html(prompt[:1000])}" # Caption is HTML escaped
                )
            except TelegramBadRequest as e:
                 logger.error(f"Failed to send photo by URL {image_url_to_send}: {e}. URL might be invalid or inaccessible.")
                 await safe_send_message(message, f"⚠️ Не удалось загрузить или отправить изображение по полученной ссылке. ({escape_html(str(e))})", reply_to_message=True, parse_mode=None, bot_instance=bot)
            except Exception as e:
                 logger.exception(f"Error sending photo from URL {image_url_to_send}: {e}")
                 await safe_send_message(message, f"⚠️ Ошибка при отправке изображения: {escape_html(str(e))}", reply_to_message=True, parse_mode=None, bot_instance=bot)
        else:
            logger.warning(f"No image URL found after all attempts for prompt: {prompt[:50]}. Final result from attempt_llm_request: {image_urls}")
            await safe_send_message(message, "❌ Не удалось получить URL изображения от моделей после нескольких попыток. Попробуйте другой запрос или модель.", reply_to_message=True, parse_mode=None, bot_instance=bot)

    except (g4f.errors.MissingAuthError, g4f.errors.MissingRequirementsError, g4f.errors.NoValidHarFileError, ValueError) as e:
        error_message_text = f"❌ Ошибка генерации изображения: {escape_html(str(e))}"
        current_img_model_key = storage.get_image_model_key(chat_id) # Get currently set model for more relevant error
        if isinstance(e, g4f.errors.MissingRequirementsError):
            error_message_text = f"❌ Провайдеру изображений (<code>{escape_html(current_img_model_key)}</code>) требуется доп. пакет: (<code>{escape_html(str(e).replace('Install ', ''))}</code>). Попробуйте /imagemodel."
        elif isinstance(e, g4f.errors.MissingAuthError) or "No cookies found" in str(e) or isinstance(e, g4f.errors.NoValidHarFileError):
             error_message_text = f"❌ Провайдер изображений (<code>{escape_html(current_img_model_key)}</code>) требует аутентификации/настройки (куки/HAR). Попробуйте /imagemodel."
        elif "did not yield a usable URL" in str(e) or "No URLs found" in str(e) or "returned unexpected type" in str(e): # From orchestrator
             error_message_text = f"❌ Модель (<code>{escape_html(current_img_model_key)}</code>) не смогла сгенерировать ссылку на изображение. Попробуйте другой запрос или модель."


        logger.error(f"Specific error during image generation process: {e}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, error_message_text, reply_to_message=True, parse_mode=ParseMode.HTML, bot_instance=bot)

    except Exception as e: # Catch-all for failures from attempts or other issues
        logger.exception(f"Unhandled error or all image provider attempts failed for prompt: {prompt[:50]}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, f"⚠️ Не удалось сгенерировать изображение после нескольких попыток. Ошибка: <code>{escape_html(str(e))}</code>", reply_to_message=True, parse_mode=ParseMode.HTML, bot_instance=bot)


@router.message(Command("sglypa", "сглыпа", "рандом"))
async def cmd_sglypa(message: types.Message, command: CommandObject):
    """Управляет режимом случайных сообщений в группе."""
    logger.info(f"Handling command sglypa from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id_cmd = message.chat.id # Renamed to avoid conflict
    user_id_cmd = message.from_user.id
    is_group_cmd = is_group_chat(message)
    chat_title_cmd = message.chat.title or f"Group {chat_id_cmd}" # Added default for chat_title_cmd
    user_name_cmd = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id_cmd, chat_title_cmd, user_name_cmd, user_id_cmd, message.chat.type)


    if not is_group_chat(message):
        await safe_send_message(message, "ℹ️ Эта команда доступна только в групповых чатах.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
        return

    chat_id_settings = message.chat.id # Use this for settings key
    args = command.args
    settings = storage.init_chat_settings(chat_id_settings)

    min_interval_val = MIN_INTERVAL if isinstance(MIN_INTERVAL, int) and MIN_INTERVAL > 0 else 60

    if args:
        try:
            minutes = int(args)
            if minutes <= 0:
                 await safe_send_message(message, "❌ Интервал должен быть положительным числом минут.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
                 return

            interval_seconds = minutes * 60

            if interval_seconds < min_interval_val:
                await safe_send_message(
                    message,
                    f"❌ Минимальный интервал для случайных сообщений: <b>{min_interval_val // 60}</b> минут. Вы указали: {minutes} мин.",
                    reply_to_message=True,
                    parse_mode=ParseMode.HTML,
                    bot_instance=message.bot
                )
                return

            settings['interval'] = interval_seconds
            settings['active'] = True
            settings['last_sent'] = time.time()

            logger.info(f"Random messages activated for chat {chat_id_settings} with interval {minutes} minutes.")
            await safe_send_message(
                message,
                f"✅ Режим случайных сообщений <b>активирован</b>!\n"
                f"Интервал: <b>{minutes}</b> минут.\n"
                f"Следующее сообщение примерно через {minutes} мин.",
                reply_to_message=False,
                parse_mode=ParseMode.HTML,
                bot_instance=message.bot
            )

        except ValueError:
            await safe_send_message(message, "❌ Неверный формат интервала. Укажите целое число минут (например, <code>/sglypa 30</code>).", reply_to_message=True, parse_mode=ParseMode.HTML, bot_instance=message.bot)
            return
        except Exception as e:
             logger.exception(f"Error processing sglypa command with args in chat {chat_id_settings}: {e}")
             await safe_send_message(message, f"⚠️ Произошла ошибка: {escape_html(str(e))}", reply_to_message=True, parse_mode=None, bot_instance=message.bot)

    else:
        settings['active'] = not settings['active']
        status = "активирован" if settings['active'] else "выключен"
        interval_minutes = settings.get('interval', RANDOM_MESSAGE_INTERVAL) // 60
        interval_info = ""

        if settings['active']:
            settings['last_sent'] = time.time()
            interval_info = (
                f"\nТекущий интервал: <b>{interval_minutes}</b> мин.\n"
                f"Следующее сообщение примерно через {interval_minutes} мин."
            )
            logger.info(f"Random messages toggled to {status} for chat {chat_id_settings}.")
        else:
            logger.info(f"Random messages toggled to {status} for chat {chat_id_settings}.")


        await safe_send_message(
            message,
            f"🔔 Режим случайных сообщений теперь <b>{status}</b>!{interval_info}",
            reply_to_message=False,
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot
        )

# Обработчик команды /memstats
@router.message(Command("memstats", "память"))
async def cmd_memstats(message: types.Message):
    """Показывает статистику по краткосрочной и долгосрочной памяти."""
    logger.info(f"Handling command memstats from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id_stats = message.chat.id # Renamed for clarity
    user_id_stats = message.from_user.id
    is_group_stats = is_group_chat(message)
    chat_title_stats = message.chat.title or f"Group {chat_id_stats}"
    user_name_stats = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id_stats, chat_title_stats, user_name_stats, user_id_stats, message.chat.type)

    # Принудительно вызываем get_context для инициализации short_term_limit в контексте, если его еще нет
    context_stats = storage.get_context(chat_id_stats, user_id_stats, is_group_stats) 
    chat_context_settings = storage.get_chat_context_settings(chat_id_stats)

    short_term_count = len(context_stats.short_term)
    long_term_count = len(context_stats.long_term)
    developer_mode_status = "Вкл" if storage.get_developer_mode(chat_id_stats) else "Выкл"


    # Display context settings
    legend_status = "Вкл" if chat_context_settings.get('legend', True) else "Выкл"
    long_status = "Вкл" if chat_context_settings.get('long', True) else "Выкл"
    short_status = "Вкл" if chat_context_settings.get('short', True) else "Выкл"
    read_general_chat_messages_status = "Вкл" if storage.get_read_general_chat_messages_enabled(chat_id_stats) else "Выкл" # New status line
    short_term_limit_display = storage.get_short_term_limit(chat_id_stats) # Get current limit
    
    response_text = (
        f"📊 <b>Статистика памяти:</b>\n"
        f"Краткосрочная память: <b>{short_term_count}</b> сообщений (лимит: {short_term_limit_display})\n" # Updated display
        f"Долгосрочная память: <b>{long_term_count}</b> сообщений\n\n"
        f"⚙️ <b>Настройки контекста:</b>\n"
        f"Легенда: <b>{legend_status}</b>\n"
        f"Долгосрочная память: <b>{long_status}</b>\n"
        f"Краткосрочная память: <b>{short_status}</b>\n"
        f"Чтение общих сообщений чата: <b>{read_general_chat_messages_status}</b>\n\n" # Updated display
        f"🛠 Режим разработчика: <b>{developer_mode_status}</b>"
    )

    await safe_send_message(
        message,
        response_text,
        reply_to_message=False,
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot
    )

@router.message(Command("devmode"))
async def cmd_devmode(message: types.Message):
    """Включает/выключает режим разработчика для чата."""
    logger.info(f"Handling command devmode from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return

    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    chat_title = message.chat.title or f"Group {chat_id}"
    user_name_from_msg = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name_from_msg, user_id, message.chat.type)

    current_dev_mode = storage.get_developer_mode(chat_id)
    new_dev_mode = not current_dev_mode
    storage.set_developer_mode(chat_id, new_dev_mode)

    status_text = "включен" if new_dev_mode else "выключен"
    await safe_send_message(
        message,
        f"🛠 Режим разработчика теперь <b>{status_text}</b> для этого чата.\n"
        f"В этом режиме LLM не будет повторять запросы при ошибках, а покажет детальное сообщение.",
        reply_to_message=False,
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot
    )


# --- Message Handlers ---

# Обработка ЛЮБЫХ текстовых сообщений в личных чатах
@router.message(F.text, F.chat.type == ChatType.PRIVATE)
async def handle_private_text_message(message: types.Message):
    """Обрабатывает любое текстовое сообщение в личном чате как команду /ask."""
    logger.info(f"Handling private text message from user {message.from_user.id if message.from_user else 'unknown'}")
    if not message.from_user or not message.text: return # Added check for message.text

    chat_id_priv = message.chat.id # Renamed
    user_id_priv = message.from_user.id
    chat_title_priv = "Private Chat"
    user_name_priv = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name
    await log_user_activity(chat_id_priv, chat_title_priv, user_name_priv, user_id_priv, message.chat.type)

    question_raw = message.text.strip()

    if question_raw.startswith('/'):
         logger.debug(f"Ignoring private message starting with command: {question_raw[:50]}")
         return

    max_len = MAX_QUESTION_LENGTH if isinstance(MAX_QUESTION_LENGTH, int) and MAX_QUESTION_LENGTH > 0 else 2000
    if len(question_raw) > max_len:
        await safe_send_message(message, f"❌ Ваш вопрос слишком длинный (макс. {max_len} символов).", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return

    current_user = message.from_user
    current_user_name = escape_html(current_user.full_name if hasattr(current_user, 'full_name') and current_user.full_name else current_user.first_name)
    question_text_escaped = escape_html(question_raw)

    # Generate explicit timestamp for LLM
    timestamp_llm_format = time.strftime('[Текущее время: %H:%M:%S, Дата: %d.%m.%Y]', time.localtime())

    full_question_text_for_llm = (
        f"{timestamp_llm_format}\n"
        f"[Пользователь: {current_user_name}]\n"
        f"[Сообщение: {question_text_escaped}]"
    )
    full_question_text_for_llm += "\n[Инструкция для LLM]: Ответь на вопрос пользователя, заданный в личном сообщении."

    await process_question(message, full_question_text_for_llm, memory_type="short", command_used="private_message")


@router.message(F.text, F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def handle_mention_or_group_text(message: types.Message, bot: Bot): # Parameter name changed to 'bot'
    """
    Handles mentions and general group text messages.
    Mention check MUST come first. If not a mention and not a command, it's a general group message.
    """
    logger.debug(f"Group/Supergroup text message received: '{message.text[:50]}' from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user or not message.text: # Check for message.text here
        if not message.text and message.reply_to_message:
             logger.info(f"Received an empty message reply in chat {message.chat.id}. User: {message.from_user.id}. Replying to {message.reply_to_message.message_id}. Bot will not generate new response, relies on command logic for this case.")
        elif not message.text:
             logger.info(f"Received an empty message (not a reply) in chat {message.chat.id}. User: {message.from_user.id}. Ignoring.")
        return


    chat_id_group = message.chat.id # Renamed
    user_id_group = message.from_user.id
    is_group_true = True # We are in this handler
    chat_title_group = message.chat.title or f"Group {chat_id_group}"
    user_name_group = message.from_user.full_name if hasattr(message.from_user, 'full_name') and message.from_user.full_name else message.from_user.first_name

    global BOT_USERNAME
    if not BOT_USERNAME:
        try:
            me = await bot.get_me() # Use the 'bot' parameter
            BOT_USERNAME = me.username or "UnknownBot"
            logger.info(f"Got bot username in handle_mention_or_group_text: @{BOT_USERNAME}")
        except Exception as e:
            logger.warning(f"BOT_USERNAME not set and failed to get it in handle_mention_or_group_text: {e}. Mention processing might fail.")

    # 1. Check for mention
    if BOT_USERNAME and BOT_USERNAME != "UnknownBot": # Ensure BOT_USERNAME is valid
        mention_pattern = re.compile(rf'^(?:@{re.escape(BOT_USERNAME)}\b\s*)+', re.IGNORECASE)
        text_without_mention = mention_pattern.sub('', message.text).lstrip()

        if len(text_without_mention) < len(message.text): # Mention was present
            await log_user_activity(chat_id_group, chat_title_group, user_name_group, user_id_group, message.chat.type) # Log for mention
            logger.info(f"Bot mentioned in chat {chat_id_group} by user {user_id_group}. Text after mention: '{text_without_mention[:50]}'")

            if text_without_mention.startswith('/'):
                logger.debug(f"Ignoring message starting with command after mention: {message.text[:50]}")
                return # Let command handlers take it

            question_from_mention_raw = text_without_mention.strip()

            logger.info(f"Incoming mention message from user {user_id_group} in chat {chat_id_group}. Formulating prompt...")

            if not question_from_mention_raw and message.reply_to_message: # Mention is empty, but is a reply
                original_text_reply = message.reply_to_message.text or message.reply_to_message.caption
                if original_text_reply:
                    question_from_mention_raw = original_text_reply.strip()
                    logger.info("Using text from replied message as question for empty mention.")
                else: # Empty mention replying to message with no text
                    logger.info("Empty mention replying to a message with no text. Ignoring.")
                    return


            if not question_from_mention_raw: # Still no question after checking reply
                logger.debug("Mention without question text and no usable reply.")
                return

            max_len = MAX_QUESTION_LENGTH if isinstance(MAX_QUESTION_LENGTH, int) and MAX_QUESTION_LENGTH > 0 else 2000
            if len(question_from_mention_raw) > max_len:
                await safe_send_message(message, f"❌ Ваш вопрос (после упоминания) слишком длинный (макс. {max_len} символов).", reply_to_message=True, parse_mode=None, bot_instance=bot)
                return

            current_user_mention = message.from_user
            current_user_name_mention = escape_html(current_user_mention.full_name if hasattr(current_user_mention, 'full_name') and current_user_mention.full_name else current_user_mention.first_name)
            question_text_escaped_mention = escape_html(question_from_mention_raw)

            # Generate explicit timestamp for LLM
            timestamp_llm_format = time.strftime('[Текущее время: %H:%M:%S, Дата: %d.%m.%Y]', time.localtime())

            full_question_text_for_llm = (
                f"{timestamp_llm_format}\n"
                f"[Пользователь: {current_user_name_mention}]\n"
                f"[Сообщение (упоминание бота): {question_text_escaped_mention}]"
            )
            full_question_text_for_llm += "\n[Инструкция для LLM]: Ответь на вопрос пользователя, заданный через упоминание в групповом чате. Твоя роль - участник чата."

            await process_question(message, full_question_text_for_llm, memory_type="short", command_used="mention")
            return # Handled as mention

    # 2. If not a mention, and not a command, it's a general group text message for context
    # These messages are NOT directly processed by LLM, only added to short-term context.
    if not message.text.startswith('/'):
        # Check the new setting for enabling general chat messages in short-term memory
        if storage.get_read_general_chat_messages_enabled(chat_id_group): 
            await log_user_activity(chat_id_group, chat_title_group, user_name_group, user_id_group, message.chat.type)
            logger.debug(f"Handling general group text message from user {user_id_group} in chat {chat_id_group} for context: '{message.text[:50]}'")

            # Add the message to the short-term context for the group (using user_id=0 as key for shared context)
            group_shared_context = storage.get_context(chat_id_group, 0, is_group_true) # get_context now ensures max_short_term is set
            
            # Add the message as 'user' role with author name
            # We also need to add the bot's responses to this shared context later
            group_shared_context.add_message("user", f"Пользователь {user_name_group}: {message.text}", "short")
            logger.info(f"Added general group text message from user {user_id_group} ({user_name_group}) to shared short-term context for chat {chat_id_group}. Option 'Чтение общих сообщений чата' is ENABLED.")
        else:
            logger.debug(f"Skipping general group text message from user {user_id_group} in chat {chat_id_group} for context: 'Чтение общих сообщений чата' is DISABLED.")
    # else: command, will be handled by command handlers


# --- Startup and Shutdown ---

async def on_startup(bot: Bot): # Parameter name changed back to 'bot'
    """Выполняется при запуске бота."""
    global BOT_USERNAME
    try:
        me = await bot.get_me() # Use the passed Bot instance
        BOT_USERNAME = me.username
        if not BOT_USERNAME:
             logger.warning("Bot username is empty or None after get_me(). Mentions might not work.")
             BOT_USERNAME = "UnknownBot"

        logger.info(f"Bot @{BOT_USERNAME} (ID: {me.id}) started successfully!")

        if PROVIDERS and PROVIDER_MAP:
             logger.info("Scheduling random message task...")
             asyncio.create_task(send_random_messages(bot), name="RandomMessageSender") # Pass the Bot instance
             logger.info("Random message task scheduled.")
        else:
             logger.warning("PROVIDERS или PROVIDER_MAP пусты. Задача случайных сообщений НЕ запланирована.")

        logger.info("Scheduling periodic state saving task...")
        asyncio.create_task(save_state_periodically(), name="StateSaver")
        logger.info("Periodic state saving task scheduled.")

    except Exception as e:
        logger.critical(f"Failed to get bot info or schedule tasks on startup: {e}", exc_info=True)

async def on_shutdown(bot: Bot): # Parameter name changed back to 'bot'
    """Выполняется при остановке бота."""
    logger.info("Bot is shutting down...")
    try:
        logger.info("Saving all contexts and chat settings before shutdown...")
        # Сохраняем долгосрочную память
        for key, context_obj_sd in storage.contexts.items():
            context_obj_sd.save_long_term()
        logger.info("All contexts saved.")

        # Сохраняем все активные настройки чатов
        # Объединяем все ключи, для которых есть настройки, чтобы ничего не потерять
        all_active_setting_keys = set(
            list(storage.models.keys()) +
            list(storage.image_models.keys()) +
            list(storage.reply_modes.keys()) +
            list(storage.context_types_enabled.keys()) +
            list(storage.read_general_chat_messages_enabled.keys()) + # Include new setting
            list(storage.short_term_limits.keys()) + # Include new setting
            list(storage.developer_modes.keys())
        )
        for key in all_active_setting_keys:
            storage._save_settings_for_key(key)
        logger.info("All chat settings saved.")


        if bot and bot.session: # Use the passed Bot instance
             await bot.session.close()
             logger.info("Bot session closed.")
        else:
             logger.info("Bot session was not active or already closed.")
    except Exception as e:
         logger.error(f"Error closing bot session or saving state: {e}", exc_info=True)
    logger.info("Shutdown process finished.")


if __name__ == "__main__":
    if bot is None: # Check the global bot initialized at the top
        logger.critical("Bot instance is None before starting polling. Exiting.")
        exit(1)

    dp.include_router(router)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown) # Corrected to on_shutdown

    try:
        logger.info("Starting bot polling...")
        dp.run_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except KeyboardInterrupt:
        logger.info("Bot stopped by KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Critical error during polling: {e}", exc_info=True)
    finally:
        logger.info("Polling finished.")
