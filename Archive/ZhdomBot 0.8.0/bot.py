# -*- coding: utf-8 -*-
# COPYRIGHT ZhdomDev
# Optimized version

__version__ = "0.8.0" # Версия бота (увеличена)

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
        IMAGE_PROVIDERS_CONFIG, # НОВОЕ: Словарь с конфигурацией провайдеров для изображений
        DEFAULT_MODEL, # Модель по умолчанию для текста
        DEFAULT_IMAGE_MODEL, # НОВОЕ: Модель по умолчанию для изображений
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
    print("Пожалуйста, убедитесь, что файл config.py существует и содержит необходимые переменные (включая IMAGE_PROVIDERS_CONFIG и DEFAULT_IMAGE_MODEL), кроме TELEGRAM_TOKEN.")
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
        escaped_code = escape_html(code)
        if lang:
            # Добавляем класс языка для возможной подсветки на клиенте
            return f'<pre><code class="language-{escape_html(lang)}">{escaped_code}</code></pre>'
        else:
            return f'<pre><code>{escaped_code}</code></pre>'
    # Паттерн для ``` optional_lang \n code ```
    text = re.sub(r'```(\w*)\s*\n(.*?)\n```', replace_code_block, text, flags=re.DOTALL)
    # Паттерн для ```code``` (без языка и переносов строк)
    text = re.sub(r'```(.*?)```', lambda m: f'<pre><code>{escape_html(m.group(1))}</code></pre>', text, flags=re.DOTALL)


    # 3. Обрабатываем инлайн код (`...`) -> <code>...</code>
    # Экранируем содержимое
    text = re.sub(r'`([^`]+?)`', lambda m: f'<code>{escape_html(m.group(1))}</code>', text)

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
        safe_url = escape_html(url).replace('"', '&quot;')
        # Экранируем текст ссылки
        safe_text = escape_html(link_text)
        return f'<a href="{safe_url}">{safe_text}</a>'
    text = re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)

    # 8. После всех замен, экранируем оставшиеся символы <, >, & которые не являются частью тегов
    # Это сложнее, т.к. нужно не затронуть уже созданные теги.
    # Простой подход: экранировать все и затем "разэкранировать" теги. Ненадежно.
    # Более сложный: использовать HTML парсер. Избыточно для Telegram.
    # Компромисс: Полагаемся на то, что LLM вернет валидный Markdown и наша конвертация корректна.
    # Дополнительное экранирование после замен может сломать теги.

    return text


async def safe_send_message(
    target: Message, # Используем Message напрямую для ответа/отправки
    text: str,
    parse_mode: Optional[str] = ParseMode.HTML, # По умолчанию HTML
    reply_to_message: bool = True, # Отвечать ли на сообщение по умолчанию
    bot_instance: Optional[Bot] = None, # ИСПРАВЛЕНО: Добавлен необязательный параметр bot_instance
    **kwargs: Any
) -> Optional[Message]:
    """
    Безопасная отправка сообщений с обработкой ошибок форматирования и длины.
    """
    max_length = 4096 # Максимальная длина сообщения в Telegram
    sent_message = None
    chat_id = target.chat.id # Получаем chat_id из target

    if not isinstance(text, str):
        logger.warning(f"safe_send_message received non-string: {type(text)}. Converting.")
        text = str(text)

    # ИСПРАВЛЕНО: Определяем метод отправки с учетом bot_instance
    actual_bot = bot_instance or target.bot # Используем переданный bot_instance или bot из target
    if not actual_bot:
        logger.error(f"safe_send_message: Cannot determine Bot instance for chat {chat_id}.")
        return None

    send_method = None
    send_args = kwargs.copy() # Копируем kwargs, чтобы не изменять оригинал

    if reply_to_message and hasattr(target, 'reply'):
        # Если нужно ответить и у target есть метод reply (это реальное сообщение)
        send_method = target.reply
        # target.reply уже знает chat_id и message_id, не нужно добавлять их в send_args
    else:
        # Если не нужно отвечать или у target нет метода reply (это заглушка)
        # Используем метод send_message переданного бота
        send_method = actual_bot.send_message
        # ИСПРАВЛЕНО: Не добавляем chat_id в send_args здесь, т.к. он будет передан явно ниже
        # send_args['chat_id'] = chat_id # УДАЛЕНО: Передаем chat_id явно ниже

    try:
        # 1. Попытка отправить с указанным parse_mode (обычно HTML)
        if parse_mode == ParseMode.HTML:
            processed_text = convert_markdown_to_html(text)
        else:
            processed_text = text

        # Разбиваем на части, если текст слишком длинный
        for i in range(0, len(processed_text), max_length):
            chunk = processed_text[i:i + max_length]
            # ИСПРАВЛЕНО: Вызываем определенный выше send_method с правильными аргументами
            # Pass chat_id explicitly ONLY if using send_message, not reply
            if send_method == actual_bot.send_message:
                 # ИСПРАВЛЕНО: Удаляем chat_id из send_args using pop
                 send_args.pop('chat_id', None)
                 sent_message = await send_method(
                     chat_id=chat_id, # Pass chat_id explicitly
                     text=chunk, # Use 'text' parameter name
                     parse_mode=parse_mode,
                     **send_args # Pass other arguments
                 )
            else: # Assumes send_method is target.reply
                 sent_message = await send_method(
                     text=chunk, # Use 'text' parameter name
                     parse_mode=parse_mode,
                     **send_args # Pass other arguments
                 )

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
                    # ИСПРАВЛЕНО: Повторная попытка также через send_method
                    if send_method == actual_bot.send_message:
                         # ИСПРАВЛЕНО: Удаляем chat_id из send_args
                         send_args.pop('chat_id', None)
                         sent_message = await send_method(
                             chat_id=chat_id, # Pass chat_id explicitly
                             text=chunk, # Use 'text' parameter name
                             parse_mode=None, # No parse mode
                             **send_args # Pass other arguments
                         )
                    else: # Assumes send_method is target.reply
                         sent_message = await send_method(
                             text=chunk, # Use 'text' parameter name
                             parse_mode=None, # No parse mode
                             **send_args
                         )
                    if len(plain_text) > max_length:
                        await asyncio.sleep(0.3)
                return sent_message
            except Exception as e_final:
                logger.exception(f"Failed to send message even without parse_mode: {e_final}")
                try:
                    error_report = f"⚠️ Не удалось отправить отформатированный ответ.\nОшибка: {escape_html(str(e))}\nПопытка без форматирования тоже не удалась: {escape_html(str(e_final))}"
                    # ИСПРАВЛЕНО: Отправка сообщения об ошибке тоже через send_method
                    # Send error report without parse mode
                    if send_method == actual_bot.send_message:
                         # ИСПРАВЛЕНО: Удаляем chat_id из send_args
                         send_args.pop('chat_id', None)
                         await send_method(
                             chat_id=chat_id, # Pass chat_id explicitly
                             text=error_report[:max_length],
                             parse_mode=None,
                             **send_args # Передаем базовые аргументы
                         )
                    else: # Assumes send_method is target.reply
                         await send_method(
                             text=error_report[:max_length],
                             parse_mode=None,
                             **send_args # Передаем базовые аргументы
                         )
                except Exception:
                    logger.error("Failed even to send the error message.")
                return None
        else:
            logger.error(f"Unhandled BadRequest error: {e}")
            return None

    except TelegramRetryAfter as e:
        logger.warning(f"Rate limit hit for chat {chat_id}. Retrying after {e.retry_after}s.")
        await asyncio.sleep(e.retry_after)
        # ИСПРАВЛЕНО: Передаем bot_instance при рекурсивном вызове
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
            # ИСПРАВЛЕНО: Отправка сообщения об ошибке тоже через send_method
            # Send error report without parse mode
            if send_method == actual_bot.send_message:
                 # ИСПРАВЛЕНО: Удаляем chat_id из send_args
                 send_args.pop('chat_id', None)
                 await send_method(
                     chat_id=chat_id, # Pass chat_id explicitly
                     text=f"⚠️ Произошла ошибка Telegram API: {escape_html(str(e))}",
                     parse_mode=None,
                     **send_args
                 )
            else: # Assumes send_method is target.reply
                 await send_method(
                     text=f"⚠️ Произошла ошибка Telegram API: {escape_html(str(e))}",
                     parse_mode=None,
                     **send_args
                 )
        except Exception:
            logger.error("Failed to send Telegram API error message.")
        return None

    except Exception as e:
        logger.exception(f"Unexpected error in safe_send_message: {e}")
        try:
             # ИСПРАВЛЕНО: Отправка сообщения об ошибке тоже через send_method
             # Send error report without parse mode
            if send_method == actual_bot.send_message:
                 # ИСПРАВЛЕНО: Удаляем chat_id из send_args
                 send_args.pop('chat_id', None)
                 await send_method(
                     chat_id=chat_id, # Pass chat_id explicitly
                     text=f"⚠️ Произошла непредвиденная ошибка: {escape_html(str(e))}",
                     parse_mode=None,
                     **send_args
                 )
            else: # Assumes send_method is target.reply
                 await send_method(
                     text=f"⚠️ Произошла непредвиденная ошибка: {escape_html(str(e))}",
                     parse_mode=None,
                     **send_args
                 )
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
        # ИСПРАВЛЕНО: Этот метод был ошибочно удален или не добавлен ранее.
        # Он нужен для send_random_messages, если не используем get_full_context.
        # Но т.к. мы используем legend и long_term напрямую в send_random_messages,
        # этот метод технически не требуется там, но может быть полезен в других местах.
        # Оставляем его для полноты класса.
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
        self.image_models: Dict[int, str] = {} # НОВОЕ: chat_id -> image_model_key (e.g., "DefaultImage")
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
        """Очищает всю память (краткосрочную и долгосрочную)."""
        key = self._get_key(chat_id, user_id, is_group)
        if key in self.contexts:
            self.contexts[key].clear_all()
        else:
             logger.warning(f"Attempted to clear full context for non-existent key {key}")

    def get_available_models(self) -> List[str]:
        """Возвращает список доступных имен моделей для текста из конфига."""
        return list(PROVIDERS.keys()) if PROVIDERS else []

    def get_available_image_models(self) -> List[str]: # НОВОЕ: Получение списка моделей для изображений
        """Возвращает список доступных имен моделей для изображений из конфига."""
        return list(IMAGE_PROVIDERS_CONFIG.keys()) if IMAGE_PROVIDERS_CONFIG else []


    def get_model_key(self, chat_id: int) -> str:
        """Возвращает ключ текущей модели для текста для чата (или дефолтную)."""
        # Используем chat_id как ключ для настроек модели, даже в ЛС,
        # чтобы настройки модели были общими для чата/ЛС, а не для пользователя.
        # Если нужна модель на пользователя, нужно изменить ключ здесь и в set_model.
        return self.models.get(chat_id, DEFAULT_MODEL)

    def set_model(self, chat_id: int, model_key: str):
        """Устанавливает модель для текста для чата."""
        if PROVIDERS and model_key in PROVIDERS:
            self.models[chat_id] = model_key
            logger.info(f"Text model for chat {chat_id} set to '{model_key}'")
        else:
            logger.warning(f"Attempted to set invalid text model '{model_key}' for chat {chat_id}. Available: {list(PROVIDERS.keys())}")

    def get_image_model_key(self, chat_id: int) -> str: # НОВОЕ: Получение ключа модели для изображений
        """Возвращает ключ текущей модели для изображений для чата (или дефолтную)."""
        return self.image_models.get(chat_id, DEFAULT_IMAGE_MODEL)

    def set_image_model(self, chat_id: int, model_key: str): # НОВОЕ: Установка модели для изображений
        """Устанавливает модель для изображений для чата."""
        if IMAGE_PROVIDERS_CONFIG and model_key in IMAGE_PROVIDERS_CONFIG:
            self.image_models[chat_id] = model_key
            logger.info(f"Image model for chat {chat_id} set to '{model_key}'")
        else:
            logger.warning(f"Attempted to set invalid image model '{model_key}' for chat {chat_id}. Available: {list(IMAGE_PROVIDERS_CONFIG.keys())}")


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
# Объявляем глобальную переменную bot здесь, чтобы она была доступна в on_startup/on_shutdown
bot: Optional[Bot] = None
try:
    # Проверяем наличие токена (imпортированного из secrets.py)
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is not defined in secrets.py")
    # Инициализируем бота и присваиваем его глобальной переменной
    bot_instance = Bot(token=TELEGRAM_TOKEN, default=default_properties)
    bot = bot_instance # Присваиваем созданный экземпляр глобальной переменной bot
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
    image_model_selection = State() # НОВОЕ: Состояние для выбора модели изображений


# --- Background Tasks ---

async def send_random_messages(bot_instance: Bot): # Принимаем объект Bot
    """Периодически отправляет случайные сообщения в активные чаты."""
    await asyncio.sleep(15) # Небольшая задержка перед первым запуском
    logger.info("Starting random message task...")
    while True:
        await asyncio.sleep(30)  # Проверяем каждые 30 секунд (можно увеличить для прода)
        current_time = time.time()

        # Создаем копию ключей, чтобы избежать проблем при удалении во время итерации
        active_chat_ids = list(storage.active_chats.keys())
        # logger.debug(f"Random message check. Active chats: {active_chat_ids}") # Отладка

        for chat_id in active_chat_ids:
            # Проверяем, существует ли чат еще в словаре (мог быть удален из-за ошибки)
            if chat_id not in storage.active_chats:
                # logger.debug(f"Chat {chat_id} no longer in active_chats dict. Skipping.")
                continue

            settings = storage.active_chats[chat_id]

            # Добавлено логирование проверки активности
            # logger.debug(f"Checking chat {chat_id}. Active: {settings.get('active')}. Interval: {settings.get('interval')}. Last sent: {settings.get('last_sent')}")

            if not settings.get('active'): # Используем .get() для безопасности
                continue

            # Проверяем интервал
            interval = settings.get('interval', RANDOM_MESSAGE_INTERVAL)
            if not isinstance(interval, (int, float)) or interval <= 0:
                logger.warning(f"Invalid interval {interval} for chat {chat_id}. Using default {RANDOM_MESSAGE_INTERVAL}s.")
                interval = RANDOM_MESSAGE_INTERVAL
                settings['interval'] = interval # Исправляем в настройках

            last_sent = settings.get('last_sent', 0.0)
            time_since_last = current_time - last_sent

            # Добавлено логирование проверки времени
            # logger.debug(f"Chat {chat_id}: Time since last: {time_since_last:.2f}s. Interval: {interval}s.")

            if time_since_last >= interval:
                logger.info(f"Attempting to send random message to chat {chat_id} (Interval passed)")
                try:
                    # Получаем контекст группы (случайные сообщения только для групп)
                    # Передаем user_id=0 т.к. это не от конкретного пользователя
                    # ИСПРАВЛЕНО: Теперь получаем контекст, который будет содержать собранную историю
                    context = storage.get_context(chat_id=chat_id, user_id=0, is_group=True)
                    model_key = storage.get_model_key(chat_id) # Получаем ключ модели для чата
                    logger.info(f"Chat {chat_id}: Using model '{model_key}' for random message.")

                    if not PROVIDERS:
                        logger.error(f"Chat {chat_id}: PROVIDERS dictionary is empty. Cannot select model.")
                        continue

                    if model_key not in PROVIDERS:
                        logger.warning(f"Chat {chat_id}: Invalid model '{model_key}', using default '{DEFAULT_MODEL}'")
                        model_key = DEFAULT_MODEL
                        if model_key not in PROVIDERS:
                            logger.error(f"Chat {chat_id}: Default model '{DEFAULT_MODEL}' not found in PROVIDERS. Skipping.")
                            continue

                    provider_config = PROVIDERS[model_key]
                    provider_name = provider_config.get("provider")
                    model_identifier = provider_config.get("model_name")

                    if not provider_name or not model_identifier:
                         logger.error(f"Chat {chat_id}: Incomplete provider config for model '{model_key}'. Skipping.")
                         continue

                    if provider_name not in PROVIDER_MAP:
                         logger.error(f"Chat {chat_id}: Provider class '{provider_name}' not found. Skipping.")
                         continue

                    provider_class = PROVIDER_MAP[provider_name]

                    # --- Формируем промпт для LLM, теперь включая собранную short_term историю ---
                    # ИСПРАВЛЕНО: Используем полный контекст, который включает short_term
                    messages = context.get_full_context()
                    prompt_info = f"legend ({len(context.legend)}) + long_term ({len(context.long_term)}) + short_term ({len(context.short_term)})"

                    # ИСПРАВЛЕНО: Удалена заметка об ошибке получения истории, т.к. теперь используем внутреннюю историю
                    # if history_fetch_error:
                    #      messages.append({
                    #         "role": "system",
                    #         "content": f"[System Note: {history_fetch_error}]"
                    #     })
                    #      prompt_info += " + history_fetch_note"

                    messages.append({
                        "role": "user",
                        "content": (
                            "Ты участник этого группового чата. "
                            "Используя контекст (включая последние сообщения), " # ИСПРАВЛЕНО: Упоминаем последние сообщения
                            "придумай СЛУЧАЙНОЕ, КОРОТКОЕ, НЕЙТРАЛЬНОЕ или ЗАБАВНОЕ сообщение для поддержания беседы. "
                            "Твой ответ должен быть ТОЛЬКО текстом самого сообщения. "
                            "НЕ представляйся, НЕ упоминай, что ты бот, НЕ используй префиксы типа 'Сообщение:', НЕ ссылайся на этот промпт. "
                            "Просто напиши сообщение так, как будто ты обычный участник чата. "
                            "Соблюдай правила форматирования из системных сообщений."
                        )
                    })
                    prompt_info += " + final_instruction"

                    logger.info(f"Chat {chat_id}: Sending request to {provider_name} ({model_identifier}). Prompt includes: {prompt_info}. Total messages: {len(messages)}")
                    # logger.debug(f"Chat {chat_id}: Full prompt messages: {messages}") # Отладка - может быть очень длинным

                    response = await g4f.ChatCompletion.create_async(
                        model=model_identifier,
                        messages=messages,
                        provider=provider_class,
                        timeout=45 # Увеличим таймаут немного
                    )

                    response_text = str(response).strip()
                    logger.info(f"Chat {chat_id}: Received response from LLM. Length: {len(response_text)}. Content (start): {response_text[:70]}...")

                    if response_text:
                        temp_message_stub = types.Message(
                            message_id=0,
                            date=int(time.time()),
                            chat=types.Chat(id=chat_id, type=ChatType.GROUP),
                            from_user=None # Заглушка не имеет from_user
                            # Важно: у заглушки нет bot
                        )
                        logger.info(f"Chat {chat_id}: Attempting to send generated message via safe_send_message...")
                        # ИСПРАВЛЕНО: Передаем bot_instance в safe_send_message
                        sent_msg = await safe_send_message(
                            target=temp_message_stub,
                            text=response_text,
                            parse_mode=ParseMode.HTML,
                            reply_to_message=False, # Важно: не отвечаем на сообщение
                            bot_instance=bot_instance # Передаем активный экземпляр бота
                        )
                        if sent_msg:
                            settings['last_sent'] = current_time
                            logger.info(f"Chat {chat_id}: Successfully sent random message.")
                            # ИСПРАВЛЕНО: Добавляем отправленное ботом сообщение в short_term контекст
                            context.add_message("assistant", response_text, "short")
                            logger.info(f"Chat {chat_id}: Added assistant random message to short-term context.")
                        else:
                            logger.error(f"Chat {chat_id}: Failed to send random message using safe_send_message.")
                            # Не обновляем last_sent, чтобы попробовать снова раньше
                    else:
                        logger.warning(f"Chat {chat_id}: Received empty random message response from {model_key}.")
                        # Не добавляем пустой ответ в контекст
                        # Не обновляем last_sent

                # ИСПРАВЛЕНО: Добавлены специфические обработчики для сетевых ошибок и таймаутов
                except (aiohttp.client_exceptions.ClientOSError, asyncio.TimeoutError) as e:
                    logger.error(f"Chat {chat_id}: Network or Timeout error contacting LLM provider {provider_name} ({model_identifier}): {e}")
                    settings['last_sent'] = current_time + 60 # Retry after 60 seconds
                    await asyncio.sleep(1) # Small delay before next iteration

                # ИСПРАВЛЕНО: Добавлена обработка ResponseStatusError
                except g4f.errors.ResponseStatusError as e:
                    logger.error(f"Chat {chat_id}: Provider {provider_name} ({model_identifier}) returned status error: {e}")
                    settings['last_sent'] = current_time + 60 # Retry after 60 seconds
                    await asyncio.sleep(1) # Small delay before next iteration


                except TelegramForbiddenError:
                    logger.warning(f"Chat {chat_id}: Bot forbidden, removing from active list.")
                    if chat_id in storage.active_chats:
                        del storage.active_chats[chat_id]
                except TelegramRetryAfter as e:
                    logger.warning(f"Chat {chat_id}: Rate limit hit during random message, retry after {e.retry_after}s")
                    await asyncio.sleep(e.retry_after)
                except Exception as e:
                    logger.exception(f"Chat {chat_id}: Unhandled error processing random message: {e}")
                    settings['last_sent'] = current_time + 60 # Retry after 60 seconds
                    await asyncio.sleep(1) # Small delay before next iteration

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
        await safe_send_message(message, "⚠️ Ошибка конфигурации: Список моделей для текста пуст.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return

    if model_key not in PROVIDERS:
        logger.error(f"Invalid text model key '{model_key}' found for chat {chat_id}. Reverting to default '{DEFAULT_MODEL}'.")
        model_key = DEFAULT_MODEL
        if model_key not in PROVIDERS:
             logger.error(f"Default text model '{DEFAULT_MODEL}' not found in PROVIDERS. Cannot process question.")
             await safe_send_message(message, f"⚠️ Ошибка конфигурации: Модель для текста по умолчанию '{DEFAULT_MODEL}' не найдена.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
             return
        storage.set_model(chat_id, model_key) # Сохраняем дефолтную модель для чата

    provider_config = PROVIDERS[model_key]
    provider_name = provider_config.get("provider")
    model_identifier = provider_config.get("model_name")

    if not provider_name or not model_identifier:
         logger.error(f"Incomplete provider config for text model '{model_key}' in chat {chat_id}. Missing 'provider' or 'model_name'.")
         await safe_send_message(message, f"⚠️ Ошибка конфигурации для модели текста '{model_key}'.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
         return

    if provider_name not in PROVIDER_MAP:
         logger.error(f"Provider class '{provider_name}' configured for text model '{model_key}' not found in PROVIDER_MAP.")
         await safe_send_message(message, f"⚠️ Ошибка конфигурации: провайдер {provider_name} для модели текста не найден.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
         return

    provider_class = PROVIDER_MAP[provider_name]

    # Добавляем вопрос пользователя в контекст
    context.add_message("user", question, memory_type)

    # Отправляем сообщение "Думаю..."
    # Передаем bot из message, если он есть (для обычных сообщений)
    status_msg = await safe_send_message(
        message,
        random.choice(THINKING_MESSAGES) if THINKING_MESSAGES else "🤔 Думаю...",
        reply_to_message=True,
        parse_mode=None, # Просто текст
        bot_instance=message.bot # Передаем бота из сообщения
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
                # Используем бота из status_msg для удаления
                await status_msg.delete()
            except TelegramAPIError as e:
                 logger.warning(f"Could not delete status message {status_msg.message_id}: {e}")
            except AttributeError:
                 logger.warning(f"Could not delete status message {status_msg.message_id}: delete method not found (possibly already deleted or bot instance issue).")


        # Отправляем ответ пользователю (safe_send_message сама разобьет на части)
        # Передаем bot из message
        await safe_send_message(
            target=message,
            text=full_response,
            parse_mode=ParseMode.HTML, # Ожидаем, что модель вернет Markdown, который конвертируется в HTML
            reply_to_message=True,
            bot_instance=message.bot
        )

    except asyncio.TimeoutError:
         logger.error(f"Timeout error processing question with {model_key} for user {user_id} in chat {chat_id}")
         if status_msg: # Попытаемся удалить "Думаю" и здесь
             try: await status_msg.delete()
             except Exception as e_del: logger.warning(f"Could not delete status message {status_msg.message_id} after timeout: {e_del}")
         await safe_send_message(message, f"⚠️ Модель ({escape_html(model_key)}) не ответила вовремя.", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot
         # Удаляем последний вопрос пользователя из контекста, так как ответа не было
         if memory_type == "short" and context.short_term and context.short_term[-1]["role"] == "user":
              context.short_term.pop()
              logger.info("Removed user question from short-term context after timeout.")
         elif memory_type == "long" and context.long_term and context.long_term[-1]["role"] == "user":
              context.long_term.pop()
              logger.info("Removed user question from long-term context after timeout.")

    # ИСПРАВЛЕНО: Добавлен специфический обработчик для ошибок статуса от провайдера
    except g4f.errors.ResponseStatusError as e:
        logger.error(f"Provider {provider_name} ({model_identifier}) returned status error for user {user_id} in chat {chat_id}: {e}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, f"⚠️ Провайдер модели ({escape_html(model_key)}) вернул ошибку: {escape_html(str(e))}\nПопробуйте позже или выберите другую модель.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        # Удаляем последний вопрос пользователя из контекста при ошибке
        if memory_type == "short" and context.short_term and context.short_term[-1]["role"] == "user":
             context.short_term.pop()
             logger.info("Removed user question from short-term context after error.")
        elif memory_type == "long" and context.long_term and context.long_term[-1]["role"] == "user":
             context.long_term.pop()
             logger.info("Removed user question from long-term context after error.")

    # ИСПРАВЛЕНО: Добавлен специфический обработчик для сетевых ошибок
    except aiohttp.client_exceptions.ClientOSError as e:
        logger.error(f"Network error contacting LLM provider {provider_name} ({model_identifier}) for user {user_id} in chat {chat_id}: {e}")
        if status_msg: # Попытаемся удалить "Думаю" и при других ошибках
             try: await status_msg.delete()
             except Exception as e_del: logger.warning(f"Could not delete status message {status_msg.message_id} after error: {e_del}")
        await safe_send_message(message, f"⚠️ Ошибка сети при обращении к модели: {escape_html(str(e))}", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot
        # Удаляем последний вопрос пользователя из контекста при ошибке
        if memory_type == "short" and context.short_term and context.short_term[-1]["role"] == "user":
             context.short_term.pop()
             logger.info("Removed user question from short-term context after error.")
        elif memory_type == "long" and context.long_term and context.long_term[-1]["role"] == "user":
             context.long_term.pop()
             logger.info("Removed user question from long-term context after error.")

    except Exception as e:
        logger.exception(f"Unhandled error processing question with {model_key} for user {user_id} in chat {chat_id}")
        if status_msg: # Попытаемся удалить "Думаю" и при других ошибках
             try: await status_msg.delete()
             except Exception as e_del: logger.warning(f"Could not delete status message {status_msg.message_id} after error: {e_del}")
        await safe_send_message(message, f"⚠️ Произошла непредвиденная ошибка при обработке вашего запроса: {escape_html(str(e))}", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot
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
    logger.info(f"Handling command {command.command} from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
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
            await safe_send_message(message, "❌ Исходное сообщение не содержит текста.", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot
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
                 await safe_send_message(message, "ℹ️ Укажите текст, который нужно добавить в долгосрочную память.", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot
                 return
            # А для /ask - это ошибка
            else:
                await safe_send_message(message, "❌ Пожалуйста, укажите ваш вопрос после команды.", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot
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
            parse_mode=None,
            bot_instance=message.bot # Передаем bot
        )
        return

    # 4. Вызов основного обработчика запросов
    await process_question(message, full_question_text, memory_type)


# --- Command Handlers ---

@router.message(Command("start", "help", "старт", "помощь"))
async def cmd_start(message: types.Message, bot: Bot): # Добавляем bot как аргумент
    """Обработчик команд /start и /help."""
    logger.info(f"Handling command start/help from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    # Получаем имя пользователя бота асинхронно, если оно еще не установлено
    global BOT_USERNAME
    if not BOT_USERNAME:
        try:
            me = await bot.get_me() # Используем переданный bot
            BOT_USERNAME = me.username or "UnknownBot"
            logger.info(f"Got bot username in cmd_start: @{BOT_USERNAME}")
        except Exception as e:
            logger.error(f"Failed to get bot username in cmd_start: {e}")
            BOT_USERNAME = "UnknownBot" # Запасной вариант

    # Используем f-string для подстановки версии и имени бота
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
        "• <code>/model</code> - Выбрать другую модель ИИ для ответов (текст).\n" # ИСПРАВЛЕНО: Уточнение
        "• <code>/imagemodel</code> - Выбрать другую модель ИИ для генерации изображений.\n\n" # НОВОЕ: Команда
        "<b>Для групповых чатов:</b>\n"
        f"• <code>/sglypa [минуты]</code> - Включить/выключить режим случайных сообщений с заданным интервалом (например, <code>/sglypa 30</code> для 30 минут). Без аргумента - переключает режим.\n"
        f"• Просто <b>упомяните меня</b> (@{BOT_USERNAME}) в начале сообщения, чтобы задать вопрос.\n\n" # Используем полученное имя
        "<i>Подсказка: Долгосрочная память полезна для задания общего контекста или инструкций боту.</i>"
    )
    # Отправляем с parse_mode=HTML, так как используем HTML теги
    # Передаем bot из аргумента
    await safe_send_message(message, help_text, reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=bot)

# Используем общий обработчик для /ask и его псевдонимов
@router.message(Command("ask", "аск", "bot", "бот", "ждёмби", "zhdomby", "ждемби", "zhdyomby"))
async def cmd_ask(message: types.Message, command: CommandObject):
    """Обработчик команды /ask и её псевдонимов."""
    logger.info(f"Handling command ask from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    await handle_command_with_question(
        message=message,
        command=command,
        allow_reply=True,    # /ask может отвечать на сообщения
        memory_type="short"  # Использует краткосрочную память
    )

@router.message(Command("long", "лонг"))
async def cmd_long(message: types.Message, command: CommandObject):
    """Обработчик команды /long."""
    logger.info(f"Handling command long from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    await handle_command_with_question(
        message=message,
        command=command,
        allow_reply=False,   # /long не отвечает на сообщения, только добавляет контекст
        memory_type="long"   # Использует долгосрочную память
    )

@router.message(Command("clear"))
async def cmd_clear(message: types.Message):
    """Очищает краткосрочную память."""
    logger.info(f"Handling command clear from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return # Игнорируем если нет пользователя
    is_group = is_group_chat(message)
    storage.clear_short(message.chat.id, message.from_user.id, is_group)
    await safe_send_message(message, "🧽 Краткосрочный контекст очищен!", reply_to_message=False, parse_mode=None, bot_instance=message.bot) # Передаем bot

@router.message(Command("fullreset"))
async def cmd_fullreset(message: types.Message):
    """Очищает всю память (краткосрочную и долгосрочную)."""
    logger.info(f"Handling command fullreset from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not message.from_user: return # Игнорируем если нет пользователя
    is_group = is_group_chat(message)
    storage.clear_all(message.chat.id, message.from_user.id, is_group)
    await safe_send_message(message, "♻️ Вся память (краткосрочная и долгосрочная) очищена!", reply_to_message=False, parse_mode=None, bot_instance=message.bot) # Передаем bot

@router.message(Command("model"))
async def cmd_model(message: types.Message, state: FSMContext):
    """Начинает процесс выбора модели ИИ для текста."""
    logger.info(f"Handling command model from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    available_models = storage.get_available_models() # Получаем только текстовые модели
    if not available_models:
         await safe_send_message(message, "❌ Нет доступных моделей для текста в конфигурации.", reply_to_message=False, parse_mode=None, bot_instance=message.bot) # Передаем bot
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
        f"🤖 Выберите модель ИИ для <b>текста</b>.\nТекущая модель: <b>{escape_html(current_model)}</b>", # Экранируем имя модели на всякий случай
        reply_to_message=False,
        reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True),
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot # Передаем bot
    )
    # Устанавливаем состояние ожидания выбора текстовой модели
    await state.set_state(Form.model_selection)

@router.message(Form.model_selection) # Обработчик для состояния выбора текстовой модели
async def process_model_selection(message: types.Message, state: FSMContext):
    """Завершает процесс выбора модели ИИ для текста."""
    logger.info(f"Handling text model selection from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    # Проверяем, что сообщение текстовое
    if not message.text:
        await safe_send_message(message, "❌ Пожалуйста, используйте кнопки для выбора модели текста.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None, bot_instance=message.bot) # Передаем bot
        # Не очищаем состояние, даем еще попытку
        return

    # Убираем маркер ✅ из текста кнопки, если он есть
    selected_model_key = message.text.replace("✅ ", "").strip()

    if selected_model_key in storage.get_available_models(): # Проверяем среди доступных текстовых моделей
        storage.set_model(message.chat.id, selected_model_key)
        await safe_send_message(
            message,
            f"✅ Модель для текста успешно изменена на <b>{escape_html(selected_model_key)}</b>!", # Экранируем имя модели
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(), # Убираем клавиатуру
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot # Передаем bot
        )
    else:
        await safe_send_message(
            message,
            "❌ Некорректный выбор модели для текста. Пожалуйста, используйте кнопки.",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=None,
            bot_instance=message.bot # Передаем bot
        )

    # Очищаем состояние FSM в любом случае
    await state.clear()

# НОВОЕ: Обработчик команды /imagemodel
@router.message(Command("imagemodel"))
async def cmd_imagemodel(message: types.Message, state: FSMContext):
    """Начинает процесс выбора модели ИИ для изображений."""
    logger.info(f"Handling command imagemodel from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    available_image_models = storage.get_available_image_models() # Получаем только графические модели
    if not available_image_models:
         await safe_send_message(message, "❌ Нет доступных моделей для изображений в конфигурации.", reply_to_message=False, parse_mode=None, bot_instance=message.bot) # Передаем bot
         return

    builder = ReplyKeyboardBuilder()
    current_image_model = storage.get_image_model_key(message.chat.id)

    for model_key in available_image_models:
        # Добавляем маркер к текущей выбранной модели
        text = f"✅ {model_key}" if model_key == current_image_model else model_key
        builder.add(KeyboardButton(text=text))

    builder.adjust(2) # Располагаем кнопки по 2 в ряд

    await safe_send_message(
        message,
        f"🖼 Выберите модель ИИ для <b>изображений</b>.\nТекущая модель: <b>{escape_html(current_image_model)}</b>", # Экранируем имя модели на всякий случай
        reply_to_message=False,
        reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True),
        parse_mode=ParseMode.HTML,
        bot_instance=message.bot # Передаем bot
    )
    # Устанавливаем состояние ожидания выбора графической модели
    await state.set_state(Form.image_model_selection)

# НОВОЕ: Обработчик для состояния выбора графической модели
@router.message(Form.image_model_selection)
async def process_image_model_selection(message: types.Message, state: FSMContext):
    """Завершает процесс выбора модели ИИ для изображений."""
    logger.info(f"Handling image model selection from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    # Проверяем, что сообщение текстовое
    if not message.text:
        await safe_send_message(message, "❌ Пожалуйста, используйте кнопки для выбора модели изображений.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None, bot_instance=message.bot) # Передаем bot
        # Не очищаем состояние, даем еще попытку
        return

    # Убираем маркер ✅ из текста кнопки, если он есть
    selected_model_key = message.text.replace("✅ ", "").strip()

    if selected_model_key in storage.get_available_image_models(): # Проверяем среди доступных графических моделей
        storage.set_image_model(message.chat.id, selected_model_key)
        await safe_send_message(
            message,
            f"✅ Модель для изображений успешно изменена на <b>{escape_html(selected_model_key)}</b>!", # Экранируем имя модели
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(), # Убираем клавиатуру
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot # Передаем bot
        )
    else:
        await safe_send_message(
            message,
            "❌ Некорректный выбор модели для изображений. Пожалуйста, используйте кнопки.",
            reply_to_message=False,
            reply_markup=ReplyKeyboardRemove(),
            parse_mode=None,
            bot_instance=message.bot # Передаем bot
        )

    # Очищаем состояние FSM в любом случае
    await state.clear()


@router.message(Command("image", "img", "имейдж", "имж", "имг"))
async def cmd_image(message: types.Message, command: CommandObject, bot: Bot): # Добавляем bot как аргумент
    """Генерирует изображение по текстовому промпту."""
    logger.info(f"Handling command image from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    prompt = None
    if command.args:
        prompt = command.args.strip()
    elif message.reply_to_message and (message.reply_to_message.text or message.reply_to_message.caption):
        prompt = (message.reply_to_message.text or message.reply_to_message.caption).strip()

    if not prompt:
        await safe_send_message(message, "❌ Пожалуйста, укажите описание для изображения после команды или ответьте на сообщение с текстом.", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot
        return

    # Проверяем длину промпта
    max_len = MAX_IMAGE_PROMPT_LENGTH if isinstance(MAX_IMAGE_PROMPT_LENGTH, int) and MAX_IMAGE_PROMPT_LENGTH > 0 else 1000 # Дефолт 1000
    if len(prompt) > max_len:
        await safe_send_message(message, f"❌ Описание слишком длинное (макс. {max_len} символов).", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot
        return

    # --- Выбор провайдера и модели для изображений из отдельного конфига ---
    image_model_key = storage.get_image_model_key(message.chat.id)

    if not IMAGE_PROVIDERS_CONFIG:
         logger.error("IMAGE_PROVIDERS_CONFIG dictionary is empty. Cannot generate image.")
         await safe_send_message(message, "❌ Ошибка конфигурации: Список моделей для изображений пуст.", reply_to_message=True, parse_mode=None, bot_instance=bot)
         return

    if image_model_key not in IMAGE_PROVIDERS_CONFIG:
        logger.error(f"Invalid image model key '{image_model_key}' found for chat {message.chat.id}. Reverting to default '{DEFAULT_IMAGE_MODEL}'.")
        image_model_key = DEFAULT_IMAGE_MODEL
        if image_model_key not in IMAGE_PROVIDERS_CONFIG:
             logger.error(f"Default image model '{DEFAULT_IMAGE_MODEL}' not found in IMAGE_PROVIDERS_CONFIG. Cannot generate image.")
             await safe_send_message(message, f"⚠️ Ошибка конфигурации: Модель для изображений по умолчанию '{DEFAULT_IMAGE_MODEL}' не найдена.", reply_to_message=True, parse_mode=None, bot_instance=bot)
             return
        storage.set_image_model(message.chat.id, image_model_key) # Сохраняем дефолтную модель для чата

    image_provider_config = IMAGE_PROVIDERS_CONFIG[image_model_key]
    image_provider_name = image_provider_config.get("provider")
    # Предполагаем, что для изображений используется либо 'image_model_name', либо 'model_name'
    image_model_identifier = image_provider_config.get("image_model_name", image_provider_config.get("model_name"))


    if not image_provider_name or not image_model_identifier:
         logger.error(f"Incomplete provider config for image model '{image_model_key}' in chat {message.chat.id}. Missing 'provider' or 'model_name'.")
         await safe_send_message(message, f"⚠️ Ошибка конфигурации для модели изображений '{escape_html(image_model_key)}'.", reply_to_message=True, parse_mode=None, bot_instance=bot)
         return

    if image_provider_name not in PROVIDER_MAP:
         logger.error(f"Provider class '{image_provider_name}' configured for image model '{image_model_key}' not found in PROVIDER_MAP.")
         await safe_send_message(message, f"⚠️ Ошибка конфигурации: провайдер {image_provider_name} для модели изображений не найден.", reply_to_message=True, parse_mode=None, bot_instance=bot)
         return

    image_provider_class = PROVIDER_MAP[image_provider_name]
    # --- Конец выбора провайдера ---


    status_msg = await safe_send_message(message, "🎨 Генерирую изображение...", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot

    try:
        user_id = message.from_user.id if message.from_user else "unknown"
        logger.info(f"Generating image for user {user_id} in chat {message.chat.id} using {image_model_key} ({image_provider_name}/{image_model_identifier})")


        image_generation_successful = False
        image_urls = []

        # Check if the dedicated image creation method exists and is callable
        # AND if the provider is known to support it (optional check, depends on g4f)
        # For simplicity, we'll try the dedicated method first if available
        if hasattr(g4f, 'image') and hasattr(g4f.image, 'create_async') and callable(g4f.image.create_async):
            logger.info(f"Attempting direct image generation via g4f.image.create_async for {image_provider_name}.")
            try:
                response = await g4f.image.create_async(
                    prompt=prompt,
                    provider=image_provider_class,
                    model=image_model_identifier,
                    timeout=180
                )
                image_urls = response if isinstance(response, list) else []
                if image_urls:
                    image_generation_successful = True
                    logger.info(f"Direct image generation successful. Found URL: {image_urls[0]}")

            except NotImplementedError:
                 logger.warning(f"g4f.image.create_async not implemented for {image_provider_name}. Falling back to ChatCompletion.")
            except Exception as e:
                 logger.exception(f"Error during direct g4f.image.create_async call for {image_provider_name}: {e}. Falling back to ChatCompletion.")
        else:
             logger.warning("g4f.image.create_async method is not available or not callable. Skipping direct image generation attempt.")


        # If direct generation failed or is not implemented/available, try ChatCompletion fallback
        if not image_generation_successful:
            logger.info("Attempting image URL extraction from ChatCompletion fallback.")
            try:
                # Просим модель сгенерировать URL изображения
                image_generation_prompt = f"Generate an image based on this description: {prompt}. Provide ONLY the direct image URL in your response."
                chat_response = await g4f.ChatCompletion.create_async(
                    model=image_model_identifier, # Use the selected image model identifier
                    provider=image_provider_class, # Use the selected image provider class
                    messages=[{"role": "user", "content": image_generation_prompt}],
                    timeout=180
                )
                response_str = str(chat_response)
                # Search for URLs in the response (basic pattern)
                image_urls = re.findall(r'https?://[^\s()\'"]+\.(?:png|jpe?g|webp|gif)\b', response_str, re.IGNORECASE)
                if image_urls:
                     logger.info(f"Fallback ChatCompletion found URL: {image_urls[0]}")
                else:
                     logger.warning(f"Fallback ChatCompletion did not return a valid image URL for prompt: {prompt[:50]}. Response: {response_str[:200]}")

            # ИСПРАВЛЕНО: Добавлена обработка NoValidHarFileError для OpenaiChat
            except g4f.errors.NoValidHarFileError as e:
                 logger.error(f"HAR file error for provider {image_provider_name} ({image_model_identifier}): {e}")
                 # Re-raise this specific error to be caught by the outer handler
                 raise e
            except Exception as e:
                 logger.exception(f"Error during ChatCompletion fallback for image prompt: {e}")
                 # If fallback fails, raise the error to be caught by the main try/except
                 raise e # Re-raise the exception to be handled below


        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass

        if image_urls:
            image_url = image_urls[0]
            logger.info(f"Found image URL: {image_url}")
            try:
                # Используем бота из message для отправки фото
                await message.reply_photo(
                    photo=image_url,
                    caption=f"🖼 {escape_html(prompt[:1000])}"
                )
            except TelegramBadRequest as e:
                 logger.error(f"Failed to send photo by URL {image_url}: {e}. URL might be invalid or inaccessible.")
                 await safe_send_message(message, f"⚠️ Не удалось загрузить или отправить изображение по полученной ссылке. Возможно, ссылка некорректна или недоступна.\n({escape_html(str(e))})", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot
            except Exception as e:
                 logger.exception(f"Error sending photo from URL {image_url}: {e}")
                 await safe_send_message(message, f"⚠️ Ошибка при отправке изображения: {escape_html(str(e))}", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot

        else:
            logger.warning(f"No image URL found in response for prompt: {prompt[:50]}. Response: {response_str[:200] if isinstance(response_str, str) else 'Non-string response'}")
            await safe_send_message(message, "❌ Не удалось получить URL изображения от модели. Попробуйте другой запрос или выберите другую модель изображений.", reply_to_message=True, parse_mode=None, bot_instance=bot) # ИСПРАВЛЕНО: Уточнено сообщение

    except asyncio.TimeoutError:
        logger.error(f"Timeout error during image generation for prompt: {prompt[:50]}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, "⚠️ Время ожидания генерации изображения истекло.", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot

    # ИСПРАВЛЕНО: Добавлена обработка NoValidHarFileError здесь
    except g4f.errors.NoValidHarFileError as e:
        logger.error(f"Image generation failed due to missing HAR file for provider {image_provider_name} ({image_model_identifier}): {e}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(
            message,
            f"❌ Не удалось сгенерировать изображение. Провайдер <b>{escape_html(image_model_key)}</b> требует специальной настройки (HAR файл), которая недоступна.\nПожалуйста, выберите другую модель для генерации изображений командой /imagemodel.", # ИСПРАВЛЕНО: Указана команда
            reply_to_message=True,
            parse_mode=ParseMode.HTML,
            bot_instance=bot
        )

    except g4f.errors.ResponseStatusError as e:
        logger.error(f"Provider {image_provider_name} ({image_model_identifier}) returned status error during image generation: {e}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, f"⚠️ Провайдер модели ({escape_html(image_model_key)}) вернул ошибку статуса при генерации изображения: {escape_html(str(e))}\nПопробуйте позже или выберите другую модель изображений командой /imagemodel.", reply_to_message=True, parse_mode=None, bot_instance=bot) # ИСПРАВЛЕНО: Указана команда


    except Exception as e:
        logger.exception(f"Unhandled error during image generation for prompt: {prompt[:50]}")
        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        await safe_send_message(message, f"⚠️ Произошла непредвиденная ошибка при генерации изображения: {escape_html(str(e))}", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot


@router.message(Command("sglypa", "сглыпа", "рандом"))
async def cmd_sglypa(message: types.Message, command: CommandObject):
    """Управляет режимом случайных сообщений в группе."""
    logger.info(f"Handling command sglypa from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    if not is_group_chat(message):
        await safe_send_message(message, "ℹ️ Эта команда доступна только в групповых чатах.", reply_to_message=False, parse_mode=None, bot_instance=message.bot) # Передаем bot
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
                 await safe_send_message(message, "❌ Интервал должен быть положительным числом минут.", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot
                 return

            interval_seconds = minutes * 60

            if interval_seconds < min_interval:
                await safe_send_message(
                    message,
                    f"❌ Минимальный интервал для случайных сообщений: <b>{min_interval // 60}</b> минут. Вы указали: {minutes} мин.",
                    reply_to_message=True,
                    parse_mode=ParseMode.HTML, # Используем HTML для жирного шрифта
                    bot_instance=message.bot # Передаем bot
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
                parse_mode=ParseMode.HTML,
                bot_instance=message.bot # Передаем bot
            )

        except ValueError:
            await safe_send_message(message, "❌ Неверный формат интервала. Укажите целое число минут (например, <code>/sglypa 30</code>).", reply_to_message=True, parse_mode=ParseMode.HTML, bot_instance=message.bot) # Передаем bot
            return
        except Exception as e:
             logger.exception(f"Error processing sglypa command with args in chat {chat_id}: {e}")
             await safe_send_message(message, f"⚠️ Произошла ошибка: {escape_html(str(e))}", reply_to_message=True, parse_mode=None, bot_instance=message.bot) # Передаем bot

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
            parse_mode=ParseMode.HTML,
            bot_instance=message.bot # Передаем bot
        )

# --- Message Handlers ---

# Обработка упоминаний бота в группах
# Важно: Этот хэндлер должен идти ПОСЛЕ обработчиков команд,
# чтобы команды типа /ask @botname обрабатывались как команды, а не упоминания.
@router.message(F.text, F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def handle_mention(message: types.Message, bot: Bot): # Добавляем bot как аргумент
    """Обрабатывает сообщения, начинающиеся с упоминания бота."""
    logger.info(f"Handling potential mention from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
    global BOT_USERNAME
    if not BOT_USERNAME:
        # Попробуем получить имя пользователя здесь, если его нет
        try:
            me = await bot.get_me() # Используем переданный bot
            BOT_USERNAME = me.username or "UnknownBot"
            logger.info(f"Got bot username in handle_mention: @{BOT_USERNAME}")
        except Exception as e:
            logger.warning(f"BOT_USERNAME not set and failed to get it in handle_mention: {e}. Cannot process mentions yet.")
            return # Не можем обработать упоминание

    # Проверяем наличие текста сообщения
    if not message.text:
        return # Не обрабатываем сообщения без текста (например, стикеры с упоминанием)

    # Паттерн для поиска упоминания в начале строки (с @ или без)
    # re.escape экранирует специальные символы в имени пользователя
    # \b - граница слова, чтобы не срабатывало на часть ника
    # re.IGNORECASE - не учитывать регистр
    mention_pattern = re.compile(rf'^(?:@{re.escape(BOT_USERNAME)}\b\s*)+', re.IGNORECASE)
    text_without_mention = mention_pattern.sub('', message.text).lstrip() # Удаляем упоминание и пробелы слева

    # Сработало ли упоминание?
    if len(text_without_mention) < len(message.text): # Если текст стал короче, значит упоминание было
        # Проверяем, что сообщение не является командой (начинается с /)
        # Это предотвратит обработку команд типа "/start @botname" как упоминания
        if text_without_mention.startswith('/'):
             logger.debug(f"Ignoring message starting with command after mention: {message.text[:50]}")
             return

        question = text_without_mention.strip()
        user_id = message.from_user.id if message.from_user else "unknown"
        chat_id = message.chat.id # Получаем chat_id
        is_group = is_group_chat(message) # Проверяем, групповой ли чат

        logger.info(f"Bot mentioned in chat {chat_id} by user {user_id}")

        # ИСПРАВЛЕНО: Добавляем входящее сообщение в short_term контекст
        if message.text: # Убедимся, что есть текст для добавления
             context = storage.get_context(chat_id, user_id, is_group)
             # Добавляем текст сообщения в контекст как сообщение пользователя
             context.add_message("user", message.text, "short")
             logger.info(f"Added incoming mention message from user {user_id} to short-term context in chat {chat_id}.")


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
            await safe_send_message(message, f"❌ Ваш вопрос слишком длинный (макс. {max_len} символов).", reply_to_message=True, parse_mode=None, bot_instance=bot) # Передаем bot
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

# ИСПРАВЛЕНО: Добавлен общий обработчик для текстовых сообщений в группах
# Этот обработчик будет добавлять все текстовые сообщения в short_term контекст,
# если бот имеет право читать сообщения (не в режиме приватности).
# Важно: Этот хэндлер должен идти ПОСЛЕ обработчика упоминаний и команд.
@router.message(F.text, F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def handle_group_text_message(message: types.Message):
    """Обрабатывает любое текстовое сообщение в группе для сохранения контекста."""
    # Проверяем, что сообщение не начинается с команды или упоминания (уже обработано)
    # Это грубая проверка, но достаточная, если хэндлеры зарегистрированы в правильном порядке
    if message.text and not (message.text.startswith('/') or (BOT_USERNAME and message.text.lower().startswith(f'@{BOT_USERNAME.lower()}'))):
        logger.info(f"Handling general group text message from user {message.from_user.id if message.from_user else 'unknown'} in chat {message.chat.id}")
        if not message.from_user: return # Игнорируем сообщения без пользователя

        chat_id = message.chat.id
        user_id = message.from_user.id
        is_group = True # Это групповой ча

        context = storage.get_context(chat_id, user_id, is_group)
        # Добавляем текст сообщения в контекст как сообщение пользователя
        context.add_message("user", message.text, "short")
        logger.info(f"Added general group text message from user {user_id} to short-term context in chat {chat_id}.")


# --- Startup and Shutdown ---

# Используем стандартное имя 'bot' для аргумента, Aiogram передаст его автоматически
async def on_startup(bot: Bot):
    """Выполняется при запуске бота."""
    global BOT_USERNAME # Убираем global bot, так как он передается как аргумент
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
             # Передаем экземпляр бота в фоновую задачу
             logger.info("Scheduling random message task...") # Лог перед созданием задачи
             asyncio.create_task(send_random_messages(bot), name="RandomMessageSender") # Используем 'bot'
             logger.info("Random message task scheduled.")
        else:
             logger.warning("PROVIDERS or PROVIDER_MAP is empty. Random message task NOT scheduled.")

        # asyncio.create_task(save_state_periodically(), name="StateSaver") # Если нужна периодическая сериализация состояния

    except Exception as e:
        logger.critical(f"Failed to get bot info or schedule tasks on startup: {e}", exc_info=True)
        # Возможно, стоит завершить работу, если не удалось получить имя пользователя
        # exit(1)

# Используем стандартное имя 'bot' для аргумента
async def on_shutdown(bot: Bot):
    """Выполняется при остановке бота."""
    logger.info("Bot is shutting down...")
    try:
        # Корректное закрытие сессии бота
        # Используем переданный экземпัส bot
        if bot and bot.session: # Добавляем проверку на None
             await bot.session.close()
             logger.info("Bot session closed.")
        else:
             logger.info("Bot session was not active or already closed.")
    except Exception as e:
         logger.error(f"Error closing bot session: {e}", exc_info=True)
    # Здесь можно добавить сохранение состояния перед выходом, если необходимо
    logger.info("Shutdown process finished.")


if __name__ == "__main__":
    # Убедимся, что глобальный 'bot' был инициализирован перед использованием
    if bot is None:
        logger.critical("Bot instance is None before starting polling. Exiting.")
        exit(1)

    # Регистрируем роутер в диспетчере *перед* регистрацией хуков
    dp.include_router(router)

    # Регистрируем обработчики запуска и остановки
    # Aiogram автоматически передаст экземпляр Bot в функции on_startup/on_shutdown
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
        # Дополнительные действия по очистке, если нужны (хотя on_shutdown должен вызы
