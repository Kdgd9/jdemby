# -*- coding: utf-8 -*-
# COPYRIGHT ZhdomDev
# Optimized version with Vision capabilities

__version__ = "0.12.2" # Версия бота (исправлена логика распознавания и контекста изображений)

import logging
import re
import asyncio
import random
import time
import html
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable, Coroutine
import io

# Aiogram imports
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, CommandObject
from aiogram.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, KeyboardButton, Message, ChatPermissions
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.enums import ChatType, ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter, TelegramBadRequest, TelegramAPIError
from aiogram.client.default import DefaultBotProperties

# g4f imports
import g4f
from g4f.Provider import BaseProvider
import aiohttp
import g4f.errors

# Gemini Vision imports
import google.generativeai as genai
from PIL import Image

# Импортируем BOT_LEGEND напрямую из bot_legend.py
try:
    from bot_legend import BOT_LEGEND
except ImportError:
    logger.critical("Ошибка: Не найден файл bot_legend.py или он не содержит переменную BOT_LEGEND. Бот не сможет использовать легенду.")
    BOT_LEGEND = [] # Устанавливаем пустой список в случае ошибки, чтобы избежать краха


# Configuration import
try:
    from config import (
        PROVIDERS,
        IMAGE_PROVIDERS_CONFIG,
        DEFAULT_MODEL,
        DEFAULT_IMAGE_MODEL,
        MAX_CONTEXT_LENGTH,
        MAX_QUESTION_LENGTH,
        MAX_IMAGE_PROMPT_LENGTH,
        THINKING_MESSAGES,
        RANDOM_MESSAGE_INTERVAL,
        MIN_INTERVAL,
        SAVE_INTERVAL_SECONDS,
        # BOT_LEGEND, # Эта строка удалена, так как BOT_LEGEND импортируется напрямую
        MAX_PROVIDER_RETRIES,
        DEFAULT_REPLY_MODE,
        ERROR_RESPONSES_1,
        ERROR_RESPONSES_FINAL,
        RETRY_DELAY_SECONDS
    )
except ImportError as e:
    print(f"Ошибка импорта из config.py: {e}")
    print("Пожалуйста, убедитесь, что файл config.py существует и содержит необходимые переменные.")
    exit(1)

try:
    from secrets import TELEGRAM_TOKEN, GEMINI_API_KEY
except ImportError:
    print("Ошибка: Не найден файл secrets.py или он не содержит переменные TELEGRAM_TOKEN и GEMINI_API_KEY.")
    print("Пожалуйста, создайте файл secrets.py и определите в нем TELEGRAM_TOKEN и GEMINI_API_KEY.")
    exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Gemini Vision Setup ---
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Google Gemini Vision model initialized successfully.")
    else:
        vision_model = None
        logger.warning("GEMINI_API_KEY is not set. Image recognition feature will be disabled.")
except Exception as e:
    vision_model = None
    logger.error(f"Failed to initialize Gemini Vision model: {e}")

# --- Provider Mapping ---
PROVIDER_MAP: Dict[str, type[BaseProvider]] = {}
all_provider_names = set()
if PROVIDERS:
    all_provider_names.update({prov_data["provider"] for prov_data in PROVIDERS.values()})
if IMAGE_PROVIDERS_CONFIG:
     all_provider_names.update({prov_data["provider"] for prov_data in IMAGE_PROVIDERS_CONFIG.values()})


if all_provider_names:
    try:
        PROVIDER_MAP = {
            name: getattr(g4f.Provider, name)
            for name in all_provider_names
            if hasattr(g4f.Provider, name)
        }
        logger.info(f"Loaded providers: {list(PROVIDER_MAP.keys())}")
        missing_providers = {
            name for name in all_provider_names
            if name not in PROVIDER_MAP
        }
        if missing_providers:
            logger.error(f"Missing provider classes in g4f library: {missing_providers}")
    except AttributeError as e:
        logger.error(f"Error accessing provider attributes in g4f: {e}. Please check g4f installation and provider names in config.py.")
    except Exception as e:
         logger.error(f"An unexpected error occurred during provider mapping: {e}")
else:
    logger.warning("PROVIDERS and IMAGE_PROVIDERS_CONFIG dictionaries in config.py are empty or not defined. No providers loaded.")


# --- Utility Functions ---

def is_group_chat(message: types.Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}

def escape_html(text: str) -> str:
    if not isinstance(text, str):
        logger.warning(f"escape_html received non-string input: {type(text)}. Converting to string.")
        text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text

def convert_markdown_to_html(text: str) -> str:
    if not isinstance(text, str):
        logger.warning(f"convert_markdown_to_html received non-string input: {type(text)}. Returning as is.")
        return str(text)

    def replace_code_block(match):
        lang = match.group(1)
        code = match.group(2)
        escaped_code = html.escape(code)
        if lang:
            return f'<pre><code class="language-{html.escape(lang)}">{escaped_code}</code></pre>'
        else:
            return f'<pre><code>{escaped_code}</code></pre>'
    text = re.sub(r'```(\w*)\s*\n(.*?)\n```', replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r'```(.*?)```', lambda m: f'<pre><code>{html.escape(m.group(1))}</code></pre>', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+?)`', lambda m: f'<code>{html.escape(m.group(1))}</code>', text)
    text = re.sub(r'\*\*(?=\S)(.+?)(?<=\S)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(?=\S)(.+?)(?<=\S)__', r'<i>\1</i>', text)
    text = re.sub(r'~~(?=\S)(.+?)(?<=\S)~~', r'<s>\1</s>', text)
    def replace_link(match):
        link_text = match.group(1)
        url = match.group(2)
        safe_url = html.escape(url).replace('"', '&quot;')
        safe_text = html.escape(link_text)
        return f'<a href="{safe_url}">{safe_text}</a>'
    text = re.sub(r'\[(.*?)\]\((.*?)\)', replace_link, text)
    return text


async def safe_send_message(
    target: types.Message,
    text: str,
    parse_mode: Optional[str] = ParseMode.HTML,
    reply_to_message: bool = True,
    bot_instance: Optional[Bot] = None,
    **kwargs: Any
) -> Optional[types.Message]:
    max_length = 4096
    sent_message = None
    chat_id = target.chat.id
    if not isinstance(text, str):
        logger.warning(f"safe_send_message received non-string: {type(text)}. Converting.")
        text = str(text)
    actual_bot = bot_instance if bot_instance else target.bot
    if not actual_bot:
        logger.error(f"safe_send_message: Cannot determine Bot instance for chat {chat_id}")
        return None
    send_method_callable: Optional[Callable[..., Coroutine[Any, Any, types.Message]]] = None
    send_args = kwargs.copy()
    if reply_to_message and hasattr(target, 'reply'):
        send_method_callable = target.reply
    else:
        send_method_callable = actual_bot.send_message
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
            current_send_args = send_args.copy()
            current_send_args['text'] = chunk
            current_send_args['parse_mode'] = parse_mode
            if send_method_callable == actual_bot.send_message:
                current_send_args['chat_id'] = chat_id
                if not reply_to_message :
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
                    current_send_args['parse_mode'] = None
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

# --- Gemini Vision Function ---
async def get_image_description_gemini(image_bytes: bytes, prompt: str) -> str:
    """
    Отправляет изображение и промпт в Gemini Pro Vision и возвращает ответ.
    """
    if not vision_model:
        return "Ошибка: Модель для распознавания изображений не инициализирована. Проверьте GEMINI_API_KEY."

    try:
        img = Image.open(io.BytesIO(image_bytes))

        if not prompt.strip():
            prompt_for_vision = "Опиши, что изображено на этой картинке, в деталях."
        else:
            prompt_for_vision = prompt

        response = await vision_model.generate_content_async([prompt_for_vision, img], request_options={"timeout": 120})

        return response.text
    except Exception as e:
        logger.error(f"Ошибка при работе с Gemini API: {e}")
        return f"К сожалению, не удалось обработать изображение. Ошибка: {e}"

# --- Gemini Text Generation Function ---
async def get_text_from_gemini_api(model_name: str, history: List[Dict[str, Any]]) -> str:
    """
    Отправляет историю чата в указанную модель Gemini и возвращает текстовый ответ.
    """
    try:
        if not history:
            return "Ошибка: история сообщений пуста."

        system_prompts = []
        chat_history_for_merge = []

        for msg in history:
            if msg['role'] == 'system':
                system_prompts.append(msg['content'])
            else:
                chat_history_for_merge.append(msg)

        merged_chat_parts = []
        if chat_history_for_merge:
            current_role = chat_history_for_merge[0]['role']
            current_content = chat_history_for_merge[0]['content']

            for i in range(1, len(chat_history_for_merge)):
                if chat_history_for_merge[i]['role'] == current_role:
                    current_content += "\n\n" + chat_history_for_merge[i]['content']
                else:
                    # Gemini API использует роль 'model' вместо 'assistant'
                    role_to_add = 'model' if current_role == 'assistant' else current_role
                    merged_chat_parts.append({'role': role_to_add, 'parts': [{'text': current_content}]})
                    current_role = chat_history_for_merge[i]['role']
                    current_content = chat_history_for_merge[i]['content']
                
            # Добавляем последнее объединенное сообщение
            role_to_add = 'model' if current_role == 'assistant' else current_role
            merged_chat_parts.append({'role': role_to_add, 'parts': [{'text': current_content}]})

        # Если первый элемент в истории не 'user', это может вызвать ошибку.
        # Gemini API чат должен начинаться с 'user'.
        # Удаляем первый элемент, если он 'model'.
        if merged_chat_parts and merged_chat_parts[0]['role'] == 'model':
            logger.warning("History starts with 'model' role, removing it for Gemini compatibility.")
            merged_chat_parts.pop(0)

        # Отделяем последний промпт пользователя
        last_user_prompt_content = ""
        if merged_chat_parts and merged_chat_parts[-1]['role'] == 'user':
            last_user_message = merged_chat_parts.pop(-1)
            # Content in last_user_message['parts'][0]['text']
            last_user_prompt_content = last_user_message['parts'][0]['text']
        elif history: # Fallback for cases where merged_chat_parts might be empty or last is not user
             # This handles initial user messages that might not go through the merge logic if it's a single message
             # Or if after merging, the last message wasn't from user.
             # We take the content of the very last message in the original history as the user's prompt.
            original_last_message = history[-1]
            if original_last_message['role'] == 'user':
                last_user_prompt_content = original_last_message['content']
            else:
                logger.warning("Last message in history is not from 'user'. Cannot determine last user prompt correctly.")
                return "Ошибка: Не удалось определить последний запрос пользователя."


        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction="\n".join(system_prompts)
        )
        chat = model.start_chat(history=merged_chat_parts)
        
        response = await chat.send_message_async(last_user_prompt_content, request_options={"timeout": 120})
        return response.text

    except Exception as e:
        logger.error(f"Ошибка при работе с Gemini API (Text): {e}")
        raise e

# --- New function for Imagen API integration ---
async def generate_image_with_imagen(model_name: str, prompt: str) -> List[str]:
    """
    Генерирует изображение с использованием Official Google Imagen API.
    Возвращает список URL изображений (data URLs).
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY не установлен для Imagen API.")
        return []

    try:
        async with aiohttp.ClientSession() as session:
            # Используем API-ключ для аутентификации
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict?key={GEMINI_API_KEY}"
            
            # --- ИСПРАВЛЕНО: Корректная структура payload для Imagen API ---
            payload = {
                "instances": [
                    {
                        "prompt": {
                            "text": prompt
                        }
                    }
                ],
                "parameters": {
                    "sampleCount": 1
                }
            }
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            headers = {'Content-Type': 'application/json'}

            logger.info(f"Выполняю прямой вызов Imagen API к {api_url} с промптом: {prompt[:50]}...")
            async with session.post(api_url, json=payload, headers=headers, timeout=120) as response:
                response.raise_for_status() # Вызывает исключение для HTTP ошибок (4xx, 5xx)
                result = await response.json()

                if result.get("predictions") and len(result["predictions"]) > 0 and result["predictions"][0].get("bytesBase64Encoded"):
                    image_base64 = result["predictions"][0]["bytesBase64Encoded"]
                    image_url = f"data:image/png;base64,{image_base64}"
                    logger.info("Вызов Imagen API успешен. Изображение сгенерировано.")
                    return [image_url]
                else:
                    logger.error(f"Ответ Imagen API не содержит predictions или bytesBase64Encoded: {result}")
                    return []

    except aiohttp.ClientError as e:
        logger.error(f"Ошибка сети или HTTP при вызове Imagen API: {e}")
        raise ValueError(f"Ошибка сети или API Imagen: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Не удалось разобрать JSON ответ Imagen API: {e}")
        raise ValueError(f"Ошибка парсинга ответа Imagen API: {e}")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при генерации изображения с Imagen: {e}")
        raise ValueError(f"Непредвиденная ошибка генерации Imagen: {e}")


# --- Context Management ---
class ChatContext:
    def __init__(self, key: int, max_short_term: int):
        self.key = key
        # Проверяем, что BOT_LEGEND является списком строк, иначе используем пустой список
        if isinstance(BOT_LEGEND, list) and all(isinstance(item, str) for item in BOT_LEGEND):
             self.legend = [{"role": "system", "content": legend_item} for legend_item in BOT_LEGEND]
        else:
             logger.error("BOT_LEGEND не является списком строк. Используется пустая легенда.")
             self.legend = []
        self.short_term: List[Dict[str, str]] = []
        self.long_term: List[Dict[str, str]] = []
        self._max_short_term = 0
        self.set_max_short_term(max_short_term)
        self.load_long_term()

    def set_max_short_term(self, new_limit: int):
        validated_limit = max(0, int(new_limit)) if isinstance(new_limit, (int, float, str)) and str(new_limit).isdigit() else (_DEFAULT_SHORT_TERM_MESSAGE_LIMIT)
        if validated_limit == 0:
            logger.warning(f"Лимит краткосрочной памяти для ключа {self.key} установлен на 0. Краткосрочная память будет отключена.")
        self._max_short_term = validated_limit
        if len(self.short_term) > self._max_short_term:
            self.short_term = self.short_term[-self._max_short_term:]
            logger.info(f"Краткосрочная память для ключа {self.key} обрезана до {self._max_short_term} сообщений.")
        logger.debug(f"Лимит краткосрочной памяти для ключа {self.key} установлен на {self._max_short_term} сообщений.")

    @property
    def max_short_term(self) -> int:
        return self._max_short_term

    def add_message(self, role: str, content: str, memory_type: str = "short"):
        if not isinstance(content, str):
             logger.warning(f"Попытка добавить нестроковое содержимое в контекст: {type(content)}. Преобразование.")
             content = str(content)
        if not content:
             logger.warning(f"Попытка добавить пустое сообщение в память {memory_type} для роли {role}.")
             return
        message = {"role": role, "content": content}
        if memory_type == "short":
            if self.max_short_term > 0:
                self.short_term.append(message)
                if len(self.short_term) > self.max_short_term:
                    self.short_term = self.short_term[-self.max_short_term:]
        elif memory_type == "long":
            self.long_term.append(message)

    def get_full_context(self) -> List[Dict[str, str]]:
        return self.legend + self.long_term + self.short_term

    def get_long_context(self) -> List[Dict[str, str]]:
        return self.legend + self.long_term

    def clear_short(self):
        self.short_term = []
        logger.info(f"Краткосрочный контекст очищен для ключа {self.key}.")

    def clear_all(self):
        self.short_term = []
        self.long_term = []
        logger.info(f"Полный контекст очищен для ключа {self.key} (краткосрочная и долгосрочная память).")
        filename = f"long_term_memory_{self.key}.json"
        if os.path.exists(filename):
            try:
                os.remove(filename)
                logger.info(f"Удален файл долгосрочной памяти: {filename}")
            except OSError as e:
                logger.error(f"Ошибка при удалении файла долгосрочной памяти {filename}: {e}")

    def save_long_term(self):
        if not self.long_term:
            filename = f"long_term_memory_{self.key}.json"
            if os.path.exists(filename):
                 try:
                    os.remove(filename)
                    logger.info(f"Удален пустой файл долгосрочной памяти: {filename}")
                 except OSError as e:
                    logger.error(f"Ошибка при удалении пустого файла долгосрочной памяти {filename}: {e}")
            return
        filename = f"long_term_memory_{self.key}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.long_term, f, ensure_ascii=False, indent=4)
        except IOError as e:
            logger.error(f"Ошибка при сохранении долгосрочной памяти для ключа {self.key} в {filename}: {e}")
        except Exception as e:
            logger.exception(f"Непредвиденная ошибка при сохранении долгосрочной памяти для ключа {self.key}: {e}")

    def load_long_term(self):
        filename = f"long_term_memory_{self.key}.json"
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        self.long_term = loaded_data
                        logger.info(f"Долгосрочная память загружена для ключа {self.key} из {filename}. Элементов: {len(self.long_term)}")
                    else:
                        logger.warning(f"Загруженные данные из {filename} не являются списком. Игнорируется.")
                        self.long_term = []
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                logger.error(f"Ошибка при загрузке долгосрочной памяти для ключа {self.key} из {filename}: {e}")
                self.long_term = []
            except Exception as e:
                logger.exception(f"Непредвиденная ошибка при загрузке долгосрочной памяти для ключа {self.key}: {e}")
                self.long_term = []
        else:
            self.long_term = []

# --- Global Storage ---
_DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED = False
_DEFAULT_SHORT_TERM_MESSAGE_LIMIT = MAX_CONTEXT_LENGTH * 2
_DEFAULT_CONTEXT_TYPES_ENABLED = {'legend': True, 'long': True, 'short': True}

class Storage:
    def __init__(self):
        self.contexts: Dict[int, ChatContext] = {}
        self.models: Dict[int, str] = {}
        self.image_models: Dict[int, str] = {}
        self.reply_modes: Dict[int, str] = {}
        self.context_types_enabled: Dict[int, Dict[str, bool]] = {}
        self.read_general_chat_messages_enabled: Dict[int, bool] = {}
        self.short_term_limits: Dict[int, int] = {}
        self.active_chats: Dict[int, Dict[str, Any]] = {}
        self.developer_modes: Dict[int, bool] = {}

    def _get_key(self, chat_id: int, user_id: int, is_group: bool) -> int:
        return chat_id if is_group else user_id

    def _get_settings_filepath(self, key: int) -> str:
        return f"chat_settings_{key}.json"

    def _load_settings_for_key(self, key: int):
        filepath = self._get_settings_filepath(key)
        if key not in self.models: self.models[key] = DEFAULT_MODEL
        if key not in self.image_models: self.image_models[key] = DEFAULT_IMAGE_MODEL
        if key not in self.reply_modes: self.reply_modes[key] = DEFAULT_REPLY_MODE
        if key not in self.context_types_enabled: self.context_types_enabled[key] = _DEFAULT_CONTEXT_TYPES_ENABLED.copy()
        if key not in self.read_general_chat_messages_enabled: self.read_general_chat_messages_enabled[key] = _DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED
        if key not in self.short_term_limits: self.short_term_limits[key] = _DEFAULT_SHORT_TERM_MESSAGE_LIMIT
        if key not in self.developer_modes: self.developer_modes[key] = False
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    settings_data = json.load(f)
                    self.models[key] = settings_data.get('model', DEFAULT_MODEL)
                    self.image_models[key] = settings_data.get('image_model', DEFAULT_IMAGE_MODEL)
                    self.reply_modes[key] = settings_data.get('reply_mode', DEFAULT_REPLY_MODE)
                    self.developer_modes[key] = settings_data.get('developer_mode', False)
                    loaded_context_types = settings_data.get('context_types_enabled', {})
                    if isinstance(loaded_context_types, dict):
                        self.context_types_enabled[key] = {**_DEFAULT_CONTEXT_TYPES_ENABLED, **loaded_context_types}
                        if 'ultra_short' in self.context_types_enabled[key]: del self.context_types_enabled[key]['ultra_short']
                        if 'group_text_to_short_term' in loaded_context_types:
                             self.read_general_chat_messages_enabled[key] = loaded_context_types['group_text_to_short_term']
                             del self.context_types_enabled[key]['group_text_to_short_term']
                    else:
                        self.context_types_enabled[key] = _DEFAULT_CONTEXT_TYPES_ENABLED.copy()
                    self.read_general_chat_messages_enabled[key] = settings_data.get('read_general_chat_messages_enabled', _DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED)
                    self.short_term_limits[key] = settings_data.get('short_term_limit', _DEFAULT_SHORT_TERM_MESSAGE_LIMIT)
                    logger.info(f"Настройки загружены для ключа {key} из {filepath}")
            except (json.JSONDecodeError, IOError, Exception) as e:
                logger.error(f"Ошибка при загрузке настроек для ключа {key} из {filepath}: {e}. Используются текущие значения по умолчанию.")
        else:
            logger.info(f"Файл настроек не найден для ключа {key}. Используются настройки по умолчанию.")

    def _save_settings_for_key(self, key: int):
        filepath = self._get_settings_filepath(key)
        context_types_to_save = self.context_types_enabled.get(key, _DEFAULT_CONTEXT_TYPES_ENABLED.copy())
        if 'ultra_short' in context_types_to_save:
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
            logger.info(f"Настройки сохранены для ключа {key} в {filepath}")
        except (IOError, Exception) as e:
            logger.error(f"Ошибка при сохранении настроек для ключа {key} в {filepath}: {e}")

    def get_context(self, chat_id: int, user_id: int, is_group: bool) -> ChatContext:
        key = self._get_key(chat_id, user_id, is_group)
        self._load_settings_for_key(key)
        current_short_term_limit = self.short_term_limits.get(key, _DEFAULT_SHORT_TERM_MESSAGE_LIMIT)
        if key not in self.contexts:
            self.contexts[key] = ChatContext(key=key, max_short_term=current_short_term_limit)
            logger.info(f"Создан новый контекст для ключа {key} с лимитом краткосрочной памяти {current_short_term_limit}.")
        else:
            if self.contexts[key].max_short_term != current_short_term_limit:
                self.contexts[key].set_max_short_term(current_short_term_limit)
                logger.info(f"Лимит краткосрочной памяти контекста для ключа {key} обновлен до {current_short_term_limit}.")
        return self.contexts[key]

    def clear_short(self, chat_id: int, user_id: int, is_group: bool):
        key = storage._get_key(chat_id, user_id, is_group)
        if key in storage.contexts:
            storage.contexts[key].clear_short()
        else:
             logger.warning(f"Попытка очистить краткосрочный контекст для несуществующего ключа {key}")

    async def clear_all(self, chat_id: int, user_id: int, is_group: bool, bot_instance: Bot):
        key = storage._get_key(chat_id, user_id, is_group)
        if key in storage.contexts:
            storage.contexts[key].clear_all()
            del storage.contexts[key]
            filepath = self._get_settings_filepath(key)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Удален файл настроек: {filepath}")
                except OSError as e:
                    logger.error(f"Ошибка при удалении файла настроек {filepath}: {e}")
            self.models.pop(key, None)
            self.image_models.pop(key, None)
            self.reply_modes.pop(key, None)
            self.context_types_enabled.pop(key, None)
            self.read_general_chat_messages_enabled.pop(key, None)
            self.short_term_limits.pop(key, None)
            self.developer_modes.pop(key, None)
            await safe_send_message(target=types.Message(chat=types.Chat(id=chat_id, type=ChatType.PRIVATE), from_user=types.User(id=user_id, is_bot=False)),
                                    text="♻️ Вся память (краткосрочная и долгосрочная) и настройки очищены!", 
                                    reply_to_message=False, parse_mode=None, bot_instance=bot_instance)
        else:
             await safe_send_message(target=types.Message(chat=types.Chat(id=chat_id, type=ChatType.PRIVATE), from_user=types.User(id=user_id, is_bot=False)),
                                    text="ℹ️ Контекст для этого чата не найден.", 
                                    reply_to_message=False, parse_mode=None, bot_instance=bot_instance)

    def get_available_models(self, for_dev_mode: bool = False) -> List[str]:
        if not PROVIDERS: return []
        return [k for k, v in PROVIDERS.items() if not v.get("developer_only", False) or for_dev_mode]

    def get_available_image_models(self, for_dev_mode: bool = False) -> List[str]:
        if not IMAGE_PROVIDERS_CONFIG: return []
        return [k for k, v in IMAGE_PROVIDERS_CONFIG.items() if not v.get("developer_only", False) or for_dev_mode]

    def get_model_key(self, chat_id: int) -> str:
        if chat_id not in self.models: self._load_settings_for_key(chat_id)
        return self.models.get(chat_id, DEFAULT_MODEL)

    def set_model(self, chat_id: int, model_key: str):
        is_dev_mode = self.get_developer_mode(chat_id)
        if model_key in self.get_available_models(for_dev_mode=is_dev_mode):
            self.models[chat_id] = model_key
            self._save_settings_for_key(chat_id)
            logger.info(f"Текстовая модель для чата {chat_id} установлена на '{model_key}'")
        else:
            logger.warning(f"Попытка установить недействительную текстовую модель '{model_key}' для чата {chat_id}.")

    def get_image_model_key(self, chat_id: int) -> str:
        if chat_id not in self.image_models: self._load_settings_for_key(chat_id)
        return self.image_models.get(chat_id, DEFAULT_IMAGE_MODEL)

    def set_image_model(self, chat_id: int, model_key: str):
        is_dev_mode = self.get_developer_mode(chat_id)
        if model_key in self.get_available_image_models(for_dev_mode=is_dev_mode):
            self.image_models[chat_id] = model_key
            self._save_settings_for_key(chat_id)
            logger.info(f"Модель изображения для чата {chat_id} установлена на '{model_key}'")
        else:
            logger.warning(f"Попытка установить недействительную модель изображения '{model_key}' для чата {chat_id}.")

    def get_reply_mode(self, chat_id: int) -> str:
        if chat_id not in self.reply_modes: self._load_settings_for_key(chat_id)
        return self.reply_modes.get(chat_id, DEFAULT_REPLY_MODE)

    def set_reply_mode(self, chat_id: int, mode: str):
        if mode in ["original", "direct"]:
            self.reply_modes[chat_id] = mode
            self._save_settings_for_key(chat_id)
            logger.info(f"Режим ответа для чата {chat_id} установлен на '{mode}'")
        else:
            logger.warning(f"Попытка установить недействительный режим ответа '{mode}' для чата {chat_id}.")

    def get_chat_context_settings(self, chat_id: int) -> Dict[str, bool]:
        if chat_id not in self.context_types_enabled: self._load_settings_for_key(chat_id)
        return self.context_types_enabled.get(chat_id, _DEFAULT_CONTEXT_TYPES_ENABLED).copy()

    def set_chat_context_setting(self, chat_id: int, setting_name: str, enabled: bool):
        if chat_id not in self.context_types_enabled: self._load_settings_for_key(chat_id)
        if setting_name in self.context_types_enabled[chat_id]:
            self.context_types_enabled[chat_id][setting_name] = enabled
            self._save_settings_for_key(chat_id)
            logger.info(f"Настройка контекста '{setting_name}' для чата {chat_id} установлена на {enabled}")
        else:
            logger.warning(f"Попытка установить недействительную настройку контекста '{setting_name}' для чата {chat_id}.")

    def get_read_general_chat_messages_enabled(self, chat_id: int) -> bool:
        if chat_id not in self.read_general_chat_messages_enabled: self._load_settings_for_key(chat_id)
        return self.read_general_chat_messages_enabled.get(chat_id, _DEFAULT_READ_GENERAL_CHAT_MESSAGES_ENABLED)

    def set_read_general_chat_messages_enabled(self, chat_id: int, enabled: bool):
        self.read_general_chat_messages_enabled[chat_id] = enabled
        self._save_settings_for_key(chat_id)
        logger.info(f"Чтение общих сообщений чата для чата {chat_id} установлено на {enabled}")

    def get_short_term_limit(self, chat_id: int) -> int:
        if chat_id not in self.short_term_limits: self._load_settings_for_key(chat_id)
        return self.short_term_limits.get(chat_id, _DEFAULT_SHORT_TERM_MESSAGE_LIMIT)

    def set_short_term_limit(self, chat_id: int, limit: int):
        validated_limit = max(0, int(limit)) if isinstance(limit, (int, float, str)) and str(limit).isdigit() else _DEFAULT_SHORT_TERM_MESSAGE_LIMIT
        self.short_term_limits[chat_id] = validated_limit
        self._save_settings_for_key(chat_id)
        logger.info(f"Лимит краткосрочных сообщений для чата {chat_id} установлен на {validated_limit}")
        group_key = self._get_key(chat_id, 0, True)
        if group_key in self.contexts:
            self.contexts[group_key].set_max_short_term(validated_limit)

    def get_developer_mode(self, chat_id: int) -> bool:
        if chat_id not in self.developer_modes: self._load_settings_for_key(chat_id)
        return self.developer_modes.get(chat_id, False)

    def set_developer_mode(self, chat_id: int, enabled: bool):
        self.developer_modes[chat_id] = enabled
        self._save_settings_for_key(chat_id)
        logger.info(f"Режим разработчика для чата {chat_id} установлен на {enabled}")

    def init_chat_settings(self, chat_id: int) -> Dict[str, Any]:
        if chat_id not in self.active_chats:
            interval = RANDOM_MESSAGE_INTERVAL if isinstance(RANDOM_MESSAGE_INTERVAL, (int, float)) and RANDOM_MESSAGE_INTERVAL > 0 else 3600
            self.active_chats[chat_id] = {'active': False, 'interval': interval, 'last_sent': 0.0}
            logger.info(f"Инициализированы настройки случайных сообщений для чата {chat_id} с интервалом {interval}с")
        return self.active_chats[chat_id]

# --- Bot Initialization ---
default_properties = DefaultBotProperties(parse_mode=ParseMode.HTML)
bot: Optional[Bot] = None
try:
    if not TELEGRAM_TOKEN: raise ValueError("TELEGRAM_TOKEN не определен")
    bot_instance = Bot(token=TELEGRAM_TOKEN, default=default_properties)
    bot = bot_instance
except (ValueError, Exception) as e:
    logger.critical(f"Не удалось инициализировать бота: {e}")
    exit(1)

dp = Dispatcher()
storage = Storage()
router = Router()
BOT_USERNAME: Optional[str] = None

# --- FSM ---
class Form(StatesGroup):
    model_selection = State()
    image_model_selection = State()
    context_settings_menu = State()
    set_short_term_limit = State()

# --- User Activity Logging ---
USER_ACTIVITY_LOG_FILE = "user_activity.log"
async def log_user_activity(chat_id: int, chat_title: str, user_name: str, user_id: int, chat_type: str):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_entry = f"{timestamp} | Chat ID: {chat_id} | Chat Type: {chat_type} | Chat Title: {chat_title} | User ID: {user_id} | User Name: {user_name}\n"
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: append_to_file(USER_ACTIVITY_LOG_FILE, log_entry))
    except Exception as e:
        logger.error(f"Ошибка записи в файл журнала активности пользователя {USER_ACTIVITY_LOG_FILE}: {e}")

def append_to_file(filename, data):
    try:
        with open(filename, 'a', encoding='utf-8') as f: f.write(data)
    except (IOError, Exception) as e:
        logger.error(f"Ошибка добавления в файл {filename}: {e}")

# --- Background Tasks ---
async def send_random_messages(bot: Bot):
    await asyncio.sleep(15)
    logger.info("Запуск задачи случайных сообщений...")
    while True:
        await asyncio.sleep(30)
        current_time = time.time()
        active_chat_ids = list(storage.active_chats.keys())
        for chat_id in active_chat_ids:
            if chat_id not in storage.active_chats: continue
            settings = storage.active_chats[chat_id]
            if not settings.get('active'): continue
            interval = settings.get('interval', RANDOM_MESSAGE_INTERVAL)
            if not isinstance(interval, (int, float)) or interval <= 0:
                interval = RANDOM_MESSAGE_INTERVAL
                settings['interval'] = interval
            last_sent = settings.get('last_sent', 0.0)
            if current_time - last_sent >= interval:
                logger.info(f"Попытка отправить случайное сообщение в чат {chat_id}")
                try:
                    context = storage.get_context(chat_id=chat_id, user_id=0, is_group=True)
                    initial_model_key = storage.get_model_key(chat_id)
                    available_text_models = storage.get_available_models(for_dev_mode=False)
                    messages_for_random_llm = context.legend + context.long_term + context.short_term
                    valid_messages_for_random = [m for m in messages_for_random_llm if isinstance(m, dict) and 'role' in m and 'content' in m]
                    random_prompt_messages = valid_messages_for_random + [{"role": "user", "content": "Ты участник этого группового чата. Используя контекст, придумай СЛУЧАЙНОЕ, КОРОТКОЕ, НЕЙТРАЛЬНОЕ или ЗАБАВНОЕ сообщение для поддержания беседы. Твой ответ должен быть ТОЛЬКО текстом самого сообщения. НЕ представляйся, НЕ упоминай, что ты бот, НЕ используй префиксы типа 'Сообщение:', НЕ ссылайся на этот промпт. Просто напиши сообщение так, как будто ты обычный участник чата."}]
                    response_text = ""
                    try:
                        response_text = await attempt_llm_request(messages_to_send=random_prompt_messages, initial_model_key=initial_model_key, model_config_dict=PROVIDERS, available_model_keys=available_text_models, g4f_function=g4f.ChatCompletion.create_async, is_image_generation=False, max_attempts=MAX_PROVIDER_RETRIES, chat_id_for_log=chat_id, user_id_for_log=0, status_message_to_update=None, bot_for_status_update=bot, is_developer_mode=False)
                    except Exception as e:
                        logger.error(f"Чат {chat_id}: Не удалось получить случайное сообщение от LLM после всех попыток: {e}")
                        settings['last_sent'] = current_time + 60
                        continue
                    if response_text:
                        temp_message_stub = types.Message(message_id=0, date=int(time.time()), chat=types.Chat(id=chat_id, type=ChatType.GROUP), from_user=None)
                        sent_msg = await safe_send_message(target=temp_message_stub, text=response_text, parse_mode=ParseMode.HTML, reply_to_message=False, bot_instance=bot)
                        if sent_msg:
                            settings['last_sent'] = current_time
                            context.add_message("assistant", response_text, "short")
                        else:
                            logger.error(f"Чат {chat_id}: Не удалось отправить случайное сообщение.")
                    else:
                        logger.warning(f"Чат {chat_id}: Получен пустой ответ случайного сообщения.")
                except (TelegramForbiddenError, TelegramRetryAfter, Exception) as e:
                    logger.exception(f"Чат {chat_id}: Необработанная ошибка при обработке случайного сообщения: {e}")
                    if isinstance(e, TelegramForbiddenError):
                        if chat_id in storage.active_chats: del storage.active_chats[chat_id]
                    elif isinstance(e, TelegramRetryAfter):
                        await asyncio.sleep(e.retry_after)
                    else:
                        settings['last_sent'] = current_time + 60
                        await asyncio.sleep(1)

async def save_state_periodically():
    save_interval = SAVE_INTERVAL_SECONDS if isinstance(SAVE_INTERVAL_SECONDS, (int, float)) and SAVE_INTERVAL_SECONDS > 0 else 300
    logger.info(f"Запуск задачи периодического сохранения состояния с интервалом {save_interval} секунд.")
    while True:
        await asyncio.sleep(save_interval)
        logger.info("Инициирование периодического сохранения состояния...")
        try:
            for key, context_obj in storage.contexts.items():
                context_obj.save_long_term()
            logger.info("Периодическое сохранение состояния завершено.")
        except Exception as e:
            logger.error(f"Ошибка во время периодического сохранения состояния: {e}", exc_info=True)

# --- Core Logic for LLM interaction ---
async def attempt_llm_request(messages_to_send: List[Dict[str, str]], initial_model_key: str, model_config_dict: Dict[str, Dict[str, Any]], available_model_keys: List[str], g4f_function: Callable, is_image_generation: bool = False, image_generation_prompt: Optional[str] = None, max_attempts: int = 3, chat_id_for_log: int = 0, user_id_for_log: Optional[int] = 0, status_message_to_update: Optional[types.Message] = None, bot_for_status_update: Optional[Bot] = None, is_developer_mode: bool = False) -> Any:
    # Проверяем, является ли выбранная модель официальной моделью Google
    initial_model_config = model_config_dict.get(initial_model_key)
    if initial_model_config and initial_model_config.get("provider") == "OfficialGoogle":
        logger.info(f"[Chat {chat_id_for_log}] [User {user_id_for_log}] Используется официальный Google API с моделью '{initial_model_key}'")
        model_name = initial_model_config.get("model_name")
        try:
            if is_image_generation:
                # Вызываем новую функцию для генерации изображений через Imagen API
                image_urls = await generate_image_with_imagen(model_name, image_generation_prompt)
                return image_urls
            else:
                # Вызываем существующую функцию для генерации текста через Gemini API
                response = await get_text_from_gemini_api(model_name, messages_to_send)
                return response
        except Exception as e:
            logger.error(f"[Chat {chat_id_for_log}] Вызов официального Google API завершился ошибкой для модели '{initial_model_key}': {e}")
            # Передаем ошибку дальше, чтобы process_question мог ее показать
            raise e

    # Если это не модель от Google, продолжаем использовать существующую логику g4f
    if not available_model_keys: raise ValueError("No available models configured.")
    effective_max_attempts = 1 if is_developer_mode else max_attempts
    ordered_keys_to_try = [initial_model_key] + [m for m in available_model_keys if m != initial_model_key]
    if initial_model_key not in ordered_keys_to_try and available_model_keys: ordered_keys_to_try = available_model_keys[:]
    if not ordered_keys_to_try: raise ValueError("Could not determine order of models to try.")
    last_exception_seen: Optional[Exception] = None
    actual_attempts_done = 0
    current_model_key_index = 0
    while actual_attempts_done < effective_max_attempts and current_model_key_index < len(ordered_keys_to_try):
        current_model_key = ordered_keys_to_try[current_model_key_index]
        # Пропускаем модели Google в цикле g4f
        current_provider_config = model_config_dict.get(current_model_key)
        if not current_provider_config or current_provider_config.get("provider") == "OfficialGoogle":
            current_model_key_index += 1
            continue

        actual_attempts_done += 1
        current_provider_config = model_config_dict.get(current_model_key)
        if not current_provider_config:
            last_exception_seen = KeyError(f"Ключ модели '{current_model_key}' не найден.")
            current_model_key_index +=1
            continue
        provider_name = current_provider_key = current_provider_config.get("provider")
        model_id_key = "image_model_name" if is_image_generation else "model_name"
        model_identifier = current_provider_config.get(model_id_key, current_provider_config.get("model_name"))
        if not provider_name or not model_identifier:
            last_exception_seen = ValueError(f"Неполная конфигурация для '{current_model_key}'.")
            current_model_key_index +=1
            continue
        if provider_name not in PROVIDER_MAP:
            last_exception_seen = KeyError(f"Класс провайдера '{provider_name}' не найден.")
            current_model_key_index +=1
            continue
        provider_class_to_use = PROVIDER_MAP[provider_name]
        logger.info(f"[Chat {chat_id_for_log}] [User {user_id_for_log}] Попытка {actual_attempts_done}/{effective_max_attempts}: Используется модель '{current_model_key}'")
        if actual_attempts_done > 1 and status_message_to_update and bot_for_status_update and not is_developer_mode:
            try:
                await bot_for_status_update.edit_message_text(text=f"{random.choice(THINKING_MESSAGES)} (Попытка {actual_attempts_done} с {escape_html(current_model_key)}...)", chat_id=status_message_to_update.chat.id, message_id=status_message_to_update.message_id, parse_mode=ParseMode.HTML)
            except Exception as e_edit:
                logger.warning(f"Не удалось отредактировать сообщение статуса: {e_edit}")
        try:
            response: Any
            if is_image_generation:
                response = await g4f_function(prompt_text=image_generation_prompt, provider_class=provider_class_to_use, model_id=model_identifier, timeout_val=180)
                if isinstance(response, list): return response
                raise ValueError(f"Генерация изображения с {current_model_key} вернула неожиданный тип.")
            else:
                response = await g4f_function(model=model_identifier, messages=messages_to_send, provider=provider_class_to_use, timeout=120)
                response_content = str(response).strip()
                if not response_content: raise ValueError(f"Пустой ответ от {current_model_key}")
                return response_content
        except (asyncio.TimeoutError, aiohttp.client_exceptions.ClientError, g4f.errors.RateLimitError, ValueError, TypeError) as e:
            last_exception_seen = e
            logger.warning(f"[Chat {chat_id_for_log}] Сбой с '{current_model_key}': {type(e).__name__} - {e}")
            if is_developer_mode: raise e
            current_model_key_index += 1
            if actual_attempts_done < effective_max_attempts: await asyncio.sleep(1)
        except (g4f.errors.MissingAuthError, g4f.errors.MissingRequirementsError, g4f.errors.NoValidHarFileError) as e:
            last_exception_seen = e
            logger.error(f"[Chat {chat_id_for_log}] Ошибка настройки провайдера с '{current_model_key}': {e}.")
            raise e
        except Exception as e:
            last_exception_seen = e
            logger.exception(f"[Chat {chat_id_for_log}] Непредвиденная ошибка с '{current_model_key}': {e}")
            if is_developer_mode: raise e
            current_model_key_index += 1
            if actual_attempts_done < effective_max_attempts: await asyncio.sleep(1)
    if last_exception_seen:
        failed_model_key_index = current_model_key_index -1 if current_model_key_index > 0 else 0
        failed_model_key = ordered_keys_to_try[failed_model_key_index] if failed_model_key_index < len(ordered_keys_to_try) else "unknown"
        logger.error(f"[Chat {chat_id_for_log}] Все {actual_attempts_done} попыток провалились. Последняя ошибка с '{failed_model_key}': {last_exception_seen}")
        raise last_exception_seen
    else:
        msg = "Все попытки провайдера завершились без конкретной операционной ошибки."
        logger.error(f"[Chat {chat_id_for_log}] {msg}")
        raise Exception(msg)

async def process_question(message: types.Message, question: str, memory_type: str = "short", command_used: Optional[str] = None):
    chat_id = message.chat.id
    if not message.from_user: return
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    is_developer_mode = storage.get_developer_mode(chat_id)
    chat_title = message.chat.title if is_group else "Private Chat"
    user_name = message.from_user.full_name or message.from_user.first_name
    await log_user_activity(chat_id, chat_title, user_name, user_id, message.chat.type)
    context = storage.get_context(chat_id, user_id, is_group) 
    initial_model_key = storage.get_model_key(chat_id)
    if not PROVIDERS:
        await safe_send_message(message, "⚠️ Ошибка: Список моделей пуст.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return
    status_msg = await safe_send_message(message, random.choice(THINKING_MESSAGES), reply_to_message=True, parse_mode=None, bot_instance=message.bot)
    full_response = ""
    max_external_attempts = 1 if is_developer_mode else 2
    attempt_count = 0
    while attempt_count < max_external_attempts:
        attempt_count += 1
        logger.info(f"Обработка вопроса от пользователя {user_id}. Внешняя попытка {attempt_count}/{max_external_attempts}. Режим разработчика: {is_developer_mode}")
        try:
            llm_messages_list: List[Dict[str, str]] = []
            chat_context_settings = storage.get_chat_context_settings(chat_id)
            if chat_context_settings.get('legend', True): llm_messages_list.extend(context.legend)
            if chat_context_settings.get('long', True): llm_messages_list.extend(context.long_term)
            if chat_context_settings.get('short', True): llm_messages_list.extend(context.short_term)
            llm_messages_list.append({"role": "user", "content": question})
            valid_llm_messages = [m for m in llm_messages_list if isinstance(m, dict) and 'role' in m and 'content' in m]
            if not valid_llm_messages: raise ValueError("Нет действительных сообщений для промпта LLM.")
            all_available_text_models = storage.get_available_models(for_dev_mode=is_developer_mode)
            full_response = await attempt_llm_request(messages_to_send=valid_llm_messages, initial_model_key=initial_model_key, model_config_dict=PROVIDERS, available_model_keys=all_available_text_models, g4f_function=g4f.ChatCompletion.create_async, is_image_generation=False, max_attempts=MAX_PROVIDER_RETRIES, chat_id_for_log=chat_id, user_id_for_log=user_id, status_message_to_update=status_msg, bot_for_status_update=message.bot, is_developer_mode=is_developer_mode)
            if full_response: break
            else:
                if not is_developer_mode and attempt_count < max_external_attempts: await asyncio.sleep(RETRY_DELAY_SECONDS); continue
                raise ValueError("Пустой ответ от LLM (внутренний).")
        except Exception as e:
            logger.error(f"Внешняя попытка {attempt_count} провалилась. Ошибка: {type(e).__name__} - {e}")
            if is_developer_mode:
                full_response = f"❌ Ошибка в режиме разработчика (модель: <code>{escape_html(initial_model_key)}</code>):\n<code>{escape_html(str(e))}</code>\nТип: <code>{type(e).__name__}</code>"
                break
            elif attempt_count == 1:
                if status_msg:
                    try:
                        await message.bot.edit_message_text(text=random.choice(ERROR_RESPONSES_1), chat_id=status_msg.chat.id, message_id=status_msg.message_id, parse_mode=None)
                    except Exception as e_edit:
                        await safe_send_message(message, random.choice(ERROR_RESPONSES_1), reply_to_message=True, parse_mode=None, bot_instance=message.bot)
                else:
                    await safe_send_message(message, random.choice(ERROR_RESPONSES_1), reply_to_message=True, parse_mode=None, bot_instance=message.bot)
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            elif attempt_count == max_external_attempts:
                full_response = random.choice(ERROR_RESPONSES_FINAL)
                break
    if status_msg:
        try: await status_msg.delete()
        except Exception as e_del: logger.warning(f"Не удалось удалить сообщение статуса {status_msg.message_id}: {e_del}")
    if full_response and full_response not in ERROR_RESPONSES_1 and full_response not in ERROR_RESPONSES_FINAL and not (is_developer_mode and "Ошибка" in full_response):
        context.add_message("user", question, memory_type)
        context.add_message("assistant", full_response[:8192], memory_type)
    else:
        logger.warning(f"Не добавляю в контекст из-за ошибки в чате {chat_id}.")
        if not is_developer_mode and (full_response in ERROR_RESPONSES_FINAL or not full_response):
            if context.short_term and context.short_term[-1]["role"] == "user": context.short_term.pop()
            elif context.long_term and context.long_term[-1]["role"] == "user": context.long_term.pop()
    reply_target_message = message
    if message.reply_to_message:
        if storage.get_reply_mode(chat_id) == "original":
            reply_target_message = message.reply_to_message
    await safe_send_message(target=reply_target_message, text=full_response, parse_mode=ParseMode.HTML, reply_to_message=True, bot_instance=message.bot)

# --- Command Handlers ---

@router.message(Command("start", "help", "старт", "помощь"))
async def cmd_start(message: types.Message, bot: Bot):
    if not message.from_user: return
    global BOT_USERNAME
    if not BOT_USERNAME:
        try:
            me = await bot.get_me()
            BOT_USERNAME = me.username or "UnknownBot"
        except Exception as e:
            logger.error(f"Не удалось получить имя пользователя бота: {e}")
            BOT_USERNAME = "UnknownBot"

    help_text = (
        f"👋 Привет! Я <b>Ждёмби-бот</b> v{__version__}\n\n"
        "Я умею отвечать на вопросы, генерировать изображения и просто болтать в чате.\n\n"
        "<b>Основные команды:</b>\n"
        "• <code>/ask [ваш вопрос]</code> - Задать вопрос боту (учитывает краткосрочную память).\n"
        "   <i>(Можно также ответить на сообщение командой /ask)</i>\n"
        "• <code>/ask [вопрос к картинке]</code> - <b>(NEW)</b> Отправьте картинку с этой командой в подписи, чтобы задать вопрос по ней.\n"
        "• <code>/long [текст]</code> - Добавить информацию в долгосрочную память бота.\n"
        "• <code>/image [описание]</code> - Сгенерировать изображение по тексту.\n"
        "• <code>/clear</code> - Очистить краткосрочную память.\n"
        "• <code>/fullreset</code> - Полностью очистить всю память и настройки.\n"
        "• <code>/model</code> & <code>/imagemodel</code> - Выбрать другие модели ИИ.\n"
        "• <code>/memstats</code> - Показать статистику по памяти.\n"
        "• <code>/context</code> - Настроить использование контекста.\n"
        "• <code>/devmode</code> - Включить/выключить режим разработчика.\n\n"
        "<b>Для групповых чатов:</b>\n"
        f"• <code>/sglypa [минуты]</code> - Включить/выключить режим случайных сообщений.\n"
        f"• Просто <b>упомяните меня</b> (@{BOT_USERNAME}) в начале сообщения, чтобы задать вопрос."
    )
    await safe_send_message(message, help_text, reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=bot)

# Найдите эту функцию в bot.py
@router.message(Command("ask", "аск", "bot", "бот", "ждёмби", "zhdomby", "ждемби", "zhdyomby", "брат"))
async def cmd_ask(message: types.Message, command: CommandObject, bot: Bot):
    """
    Handles the /ask command for both text and image queries.
    It checks for an image in the current message or the replied-to message.
    It now correctly handles albums by focusing on the specific replied-to image.
    """
    if not message.from_user: return

    # --- НОВАЯ УЛУЧШЕННАЯ ЛОГИКА ---
    photo_message: Optional[types.Message] = None
    question_source_message: types.Message = message

    # Сценарий 1: Команда в подписи к фото (даже если это часть альбома)
    if message.photo:
        photo_message = message
        logger.info(f"Изображение найдено в сообщении с командой (ID: {photo_message.message_id})")

    # Сценарий 2: Команда является ответом на сообщение с фото
    elif message.reply_to_message and message.reply_to_message.photo:
        photo_message = message.reply_to_message
        question_source_message = message.reply_to_message # Вопрос будет ассоциироваться с картинкой
        logger.info(f"Изображение найдено в сообщении, на которое ответили (ID: {photo_message.message_id})")
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    # --- SCENARIO 1: Process Image Query ---
    if photo_message:
        logger.info(f"Обработка /ask с изображением от пользователя {message.from_user.id} в чате {message.chat.id}")
        status_msg = await safe_send_message(message, "🖼️ Анализирую изображение...", reply_to_message=True, parse_mode=None, bot_instance=bot)

        try:
            # Берем фото самого высокого разрешения
            photo = photo_message.photo[-1]
            image_bytes_io = await bot.download(photo)
            if image_bytes_io is None:
                raise ValueError("Не удалось скачать файл, download вернул None.")
            image_bytes = image_bytes_io.read()
        except Exception as e:
            logger.error(f"Не удалось скачать фото: {e}")
            if status_msg: await status_msg.delete()
            await safe_send_message(message, "❌ Не удалось скачать изображение.", reply_to_message=True, parse_mode=None, bot_instance=bot)
            return

        user_prompt_for_vision = ""
        # Если команда была в подписи к картинке
        if message.caption and command.command:
            user_prompt_for_vision = message.caption.replace(f"/{command.command}", '', 1).strip()
        # Если команда была в отдельном сообщении-ответе
        elif command.args:
            user_prompt_for_vision = command.args.strip()

        vision_response = await get_image_description_gemini(image_bytes, user_prompt_for_vision)

        if "Ошибка:" in vision_response or "не удалось обработать" in vision_response:
            if status_msg: await status_msg.delete()
            await safe_send_message(message, f"⚠️ {vision_response}", reply_to_message=True, parse_mode=None, bot_instance=bot)
            return

        current_user_name = escape_html(message.from_user.full_name or message.from_user.first_name)
        timestamp_llm_format = time.strftime('[Время: %H:%M, Дата: %d.%m.%Y]', time.localtime())

        question_for_main_llm = (
            f"{timestamp_llm_format}\n"
            f"[Пользователь: {current_user_name}]\n"
            f"[Контекст из изображения]: На картинке изображено следующее: \"{escape_html(vision_response)}\".\n"
        )

        if user_prompt_for_vision:
            question_for_main_llm += f"[Вопрос пользователя к картинке]: \"{escape_html(user_prompt_for_vision)}\"\n"
            question_for_main_llm += "[Инструкция для LLM]: Твоя задача - работать с ТЕКСТОВЫМ ОПИСАНИЕМ изображения. Основываясь на этом описании, ответь на вопрос пользователя. Не говори, что ты не видишь картинки, анализируй предоставленный текст."
        else:
            question_for_main_llm += "[Инструкция для LLM]: Пользователь прислал картинку без дополнительного вопроса. Твоя задача - дать развернутый и интересный комментарий к тому, что изображено, основываясь на предоставленном описании. Не говори, что ты не видишь картинки, анализируй предоставленный текст."

        if status_msg:
            try: await status_msg.delete()
            except TelegramAPIError: pass
        
        # Мы передаем message, чтобы ответ был на команду, а не на картинку
        await process_question(message, question_for_main_llm, memory_type="short", command_used="ask_vision")

    # --- SCENARIO 2: Process Text-Only Query ---
    else:
        logger.info(f"Обработка команды ask/alias (только текст) от пользователя {message.from_user.id} в чате {message.chat.id}")
        await handle_command_with_question(message=message, command=command, allow_reply=True, memory_type="short")


async def handle_command_with_question(message: types.Message, command: CommandObject, allow_reply: bool, memory_type: str):
    """
    Вспомогательная функция для форматирования текстового вопроса для LLM.
    Сейчас используется только для текстовых команд, таких как /ask (без изображения) и /long.
    """
    command_name = command.command.lower() if command.command else "unknown"
    if not message.from_user: return
    
    chat_id, user_id = message.chat.id, message.from_user.id
    await log_user_activity(chat_id, message.chat.title or "Private", message.from_user.full_name, user_id, message.chat.type)

    current_user_name = escape_html(message.from_user.full_name or message.from_user.first_name)
    full_question_text_for_llm: Optional[str] = None
    raw_question_content: Optional[str] = None
    llm_instruction = ""
    timestamp_llm_format = time.strftime('[Время: %H:%M:%S, Дата: %d.%m.%Y]', time.localtime())

    if command.args:
        raw_question_content = command.args.strip()
        if allow_reply and message.reply_to_message:
            original_msg = message.reply_to_message
            original_author_name = escape_html(original_msg.from_user.full_name or original_msg.from_user.first_name) if original_msg.from_user else "Неизвестный"
            original_text_content = original_msg.text or original_msg.caption
            if original_text_content:
                original_text_escaped = escape_html(original_text_content.strip())[:MAX_QUESTION_LENGTH]
                command_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])
                full_question_text_for_llm = (f"{timestamp_llm_format}\n[Пользователь: {current_user_name}]\n[Ответ на сообщение (Автор: {original_author_name})]\n"
                                              f"[Текст оригинала: {original_text_escaped}]\n[Текст ответа: {command_text_escaped}]")
                llm_instruction = "Ответь на комментарий, учитывая контекст сообщения, на которое он отвечает."
            else:
                question_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])
                full_question_text_for_llm = (f"{timestamp_llm_format}\n[Пользователь: {current_user_name}]\n[Сообщение: {question_text_escaped}]")
        else:
            question_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])
            full_question_text_for_llm = (f"{timestamp_llm_format}\n[Пользователь: {current_user_name}]\n[Сообщение: {question_text_escaped}]")
    elif allow_reply and message.reply_to_message:
        original_msg = message.reply_to_message
        raw_question_content = original_msg.text or original_msg.caption
        if not raw_question_content:
            await safe_send_message(message, f"ℹ️ Команда /{command_name} была использована в ответ на сообщение без текста.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
            return
        raw_question_content = raw_question_content.strip()
        original_author_name = escape_html(original_msg.from_user.full_name or original_msg.from_user.first_name) if original_msg.from_user else "Неизвестный"
        original_text_escaped = escape_html(raw_question_content[:MAX_QUESTION_LENGTH])
        full_question_text_for_llm = (f"{timestamp_llm_format}\n[Пользователь: {current_user_name}]\n[Контекст (Автор: {original_author_name})]: {original_text_escaped}\n"
                                      f"[Подразумеваемый вопрос: (относится к тексту выше)]")
        llm_instruction = f"Пользователь {current_user_name} ответил командой /{command_name} на сообщение от {original_author_name}. Сформулируй релевантный ответ или задай уточняющий вопрос к этому сообщению."
    else:
        if memory_type == "long": await safe_send_message(message, "ℹ️ Укажите текст для долгосрочной памяти.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        else: await safe_send_message(message, "❌ Укажите ваш вопрос после команды или ответьте на сообщение.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return

    if raw_question_content and len(raw_question_content) > MAX_QUESTION_LENGTH:
        await safe_send_message(message, f"❌ Текст вопроса слишком длинный (>{MAX_QUESTION_LENGTH} симв.).", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return

    if full_question_text_for_llm and llm_instruction:
        full_question_text_for_llm += f"\n[Инструкция для LLM]: {llm_instruction}"
    elif not full_question_text_for_llm:
        logger.error("full_question_text_for_llm is None before processing.")
        await safe_send_message(message, "Внутренняя ошибка.", reply_to_message=True, bot_instance=message.bot)
        return

    await process_question(message, full_question_text_for_llm, memory_type, command_used=command_name)


@router.message(Command("long", "лонг"))
async def cmd_long(message: types.Message, command: CommandObject):
    logger.info(f"Обработка команды long от пользователя {message.from_user.id if message.from_user else 'unknown'} в чате {message.chat.id}")
    await handle_command_with_question(message=message, command=command, allow_reply=False, memory_type="long")

@router.message(Command("clear", "клир", "клиар", "отчистить", "сброс"))
async def cmd_clear(message: types.Message):
    if not message.from_user: return
    key = storage._get_key(message.chat.id, message.from_user.id, is_group_chat(message))
    if key in storage.contexts:
        storage.contexts[key].clear_short()
        await safe_send_message(message, "🧽 Краткосрочный контекст очищен!", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
    else:
        await safe_send_message(message, "ℹ️ Контекст не найден.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)

@router.message(Command("fullreset"))
async def cmd_fullreset(message: types.Message):
    if not message.from_user: return
    await storage.clear_all(message.chat.id, message.from_user.id, is_group_chat(message), message.bot)

@router.message(Command("model", "модель"))
async def cmd_model(message: types.Message, state: FSMContext):
    if not message.from_user: return
    is_dev_mode = storage.get_developer_mode(message.chat.id)
    available_models = storage.get_available_models(for_dev_mode=is_dev_mode)
    if not available_models:
         await safe_send_message(message, "❌ Нет доступных моделей.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
         return
    builder = ReplyKeyboardBuilder()
    current_model = storage.get_model_key(message.chat.id)
    for model_key in available_models:
        builder.add(KeyboardButton(text=f"✅ {model_key}" if model_key == current_model else model_key))
    builder.adjust(2)
    dev_mode_hint = " (Режим разработчика)" if is_dev_mode else ""
    await safe_send_message(message, f"🤖 Выберите модель для <b>текста</b>.{dev_mode_hint}\nТекущая: <b>{escape_html(current_model)}</b>", reply_to_message=False, reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True), parse_mode=ParseMode.HTML, bot_instance=message.bot)
    await state.set_state(Form.model_selection)

@router.message(Form.model_selection)
async def process_model_selection(message: types.Message, state: FSMContext):
    if not message.from_user or not message.text: return
    selected_model_key = message.text.replace("✅ ", "").strip()
    storage.set_model(message.chat.id, selected_model_key)
    current_model_after_set = storage.get_model_key(message.chat.id)
    if current_model_after_set == selected_model_key:
        await safe_send_message(message, f"✅ Модель изменена на <b>{escape_html(selected_model_key)}</b>!", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML, bot_instance=message.bot)
    else:
        await safe_send_message(message, f"❌ Некорректный выбор или модель '<b>{escape_html(selected_model_key)}</b>' доступна только в режиме разработчика.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML, bot_instance=message.bot)
    await state.clear()

@router.message(Command("imagemodel"))
async def cmd_imagemodel(message: types.Message, state: FSMContext):
    if not message.from_user: return
    is_dev_mode = storage.get_developer_mode(message.chat.id)
    available_image_models = storage.get_available_image_models(for_dev_mode=is_dev_mode)
    if not available_image_models:
         await safe_send_message(message, "❌ Нет доступных моделей для изображений.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
         return
    builder = ReplyKeyboardBuilder()
    current_image_model = storage.get_image_model_key(message.chat.id)
    for model_key in available_image_models:
        builder.add(KeyboardButton(text=f"✅ {model_key}" if model_key == current_image_model else model_key))
    builder.adjust(2)
    dev_mode_hint = " (Режим разработчика)" if is_dev_mode else ""
    await safe_send_message(message, f"🖼 Выберите модель для <b>изображений</b>.{dev_mode_hint}\nТекущая: <b>{escape_html(current_image_model)}</b>", reply_to_message=False, reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True), parse_mode=ParseMode.HTML, bot_instance=message.bot)
    await state.set_state(Form.image_model_selection)

@router.message(Form.image_model_selection)
async def process_image_model_selection(message: types.Message, state: FSMContext):
    if not message.from_user or not message.text: return
    selected_model_key = message.text.replace("✅ ", "").strip()
    storage.set_image_model(message.chat.id, selected_model_key)
    current_image_model_after_set = storage.get_image_model_key(message.chat.id)
    if current_image_model_after_set == selected_model_key:
        await safe_send_message(message, f"✅ Модель для изображений изменена на <b>{escape_html(selected_model_key)}</b>!", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML, bot_instance=message.bot)
    else:
        await safe_send_message(message, f"❌ Некорректный выбор или модель '<b>{escape_html(selected_model_key)}</b>' доступна только в режиме разработчика.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML, bot_instance=message.bot)
    await state.clear()

@router.message(Command("replymode"))
async def cmd_replymode(message: types.Message):
    if not message.from_user: return
    chat_id = message.chat.id
    current_mode = storage.get_reply_mode(chat_id)
    new_mode = "direct" if current_mode == "original" else "original"
    storage.set_reply_mode(chat_id, new_mode)
    mode_description = "на исходное сообщение" if new_mode == "original" else "напрямую на команду"
    await safe_send_message(message, f"⚙️ Режим ответа изменен на: <b>{mode_description}</b>.", reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=message.bot)

@router.message(Command("context", "контекст"))
async def cmd_context_settings(message: types.Message, state: FSMContext):
    if not message.from_user: return
    await _send_context_settings_menu(message, state)

async def _send_context_settings_menu(message: types.Message, state: FSMContext):
    chat_id = message.chat.id
    settings_prompt = storage.get_chat_context_settings(chat_id)
    read_general = storage.get_read_general_chat_messages_enabled(chat_id)
    limit = storage.get_short_term_limit(chat_id)
    builder = ReplyKeyboardBuilder()
    buttons_info = {"Легенда": "legend", "Долгосрочная память": "long", "Краткосрочная память": "short"}
    for name, key in buttons_info.items():
        status = "Вкл" if settings_prompt.get(key, True) else "Выкл"
        builder.add(KeyboardButton(text=f"{name}: {status}"))
    read_status = "Вкл" if read_general else "Выкл"
    builder.add(KeyboardButton(text=f"Чтение общих сообщений чата: {read_status}"))
    builder.add(KeyboardButton(text=f"Лимит краткосрочной памяти: {limit}"))
    builder.add(KeyboardButton(text="Готово"))
    builder.adjust(1)
    await safe_send_message(message, "⚙️ Настройте использование контекста:", reply_to_message=False, reply_markup=builder.as_markup(resize_keyboard=True, one_time_keyboard=True), parse_mode=ParseMode.HTML, bot_instance=message.bot)
    await state.set_state(Form.context_settings_menu)

@router.message(Form.context_settings_menu)
async def process_context_settings_selection(message: types.Message, state: FSMContext):
    if not message.from_user or not message.text: return
    chat_id = message.chat.id
    selected_text = message.text.strip()
    if selected_text == "Готово":
        await safe_send_message(message, "✅ Настройки сохранены.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=None, bot_instance=message.bot)
        await state.clear()
        return
    setting_map = {"Легенда": "legend", "Долгосрочная память": "long", "Краткосрочная память": "short"}
    found_key: Optional[str] = None
    for name, key in setting_map.items():
        if selected_text.startswith(name): found_key = key; break
    if found_key:
        current_state = storage.get_chat_context_settings(chat_id).get(found_key, True)
        storage.set_chat_context_setting(chat_id, found_key, not current_state)
        await _send_context_settings_menu(message, state)
        return
    if selected_text.startswith("Чтение общих сообщений чата:"):
        current = storage.get_read_general_chat_messages_enabled(chat_id)
        storage.set_read_general_chat_messages_enabled(chat_id, not current)
        await _send_context_settings_menu(message, state)
        return
    if selected_text.startswith("Лимит краткосрочной памяти:"):
        await safe_send_message(message, f"🔢 Введите новый лимит (число).\nТекущий: <b>{storage.get_short_term_limit(chat_id)}</b>.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML, bot_instance=message.bot)
        await state.set_state(Form.set_short_term_limit)
        return
    await safe_send_message(message, "❌ Неизвестная настройка.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)

@router.message(Form.set_short_term_limit)
async def process_short_term_limit_input(message: types.Message, state: FSMContext):
    if not message.from_user or not message.text: return
    try:
        new_limit = int(message.text.strip())
        if new_limit < 0:
            await safe_send_message(message, "❌ Лимит должен быть >= 0.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
            return
        storage.set_short_term_limit(message.chat.id, new_limit)
        await safe_send_message(message, f"✅ Лимит установлен: <b>{new_limit}</b>.", reply_to_message=False, reply_markup=ReplyKeyboardRemove(), parse_mode=ParseMode.HTML, bot_instance=message.bot)
        await _send_context_settings_menu(message, state)
    except ValueError:
        await safe_send_message(message, "❌ Введите целое число.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
        await state.set_state(Form.set_short_term_limit)

@router.message(Command("image", "img", "имейдж", "имж", "имг"))
async def cmd_image(message: types.Message, command: CommandObject, bot: Bot):
    if not message.from_user: return
    chat_id, user_id = message.chat.id, message.from_user.id
    is_developer_mode = storage.get_developer_mode(chat_id)
    prompt = command.args.strip() if command.args else (message.reply_to_message.text or message.reply_to_message.caption if message.reply_to_message else None)
    if not prompt:
        await safe_send_message(message, "❌ Укажите описание для изображения.", reply_to_message=True, parse_mode=None, bot_instance=bot)
        return
    if len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        await safe_send_message(message, f"❌ Описание слишком длинное (>{MAX_IMAGE_PROMPT_LENGTH} симв.).", reply_to_message=True, parse_mode=None, bot_instance=bot)
        return
    initial_image_model_key = storage.get_image_model_key(chat_id)
    if not IMAGE_PROVIDERS_CONFIG:
         await safe_send_message(message, "❌ Список моделей для изображений пуст.", reply_to_message=True, parse_mode=None, bot_instance=bot)
         return
    status_msg = await safe_send_message(message, "🎨 Генерирую изображение...", reply_to_message=True, parse_mode=None, bot_instance=bot)
    async def image_generation_orchestrator(prompt_text:str, provider_class:type[BaseProvider], model_id:str, timeout_val:int) -> List[str]:
        _image_urls_internal: List[str] = []
        try:
            response_list = await g4f.image.create_async(prompt=prompt_text, provider=provider_class, model=model_id, timeout=timeout_val)
            if isinstance(response_list, list) and all(isinstance(url, str) for url in response_list):
                return response_list
        except (NotImplementedError, Exception):
            pass
        image_gen_prompt_for_chat = f"Generate an image based on: {prompt_text}. Provide ONLY the direct image URL."
        chat_response_obj = await g4f.ChatCompletion.create_async(model=model_id, provider=provider_class, messages=[{"role": "user", "content": image_gen_prompt_for_chat}], timeout=timeout_val)
        _response_str_for_fallback = str(chat_response_obj)
        direct_urls = re.findall(r'https?://[^\s()\'"]+\.(?:png|jpe?g|webp|gif)\b', _response_str_for_fallback, re.IGNORECASE)
        markdown_urls = re.findall(r'\[.*?\]\((https?://[^\s()\'"]+)\)', _response_str_for_fallback)
        _image_urls_internal = direct_urls + markdown_urls
        if _image_urls_internal: return _image_urls_internal
        raise ValueError(f"Генерация изображения не удалась для {provider_class.__name__}: URL не найдены.")
    try:
        all_available_image_models = storage.get_available_image_models(for_dev_mode=is_developer_mode)
        image_urls = await attempt_llm_request(messages_to_send=[], initial_model_key=initial_image_model_key, model_config_dict=IMAGE_PROVIDERS_CONFIG, available_model_keys=all_available_image_models, g4f_function=image_generation_orchestrator, is_image_generation=True, image_generation_prompt=prompt, max_attempts=MAX_PROVIDER_RETRIES, chat_id_for_log=chat_id, user_id_for_log=user_id, status_message_to_update=status_msg, bot_for_status_update=bot, is_developer_mode=is_developer_mode)
        if status_msg: await status_msg.delete()
        if image_urls and isinstance(image_urls, list) and image_urls[0]:
            try:
                await message.reply_photo(photo=image_urls[0], caption=f"🖼 {escape_html(prompt[:1000])}")
            except (TelegramBadRequest, Exception) as e:
                 await safe_send_message(message, f"⚠️ Не удалось отправить изображение. ({escape_html(str(e))})", reply_to_message=True, parse_mode=None, bot_instance=bot)
        else:
            await safe_send_message(message, "❌ Не удалось получить URL изображения.", reply_to_message=True, parse_mode=None, bot_instance=bot)
    except Exception as e:
        if status_msg: await status_msg.delete()
        error_message_text = f"❌ Ошибка генерации: {escape_html(str(e))}"
        await safe_send_message(message, error_message_text, reply_to_message=True, parse_mode=ParseMode.HTML, bot_instance=bot)

@router.message(Command("sglypa", "сглыпа", "рандом"))
async def cmd_sglypa(message: types.Message, command: CommandObject):
    if not is_group_chat(message):
        await safe_send_message(message, "ℹ️ Команда доступна только в группах.", reply_to_message=False, parse_mode=None, bot_instance=message.bot)
        return
    chat_id = message.chat.id
    settings = storage.init_chat_settings(chat_id)
    if command.args:
        try:
            minutes = int(command.args)
            if minutes * 60 < MIN_INTERVAL:
                await safe_send_message(message, f"❌ Минимальный интервал: <b>{MIN_INTERVAL // 60}</b> минут.", reply_to_message=True, parse_mode=ParseMode.HTML, bot_instance=message.bot)
                return
            settings['interval'] = minutes * 60
            settings['active'] = True
            settings['last_sent'] = time.time()
            await safe_send_message(message, f"✅ Режим случайных сообщений <b>активирован</b> с интервалом <b>{minutes}</b> мин.", reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=message.bot)
        except (ValueError, Exception):
            await safe_send_message(message, "❌ Укажите целое число минут.", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
    else:
        settings['active'] = not settings['active']
        status = "активирован" if settings['active'] else "выключен"
        interval_minutes = settings.get('interval', RANDOM_MESSAGE_INTERVAL) // 60
        interval_info = f"\nИнтервал: <b>{interval_minutes}</b> мин." if settings['active'] else ""
        await safe_send_message(message, f"🔔 Режим случайных сообщений <b>{status}</b>!{interval_info}", reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=message.bot)

@router.message(Command("memstats", "память"))
async def cmd_memstats(message: types.Message):
    if not message.from_user: return
    context = storage.get_context(message.chat.id, message.from_user.id, is_group_chat(message))
    settings = storage.get_chat_context_settings(message.chat.id)
    response_text = (f"📊 <b>Статистика памяти:</b>\n"
                     f"Краткосрочная: <b>{len(context.short_term)}</b> (лимит: {storage.get_short_term_limit(message.chat.id)})\n"
                     f"Долгосрочная: <b>{len(context.long_term)}</b>\n\n"
                     f"⚙️ <b>Настройки контекста:</b>\n"
                     f"Легенда: <b>{'Вкл' if settings.get('legend', True) else 'Выкл'}</b>\n"
                     f"Долгосрочная: <b>{'Вкл' if settings.get('long', True) else 'Выкл'}</b>\n"
                     f"Краткосрочная: <b>{'Вкл' if settings.get('short', True) else 'Выкл'}</b>\n"
                     f"Чтение чата: <b>{'Вкл' if storage.get_read_general_chat_messages_enabled(message.chat.id) else 'Выкл'}</b>\n\n"
                     f"🛠 Режим разработчика: <b>{'Вкл' if storage.get_developer_mode(message.chat.id) else 'Выкл'}</b>")
    await safe_send_message(message, response_text, reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=message.bot)

@router.message(Command("devmode"))
async def cmd_devmode(message: types.Message):
    if not message.from_user: return
    chat_id = message.chat.id
    new_mode = not storage.get_developer_mode(chat_id)
    storage.set_developer_mode(chat_id, new_mode)
    await safe_send_message(message, f"🛠 Режим разработчика <b>{'включен' if new_mode else 'выключен'}</b>.", reply_to_message=False, parse_mode=ParseMode.HTML, bot_instance=message.bot)

# --- Message Handlers ---
@router.message(F.text, F.chat.type == ChatType.PRIVATE)
async def handle_private_text_message(message: types.Message):
    if not message.from_user or not message.text or message.text.startswith('/'): return
    if len(message.text) > MAX_QUESTION_LENGTH:
        await safe_send_message(message, f"❌ Вопрос слишком длинный (>{MAX_QUESTION_LENGTH} симв.).", reply_to_message=True, parse_mode=None, bot_instance=message.bot)
        return
    current_user_name = escape_html(message.from_user.full_name or message.from_user.first_name)
    timestamp = time.strftime('[Время: %H:%M:%S]', time.localtime())
    question_text = (f"{timestamp}\n[Пользователь: {current_user_name}]\n[Сообщение: {escape_html(message.text)}]\n"
                     f"[Инструкция]: Ответь на вопрос в личном сообщении.")
    await process_question(message, question_text, memory_type="short", command_used="private_message")

@router.message(F.text, F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def handle_mention_or_group_text(message: types.Message, bot: Bot):
    if not message.from_user or not message.text: return
    global BOT_USERNAME
    if not BOT_USERNAME:
        try: me = await bot.get_me(); BOT_USERNAME = me.username or "UnknownBot"
        except Exception: BOT_USERNAME = "UnknownBot"
    if BOT_USERNAME and BOT_USERNAME != "UnknownBot":
        mention_pattern = re.compile(rf'^(?:@{re.escape(BOT_USERNAME)}\b\s*)+', re.IGNORECASE)
        text_after_mention = mention_pattern.sub('', message.text).lstrip()
        if len(text_after_mention) < len(message.text):
            if text_after_mention.startswith('/'): return
            question_raw = text_after_mention.strip() or (message.reply_to_message.text or message.reply_to_message.caption if message.reply_to_message else None)
            if not question_raw: return
            if len(question_raw) > MAX_QUESTION_LENGTH:
                await safe_send_message(message, f"❌ Вопрос слишком длинный (>{MAX_QUESTION_LENGTH} симв.).", reply_to_message=True, parse_mode=None, bot_instance=bot)
                return
            user_name = escape_html(message.from_user.full_name or message.from_user.first_name)
            timestamp = time.strftime('[Время: %H:%M:%S]', time.localtime())
            question_text = (f"{timestamp}\n[Пользователь: {user_name}]\n[Сообщение (упоминание): {escape_html(question_raw)}]\n"
                             f"[Инструкция]: Ответь на вопрос в групповом чате.")
            await process_question(message, question_text, memory_type="short", command_used="mention")
            return
    if not message.text.startswith('/'):
        if storage.get_read_general_chat_messages_enabled(message.chat.id):
            context = storage.get_context(message.chat.id, 0, True)
            user_name = message.from_user.full_name or message.from_user.first_name
            context.add_message("user", f"Пользователь {user_name}: {message.text}", "short")

# --- Startup and Shutdown ---
async def on_startup(bot: Bot):
    global BOT_USERNAME
    try:
        me = await bot.get_me()
        BOT_USERNAME = me.username or "UnknownBot"
        logger.info(f"Бот @{BOT_USERNAME} запущен!")
        if PROVIDERS and PROVIDER_MAP: asyncio.create_task(send_random_messages(bot))
        asyncio.create_task(save_state_periodically())
    except Exception as e:
        logger.critical(f"Запуск не удался: {e}", exc_info=True)

async def on_shutdown(bot: Bot):
    logger.info("Завершение работы...")
    try:
        logger.info("Сохранение состояния...")
        for key, context in storage.contexts.items(): context.save_long_term()
        all_keys = set(list(storage.models.keys()) + list(storage.image_models.keys()) + list(storage.reply_modes.keys()) + list(storage.context_types_enabled.keys()) + list(storage.read_general_chat_messages_enabled.keys()) + list(storage.short_term_limits.keys()) + list(storage.developer_modes.keys()))
        for key in all_keys: storage._save_settings_for_key(key)
        if bot and bot.session: await bot.session.close()
        logger.info("Завершение работы завершено.")
    except Exception as e:
         logger.error(f"Ошибка при завершении работы: {e}", exc_info=True)

if __name__ == "__main__":
    if bot is None:
        logger.critical("Экземпляр бота равен None. Выход.")
        exit(1)
    dp.include_router(router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    try:
        logger.info("Запуск опроса...")
        dp.run_polling(bot, allowed_updates=dp.resolve_used_update_types())
    except (KeyboardInterrupt, Exception) as e:
        logger.critical(f"Опрос остановлен: {e}", exc_info=True)
