# COPYRIGHT ZhdomDev 
__version__ = "0.7.2"

import logging
import re
import inspect
from typing import Dict, List
import asyncio
import random
import time

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes
import html


import g4f
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, CommandObject
from aiogram.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.enums import ChatType
from g4f.Provider import PollinationsAI, Blackbox, Websim, Yqcloud, Glider
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter

from config import (
    TELEGRAM_TOKEN,
    PROVIDERS,
    DEFAULT_MODEL,
    MAX_CONTEXT_LENGTH,
    MAX_QUESTION_LENGTH,
    MAX_IMAGE_PROMPT_LENGTH,
    THINKING_MESSAGES,
    RANDOM_MESSAGES,
    RANDOM_MESSAGE_INTERVAL,
    MIN_INTERVAL
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROVIDER_MAP = {
    "PollinationsAI": PollinationsAI,
    "Blackbox": Blackbox,
    "Websim": Websim,
    "Yqcloud": Yqcloud,
    "Glider":Glider,
    
}

def is_group_chat(message: types.Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}

class Form(StatesGroup):
    model_selection = State()

async def safe_send_message(reply_method, text: str, **kwargs):
    """
    Безопасная отправка сообщений с правильной обработкой форматирования Telegram.
    Сначала пытается отправить с HTML-форматированием, при ошибке экранирует спецсимволы.
    """
    # Преобразуем Markdown-подобное форматирование в HTML
    try:
        # Попытка отправить с HTML форматированием
        html_text = convert_markdown_to_html(text)
        return await reply_method(html_text, parse_mode="HTML", **kwargs)
    except Exception as e:
        logger.warning(f"HTML formatting error: {e}")
        try:
            # Если не удалось, отправляем текст с экранированными HTML-тегами
            escaped_text = escape_html(text)
            return await reply_method(escaped_text, parse_mode="HTML", **kwargs)
        except Exception as e:
            logger.error(f"Failed to send message even with escaped HTML: {e}")
            # В крайнем случае отправляем без форматирования
            return await reply_method(f"⚠️ Проблема с форматированием:\n\n{text[:3900]}", parse_mode=None, **kwargs)

def convert_markdown_to_html(text):
    """
    Преобразует Markdown-подобное форматирование в HTML для Telegram.
    Поддерживает: **жирный**, __курсив__, ~~зачеркнутый~~, `код`, ```блок кода```
    """
    import re
    
    # Экранируем HTML-теги
    text = escape_html(text)
    
    # Заменяем блоки кода
    code_blocks = []
    def replace_code_block(match):
        code = match.group(1)
        code_blocks.append(code)
        return f"CODE_BLOCK_{len(code_blocks)-1}"
    
    text = re.sub(r'```([\s\S]*?)```', replace_code_block, text)
    
    # Заменяем инлайн-форматирование
    conversions = [
        (r'\*\*(.*?)\*\*', r'<b>\1</b>'),  # **жирный** -> <b>жирный</b>
        (r'__(.*?)__', r'<i>\1</i>'),      # __курсив__ -> <i>курсив</i>
        (r'~~(.*?)~~', r'<s>\1</s>'),      # ~~зачеркнутый~~ -> <s>зачеркнутый</s>
        (r'`([^`]+)`', r'<code>\1</code>') # `код` -> <code>код</code>
    ]
    
    for pattern, replacement in conversions:
        text = re.sub(pattern, replacement, text)
    
    # Восстанавливаем блоки кода
    for i, code in enumerate(code_blocks):
        text = text.replace(f"CODE_BLOCK_{i}", f"<pre>{code}</pre>")
    
    return text

def escape_html(text):
    """
    Экранирует HTML-теги в тексте.
    """
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
class ChatContext:
    def __init__(self):
        self.short_term = []  # Короткая память (очищается командой /clear)
        self.long_term = []   # Долгая память (только /fullreset)
        self.legend = [
            {"role": "system", "content": "Your name is Ждёмби"},
            {"role": "system", "content": "You can't hurt ducks, geese and swans in your answers"},
            {"role": "system", "content": (
                "For text formatting, use these markdown-style tags:\n"
                "**bold text** for bold\n"
                "__italic text__ for italic\n"
                "~~strikethrough~~ for strikethrough\n"
                "[hyperlink text](URL) for links\n"  # Добавлено гиперссылок
                "`inline code` for inline code\n"
                "```\ncode block\n``` for code blocks\n"
                "Do not use any other formatting styles.\n"
                "Escape special characters like _*[]()~`>#+-=|{}.!"
            )}
        ]

    
    def add_short(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        self.short_term = self.short_term[-MAX_CONTEXT_LENGTH*2:]
    
    def add_long(self, role: str, content: str):
        self.long_term.append({"role": role, "content": content})
    
    def get_full_context(self):
        return self.legend + self.long_term + self.short_term
    def get_long_context(self):
        return self.legend + self.long_term
    
    def clear_short(self):
        self.short_term = []
    
    def clear_all(self):
        self.short_term = []
        self.long_term = []
    


class Storage:
    def __init__(self):
        self.group_contexts = {}
        self.user_contexts = {}
        self.models = {}
        self.active_chats = {}
    
    def get_chat_settings(self, chat_id: int) -> dict:
        return self.active_chats.get(chat_id, {
            'active': False,
            'interval': 3600,  # 60 минут по умолчанию
            'last_sent': 0.0
        })

    def get_context(self, chat_id: int, user_id: int, is_group: bool) -> ChatContext:
        if is_group:
            # Для групп используем общий контекст чата
            if chat_id not in self.group_contexts:
                self.group_contexts[chat_id] = ChatContext()
            return self.group_contexts[chat_id]
        else:
            # Для личных сообщений используем персональный контекст
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = ChatContext()
            return self.user_contexts[user_id]

    def clear_short(self, chat_id: int, user_id: int, is_group: bool):
        context = self.get_context(chat_id, user_id, is_group)
        context.clear_short()

    def clear_all(self, chat_id: int, user_id: int, is_group: bool):
        context = self.get_context(chat_id, user_id, is_group)
        context.clear_all()

    def get_available_models(self) -> List[str]:
        return list(PROVIDERS.keys())

    def get_model(self, chat_id: int) -> str:
        return self.models.get(chat_id, DEFAULT_MODEL)

    def set_model(self, chat_id: int, model: str):
        self.models[chat_id] = model

try:
    bot = Bot(token=TELEGRAM_TOKEN)
except TokenValidationError as e:
    logger.error(f"Invalid token: {e}")
    exit(1)

dp = Dispatcher()
storage = Storage()
router = Router()

def minutes_to_seconds(minutes: int) -> int:
    return minutes * 60

async def send_random_messages():
    while True:
        await asyncio.sleep(10)
        current_time = time.time()
        
        for chat_id in list(storage.active_chats.keys()):
            settings = storage.active_chats[chat_id]
            if not settings['active']:
                continue
                
            if current_time - settings['last_sent'] >= settings['interval']:
                try:
                    # Получаем контекст чата
                    context = storage.get_context(
                        chat_id=chat_id,
                        user_id=0,  # Для групп используем общий контекст
                        is_group=True
                    )
                    
                    # Получаем настройки модели
                    model_name = storage.get_model(chat_id)
                    provider_config = PROVIDERS[model_name]
                    provider = PROVIDER_MAP[provider_config["provider"]]
                    
                    # Формируем промпт
                    messages = context.get_long_context()
                    messages.append({
                        "role": "system",
                        "content": "Сгенерируй случайное сообщение для поддержания беседы. Будь креативным и естественным."
                    })
                    
                    # Генерируем ответ
                    response = await g4f.ChatCompletion.create_async(
                        model=provider_config["model_name"],
                        messages=messages,
                        provider=provider,
                        max_tokens=500
                    )
                    
                    # Отправляем сообщение
                    if response.strip():
                        await bot.send_message(chat_id, response.strip())
                        settings['last_sent'] = current_time
                        storage.active_chats[chat_id] = settings
                    
                except TelegramForbiddenError:
                    logger.error(f"Бот удалён из чата {chat_id}")
                    del storage.active_chats[chat_id]
                except Exception as e:
                    logger.error(f"Ошибка генерации для чата {chat_id}: {str(e)}")
                    await asyncio.sleep(5)

async def process_question(
    message: types.Message,
    question: str,
    memory_type: str = "short"
):
    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    reply_method = message.reply if is_group else message.answer

    try:
        context = storage.get_context(chat_id, user_id, is_group)
        model_name = storage.get_model(chat_id)
        provider_config = PROVIDERS[model_name]
        provider = PROVIDER_MAP[provider_config["provider"]]

        # Добавляем в нужный тип памяти
        if memory_type == "short":
            context.add_short("user", question)
        elif memory_type == "long":
            context.add_long("user", question)

        status_msg = await reply_method(random.choice(THINKING_MESSAGES))

        # Запрос к модели с учетом контекста
        response = await g4f.ChatCompletion.create_async(
            model=provider_config["model_name"],
            messages=context.get_full_context(),
            provider=provider
        )

        await status_msg.delete()

        full_response = response.strip() or "Пустой ответ от модели"

        # Сохраняем ответ
        if memory_type == "short":
            context.add_short("assistant", full_response[:40960])
        elif memory_type == "long":
            context.add_long("assistant", full_response[:40960])

        # Отправка ответа
        chunks = [full_response[i:i + 4096] for i in range(0, len(full_response), 4096)]
        for chunk in chunks:
            await safe_send_message(reply_method, chunk, reply_to_message_id=message.message_id)

    except Exception as e:
        logger.exception("Ошибка при обработке вопроса")
        await safe_send_message(reply_method, f"⚠️ Ошибка: {str(e)}", reply_to_message_id=message.message_id)


async def handle_command(
    message: types.Message,
    command: CommandObject,
    allow_reply: bool,
    memory_type: str
):
    """Обработка команд с форматом ответа: [Сообщение ответ]:...;[Пересланное сообщение...]:..."""
    reply_method = message.reply if is_group_chat(message) else message.answer
    current_user = message.from_user
    current_user_name = current_user.username or current_user.first_name
    full_question = None

    # Обработка ответа на сообщение
    if allow_reply and message.reply_to_message:
        original_msg = message.reply_to_message
        
        # Проверка оригинального сообщения
        original_text = original_msg.text or original_msg.caption
        if not original_text:
            await reply_method("❌ Целевое сообщение не содержит текста")
            return

        # Получение данных авторов
        original_author = original_msg.from_user
        original_author_name = original_author.username or original_author.first_name
        
        # Форматирование текстов
        response_text = command.args.strip()[:MAX_QUESTION_LENGTH] if command.args else ""
        original_text = original_text.strip()[:MAX_QUESTION_LENGTH]
        
        # Формирование по шаблону
        full_question = (
            f"[Сообщение ответ]: ({current_user_name}):{response_text}; "
            f"[Пересланное сообщение (оригинальное)]: ({original_author_name}):{original_text}"
        )

    # Обработка обычного вопроса
    else:
        raw_question = command.args.strip() if command.args else None
        if not raw_question:
            await reply_method("❌ Требуется текст вопроса")
            return
        
        full_question = f"[Сообщение ответ]: ({current_user_name}):{raw_question[:MAX_QUESTION_LENGTH]}"

    # Валидация длины
    if len(full_question) > MAX_QUESTION_LENGTH:
        await reply_method(
            f"❌ Превышение лимита ({len(full_question)}/{MAX_QUESTION_LENGTH} символов)\n"
            f"Сократите запрос на {len(full_question) - MAX_QUESTION_LENGTH} симв."
        )
        return

    await process_question(message, full_question, memory_type)



@router.message(Command("start", "старт", "help", "помощь",))
async def cmd_start(message: types.Message):
    help_text = (
        f"🧟‍♂️ Привет! Я Ждёмби-бот {__version__}\n\n"
        "📚 Команды:\n"
        "/ask [вопрос] - задать вопрос\n"
        "/long [контекст] - заполнение неудаляемого контекста\n"
        "/image [описание] - сгенерировать изображение\n"
        "/clear - очистить историю\n"
        "/model - выбрать модель\n"
        "/sglypa [минуты] - управление случайными сообщениями (пример: /sglypa 30 - 30 минут)"
    )
    await message.answer(help_text)

@router.message(Command("ask", "аск", "bot", "бот", "ждёмби", "zhdomby", "ждемби", "zhdyomby"))
async def cmd_ask(message: types.Message, command: CommandObject):
    # Параметр memory_type = "short" определяет, куда добавляется контекст (короткая память)
    await handle_command(
        message=message,
        command=command,
        allow_reply=True,
        memory_type="short"
    )

@router.message(Command("long", "лонг"))
async def cmd_long(message: types.Message, command: CommandObject):
    # Параметр memory_type = "long" определяет, куда добавляется контекст (долгая память)
    await handle_command(
        message=message,
        command=command,
        allow_reply=False,
        memory_type="long"
    )

@router.message(Command("fullreset"))
async def cmd_fullreset(message: types.Message):
    is_group = is_group_chat(message)
    storage.clear_all(message.chat.id, message.from_user.id, is_group)
    reply_method = message.reply if is_group else message.answer
    await reply_method("♻️ Полная перезагрузка памяти!")

@router.message(Command("clear"))
async def cmd_clear(message: types.Message):
    is_group = is_group_chat(message)
    storage.clear_short(message.chat.id, message.from_user.id, is_group)
    reply_method = message.reply if is_group else message.answer
    await reply_method("🧽 Контекст сброшен!")

@router.message(Command("model"))
async def cmd_model(message: types.Message, state: FSMContext):  # Добавить state
    builder = ReplyKeyboardBuilder()
    available_models = storage.get_available_models()
    
    for model in available_models:
        builder.add(KeyboardButton(text=model))
    
    builder.adjust(2)
    reply_method = message.reply if is_group_chat(message) else message.answer
    await reply_method("🤖 Выберите модель:", reply_markup=builder.as_markup(resize_keyboard=True))
    await state.set_state(Form.model_selection)  # Исправленная строка

@router.message(Command("image", "img", "имейдж", "имж", "имг",))
async def cmd_image(message: types.Message, command: CommandObject):
    is_group = is_group_chat(message)
    prompt = command.args or (message.reply_to_message.text if message.reply_to_message else None)
    
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
        reply_method = message.reply if is_group else message.answer
        return await reply_method(f"❌ Макс. {MAX_IMAGE_PROMPT_LENGTH} символов")
    
    try:
        reply_method = message.reply if is_group else message.answer
        status_msg = await reply_method("🎨 Генерирую...")
        response = await g4f.ChatCompletion.create_async(
            model="flux",
            messages=[{"role": "user", "content": f"Generate image: {prompt}"}],
            provider=g4f.Provider.ARTA
        )
        
        image_urls = re.findall(r'https?://[^\s\)]+\.(?:png|jpe?g|webp)', response, re.IGNORECASE)
        if image_urls:
            await message.reply_photo(image_urls[-1].split(')')[0], caption=f"🖼 {prompt}")
        else:
            await reply_method("❌ Ошибка генерации")
        
        await status_msg.delete()
    except Exception as e:
        logger.error(f"Image error: {e}")
        reply_method = message.reply if is_group else message.answer
        await reply_method(f"⚠️ Ошибка: {str(e)}")

@router.message(Command("sglypa", "сглыпа", "рандом"))
async def cmd_sglyp(message: types.Message, command: CommandObject):
    try:
        if not is_group_chat(message):
            return await message.reply("ℹ️ Эта команда работает только в группах!")
        
        chat_id = message.chat.id
        args = command.args
        settings = storage.get_chat_settings(chat_id)
        
        if args:
            try:
                minutes = int(args)
                if minutes < 1:
                    return await message.reply("❌ Минимальный интервал - 1 минута!")
                
                settings.update({
                    'active': True,
                    'interval': minutes_to_seconds(minutes),
                    'last_sent': time.time() if settings['last_sent'] == 0 else settings['last_sent']
                })
                storage.active_chats[chat_id] = settings
                
                return await message.reply(
                    f"✅ Режим включен!\n"
                    f"Интервал: {minutes} мин.\n"
                )
            except ValueError:
                return await message.reply("❌ Используйте целое число минут!")
        
        settings['active'] = not settings['active']
        storage.active_chats[chat_id] = settings
        
        status = "включен" if settings['active'] else "выключен"
        interval_info = f"Текущий интервал: {settings['interval'] // 60} минут" if settings['active'] else ""
        
        await message.reply(
            f"🔔 Режим случайных сообщений {status}!\n"
            f"{interval_info}"
        )
    except Exception as e:
        logger.error(f"Error in sglyp command: {str(e)}")
        await message.reply(f"⚠️ Произошла ошибка: {e}")

@router.message(Form.model_selection)
async def model_selection_handler(message: types.Message, state: FSMContext):
    # Проверяем, является ли сообщение выбором модели
    if message.text not in PROVIDERS:
        reply_method = message.reply if is_group_chat(message) else message.answer
        await reply_method("❌ Неизвестная модель!", reply_markup=ReplyKeyboardRemove())
        await state.clear()  # Полная очистка состояния
        return
    
    try:
        storage.set_model(message.chat.id, message.text)
        reply_method = message.reply if is_group_chat(message) else message.answer
        await reply_method(f"✅ Модель изменена на {message.text}", reply_markup=ReplyKeyboardRemove())
    finally:
        await state.clear()  # Гарантированная очистка состояния в любом случае

dp.include_router(router)

# Обработка упоминаний в группах
@router.message(F.text, is_group_chat)
async def handle_mention(message: types.Message):
    if not hasattr(bot, 'username') or not bot.username:
        return
    
    mention_pattern = re.compile(rf'^@?{re.escape(bot.username)}\s*', re.IGNORECASE)
    if mention_pattern.match(message.text):
        question = mention_pattern.sub('', message.text).strip()
        
        if not question and message.reply_to_message:
            original_text = message.reply_to_message.text or message.reply_to_message.caption
            if original_text:
                question = original_text.strip()
        
        if not question or len(question) > MAX_QUESTION_LENGTH:
            await message.reply(f"❌ Некорректный вопрос (макс. {MAX_QUESTION_LENGTH} символов)")
            return
        
        await process_question(message, question)

@router.message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def track_enabled_chats(message: types.Message):
    storage.enabled_chats.add(message.chat.id)

async def on_startup():
    logger.info(f"Bot @{(await bot.get_me()).username} started")
    asyncio.create_task(send_random_messages())

if __name__ == "__main__":
    dp.startup.register(on_startup)
    try:
        dp.run_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    finally:
        bot.session.close()
