# COPYRIGHT ZhdomDev 
__version__ = "0.4.2"

import logging
import re
import inspect
from typing import Dict, List
import asyncio

import g4f
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, CommandObject
from aiogram.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.enums import ChatType
from g4f.Provider import PollinationsAI, Blackbox, Websim
from aiogram.fsm.context import FSMContext

from config import (
    TELEGRAM_TOKEN,
    PROVIDERS,
    DEFAULT_MODEL,
    MAX_CONTEXT_LENGTH,
    MAX_QUESTION_LENGTH,
    MAX_IMAGE_PROMPT_LENGTH
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
}

class Form(StatesGroup):
    model_selection = State()

class ChatContext:
    def __init__(self):
        self.history = [
            {"role": "system", "content": "Use simple Markdown: **bold**, *italic*, `code`. Avoid special symbols."}
        ]
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.history = self.history[-MAX_CONTEXT_LENGTH*2:]

class Storage:
    def __init__(self):
        self.contexts: Dict[int, ChatContext] = {}
        self.user_contexts: Dict[int, ChatContext] = {}  # Для личных чатов
        self.models: Dict[int, str] = {}
    
    def get_context(self, chat_id: int, user_id: int, is_group: bool) -> ChatContext:
        if is_group:
            if chat_id not in self.contexts:
                self.contexts[chat_id] = ChatContext()
            return self.contexts[chat_id]
        else:
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = ChatContext()
            return self.user_contexts[user_id]
    
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

def is_group_chat(message: types.Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}

async def process_question(message: types.Message, question: str):
    chat_id = message.chat.id
    user_id = message.from_user.id
    is_group = is_group_chat(message)
    reply_method = message.reply if is_group else message.answer
    status_msg = None
    
    try:
        context = storage.get_context(chat_id, user_id, is_group)
        model_name = storage.get_model(chat_id)
        context.add_message("user", question)
        
        provider_config = PROVIDERS[model_name]
        provider = PROVIDER_MAP[provider_config["provider"]]
        
        # Отправляем временное сообщение
        status_msg = await reply_method("🧠 Думаю...")
        
        response = await g4f.ChatCompletion.create_async(
            model=provider_config["model_name"],
            messages=context.history,
            provider=provider
        )
        
        # Удаляем временное сообщение
        if status_msg:
            await status_msg.delete()
        
        full_response = response.strip()
        if not full_response:
            full_response = "Пустой ответ от модели"
        
        # Сохраняем в контекст первые 40960 символов
        context.add_message("assistant", full_response[:40960])
        
        # Улучшенное разбиение на части
        max_chunk_size = 4000  # Оставляем запас для мета-информации
        chunks = []
        current_chunk = ""
        
        # Разбиваем по предложениям, если возможно
        sentences = re.split(r'(?<=[.!?])\s+', full_response)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Если текст слишком длинный и не разбился на предложения
        if not chunks:
            chunks = [full_response[i:i+max_chunk_size] for i in range(0, len(full_response), max_chunk_size)]
        
        # Отправляем части с индикацией прогресса
        for i, chunk in enumerate(chunks, 1):
            try:
                if len(chunks) > 1:
                    prefix = f"📝 Часть {i}/{len(chunks)}:\n"
                    chunk = prefix + chunk
                
                # Пытаемся отправить с Markdown
                await reply_method(
                    text=chunk,
                    parse_mode="Markdown",
                    disable_web_page_preview=True
                )
            except Exception as e:
                logger.warning(f"Markdown error, sending as plain text: {e}")
                await reply_method(chunk)
            
            # Увеличиваем задержку между частями
            await asyncio.sleep(1 if is_group else 0.7)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if status_msg:
            await status_msg.delete()
        
        error_msg = f"⚠️ Ошибка: {type(e).__name__}"
        if str(e):
            error_msg += f" - {str(e)}"
        await reply_method(error_msg)


@router.message(Command("start", "старт", "help", "помощь",))
async def cmd_start(message: types.Message):
    help_text = (
        f"🧟‍♂️ Привет! Я Ждёмби-бот {__version__}\n"
        "Команды:\n"
        "/ask [вопрос] - задать вопрос\n"
        "/image [описание] - сгенерировать изображение\n"
        "/clear - очистить историю\n"
        "/model - выбрать модель"
    )
    await message.answer(help_text)

@router.message(Command("clear"))
async def cmd_clear(message: types.Message):
    chat_id = message.chat.id
    if chat_id in storage.contexts:
        del storage.contexts[chat_id]
    reply_method = message.reply if is_group_chat(message) else message.answer
    await reply_method("🗑️ История очищена!")

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

@router.message(Command("ask", "аск", "bot", "бот", "ждёмби", "zhdomby", "ждемби"))
async def cmd_ask(message: types.Message, command: CommandObject):
    question = None

    if message.reply_to_message:
        original_text = message.reply_to_message.text or message.reply_to_message.caption
        if not original_text:
            reply_method = message.reply if is_group_chat(message) else message.answer
            await reply_method("❌ Сообщение, на которое вы ответили, не содержит текста")
            return
        
        original_text = original_text.strip()
        command_args = command.args.strip() if command.args else ''
        
        if command_args:
            question = f"{command_args}: {original_text}"
        else:
            question = original_text
    else:
        question = command.args.strip() if command.args else None

    if not question or len(question) > MAX_QUESTION_LENGTH:
        reply_method = message.reply if is_group_chat(message) else message.answer
        await reply_method(f"❌ Некорректный вопрос (макс. {MAX_QUESTION_LENGTH} символов)")
        return

    await process_question(message, question)

@router.message(Command("image", "img"))
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

async def on_startup():
    logger.info(f"Bot @{(await bot.get_me()).username} started")

if __name__ == "__main__":
    dp.startup.register(on_startup)
    try:
        dp.run_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    finally:
        bot.session.close()