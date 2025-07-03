#0.2.5

import logging
import re
import inspect
from typing import Dict, List

import g4f
from aiogram import Bot, Dispatcher, Router, F, types
from aiogram.filters import Command, CommandObject
from aiogram.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.enums import ChatType
from g4f.Provider import PollinationsAI, Blackbox, Websim

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
        self.history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.history = self.history[-MAX_CONTEXT_LENGTH*2:]

class Storage:
    def __init__(self):
        self.contexts: Dict[int, ChatContext] = {}
        self.models: Dict[int, str] = {}
    
    def get_context(self, chat_id: int) -> ChatContext:
        if chat_id not in self.contexts:
            self.contexts[chat_id] = ChatContext()
        return self.contexts[chat_id]
    
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
    try:
        chat_id = message.chat.id
        is_group = is_group_chat(message)
        
        context = storage.get_context(chat_id)
        model_name = storage.get_model(chat_id)
        context.add_message("user", question)
        
        provider_config = PROVIDERS[model_name]
        provider = PROVIDER_MAP[provider_config["provider"]]
        
        response = await g4f.ChatCompletion.create_async(
            model=provider_config["model_name"],
            messages=context.history,
            provider=provider
        )
        
        response = response[:4096].strip()
        context.add_message("assistant", response)
        reply_method = message.reply if is_group else message.answer
        await reply_method(response or "Пустой ответ от модели")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        error_msg = f"⚠️ Ошибка: {type(e).__name__} - {str(e)}"
        reply_method = message.reply if is_group else message.answer
        await reply_method(error_msg)

@router.message(Command("start"))
async def cmd_start(message: types.Message):
    help_text = (
        "🧟‍♂️ Привет! Я Ждёмби-бот 0.2.5\n"
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
async def cmd_model(message: types.Message):
    builder = ReplyKeyboardBuilder()
    for model in PROVIDERS:
        builder.add(KeyboardButton(text=model))
    builder.adjust(2)
    reply_method = message.reply if is_group_chat(message) else message.answer
    await reply_method("🤖 Выберите модель:", reply_markup=builder.as_markup(resize_keyboard=True))
    await dp.current_state().set_state(Form.model_selection)

@router.message(Command("ask"))
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

@router.message(Command("image"))
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
async def model_selection_handler(message: types.Message):
    if message.text not in PROVIDERS:
        reply_method = message.reply if is_group_chat(message) else message.answer
        return await reply_method("❌ Неизвестная модель!", reply_markup=ReplyKeyboardRemove())
    
    storage.set_model(message.chat.id, message.text)
    reply_method = message.reply if is_group_chat(message) else message.answer
    await reply_method(f"✅ Модель изменена на {message.text}", reply_markup=ReplyKeyboardRemove())
    await dp.current_state().reset_state()

dp.include_router(router)

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