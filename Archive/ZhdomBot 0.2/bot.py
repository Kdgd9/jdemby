import logging
import traceback  # Для логирования ошибок
from typing import Dict, List

import g4f
from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command, CommandObject
from aiogram import F
from aiogram.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.utils.token import TokenValidationError
import asyncio
from asyncio import Lock
from datetime import datetime  # Импорт для работы с временем
import asyncio  # Для асинхронных операций
import re  # Для регулярных выражений
import inspect  # Для проверки атрибутов
from aiogram.enums import ChatType  # Для проверки типа чат



# Конфигурация
from config import (
    TELEGRAM_TOKEN,
    PROVIDERS,
    DEFAULT_MODEL,
    MAX_CONTEXT_LENGTH,
    MAX_QUESTION_LENGTH,
    MAX_IMAGE_PROMPT_LENGTH,
    MAX_CONCURRENT_IMAGE_REQUESTS

    # Лимиты
  # Максимум одновременных запросов
)

# Импорт провайдеров
from g4f.Provider import (
    PollinationsAI,
    Blackbox,
    # Добавьте другие провайдеры по необходимости
)

# Маппинг провайдеров
PROVIDER_MAP = {
    "PollinationsAI": PollinationsAI,
    "Blackbox": Blackbox,
    # Добавьте другие провайдеры
}

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Form(StatesGroup):
    model_selection = State()

class ChatContext:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._trim_history()
    
    def _trim_history(self):
        self.history = self.history[-MAX_CONTEXT_LENGTH*2:]

class Storage:
    def __init__(self):
        self.private_contexts: Dict[int, ChatContext] = {}
        self.group_contexts: Dict[int, ChatContext] = {}
        self.private_models: Dict[int, str] = {}
        self.group_models: Dict[int, str] = {}
    
    def get_context(self, chat_id: int, is_group: bool) -> ChatContext:
        storage_dict = self.group_contexts if is_group else self.private_contexts
        if chat_id not in storage_dict:
            storage_dict[chat_id] = ChatContext()
        return storage_dict[chat_id]
    
    def get_model(self, chat_id: int, is_group: bool) -> str:
        model_dict = self.group_models if is_group else self.private_models
        return model_dict.get(chat_id, DEFAULT_MODEL)
    
    def set_model(self, chat_id: int, model: str, is_group: bool):
        model_dict = self.group_models if is_group else self.private_models
        model_dict[chat_id] = model

# Инициализация бота
try:
    bot = Bot(token=TELEGRAM_TOKEN)
    bot_username = None  # Будет установлено при запуске
except TokenValidationError as e:
    logger.error(f"Invalid token: {e}")
    exit(1)

dp = Dispatcher()
storage = Storage()

# Фильтры для роутеров
private_router = Router()
private_router.message.filter(F.chat.type == ChatType.PRIVATE)  # Для личных чатов

group_router = Router()
group_router.message.filter(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))  # Для групп

# Общие утилиты
async def process_question(message: types.Message, chat_id: int, is_group: bool, question: str):
    try:
        context = storage.get_context(chat_id, is_group)
        model_name = storage.get_model(chat_id, is_group)
        
        context.add_message("user", question)
        
        provider_config = PROVIDERS[model_name]
        provider = PROVIDER_MAP[provider_config["provider"]]
        
        response = await g4f.ChatCompletion.create_async(
            model=provider_config["model_name"],
            messages=context.history,
            provider=provider
        )
        
        response = response[:4096].strip()
        if not response:
            raise ValueError("Пустой ответ от модели")
        
        context.add_message("assistant", response)
        
        reply_method = message.reply if is_group else message.answer
        await reply_method(response)
        
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        error_msg = (
            "⚠️ Произошла ошибка при обработке запроса:\n"
            f"• Тип: {type(e).__name__}\n"
            f"• Описание: {str(e)}"
        )
        reply_method = message.reply if is_group else message.answer
        await reply_method(error_msg)

# Обработчики для личных чатов
@private_router.message(Command("start"))
async def private_start(message: types.Message):
    help_text = (
        "🧟‍♂️ Привет! Я Ждёмби-бот\n 0.2\n"
        "Доступные команды:\n"
        "/ask [вопрос] - задать вопрос\n"
        "/image [описание] - сгенерировать изображение\n"
        "/clear - очистить историю\n"
        "/model - выбрать модель нейросети"
    )
    await message.answer(help_text)

@private_router.message(Command("clear"))
async def private_clear(message: types.Message):
    user_id = message.from_user.id
    if user_id in storage.private_contexts:
        del storage.private_contexts[user_id]
    await message.answer("🗑️ История переписки очищена!")

@private_router.message(Command("model"))
async def private_model(message: types.Message):
    builder = ReplyKeyboardBuilder()
    for model in PROVIDERS.keys():
        builder.add(KeyboardButton(text=model))
    builder.adjust(2)
    
    await message.answer(
        "🤖 Выберите модель:",
        reply_markup=builder.as_markup(
            resize_keyboard=True,
            one_time_keyboard=True
        )
    )
    await dp.current_state().set_state(Form.model_selection)

@private_router.message(Command("ask"))
async def private_ask(message: types.Message, command: CommandObject):
    user_id = message.from_user.id
    question = None

    if message.reply_to_message:
        original_text = message.reply_to_message.text or message.reply_to_message.caption
        if not original_text:
            await message.answer("❌ Сообщение, на которое вы ответили, не содержит текста")
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
        await message.answer(f"❌ Некорректный вопрос (макс. {MAX_QUESTION_LENGTH} символов)")
        return

    await process_question(message, user_id, False, question)

@private_router.message()
async def private_default(message: types.Message):
    await message.answer("ℹ️ Используйте команды:\n/ask - задать вопрос\n/clear - очистить историю\n/model - выбрать модель")

# Обработчики для групповых чатов
@group_router.message(Command("start"))
async def group_start(message: types.Message):
    help_text = (
        "🧟‍♂️ Я Ждёмби-бот\n 0.2\n"
        "Доступные команды для групп:\n"
        "/ask [вопрос] - задать вопрос\n"
        "/image [описание] - сгенерировать изображение\n"
        "/clear - очистить историю группы"
    )
    await message.reply(help_text)

@group_router.message(Command("clear"))
async def group_clear(message: types.Message):
    chat_id = message.chat.id
    if chat_id in storage.group_contexts:
        del storage.group_contexts[chat_id]
    await message.reply("🗑️ История переписки в группе очищена!")

@group_router.message(Command("ask"))
async def group_ask(message: types.Message, command: CommandObject):
    chat_id = message.chat.id
    question = None

    if message.reply_to_message:
        original_text = message.reply_to_message.text or message.reply_to_message.caption
        if not original_text:
            await message.reply("❌ Сообщение, на которое вы ответили, не содержит текста")
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
        await message.reply(f"❌ Некорректный вопрос (макс. {MAX_QUESTION_LENGTH} символов)")
        return

    await process_question(message, chat_id, True, question)

# Общие обработчики для всех типов чатов

@dp.message(Command("image"))
async def cmd_image(message: types.Message, command: CommandObject):
    try:
        prompt = None
        is_group = message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
        
        if message.reply_to_message:
            prompt = message.reply_to_message.text or message.reply_to_message.caption
        elif command.args:
            prompt = command.args.strip()
        
        if not prompt or len(prompt) > MAX_IMAGE_PROMPT_LENGTH:
            reply_method = message.reply if is_group else message.answer
            await reply_method(f"❌ Некорректный промпт (макс. {MAX_IMAGE_PROMPT_LENGTH} символов)")
            return
        
        status_msg = await (message.reply if is_group else message.answer)("🎨 Генерирую изображение...")
        
        response = await g4f.ChatCompletion.create_async(
            model="flux",
            messages=[{"role": "user", "content": f"Generate image: {prompt}"}],
            provider=g4f.Provider.ARTA
        )
        
        # Ищем URL изображений с разными расширениями
        image_urls = re.findall(r'https?://[^\s\)]+\.(?:png|jpe?g|webp)', response, re.IGNORECASE)
        
        if image_urls:
            # Выбираем последний URL (часто первый - preview, второй - полное изображение)
            image_url = image_urls[-1].split(')')[0]
            reply_method = message.reply_photo if is_group else message.answer_photo
            await reply_method(image_url, caption=f"🖼 Результат для: {prompt}")
        else:
            error_message = "❌ Не удалось извлечь ссылку на изображение"
            if response:
                logger.error(f"Ответ провайдера: {response}")
                error_message += f"\nОтвет сервера: {response[:200]}..."
            reply_method = message.reply if is_group else message.answer
            await reply_method(error_message)
        
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Image error: {traceback.format_exc()}")
        error_msg = f"⚠️ Ошибка генерации: {str(e)}"
        
        if hasattr(e, 'response'):
            try:
                if inspect.isawaitable(e.response.text):
                    provider_response = await e.response.text()
                else:
                    provider_response = e.response.text()
                
                error_msg += f"\nОтвет сервера: {provider_response[:200]}..."
            except Exception as ex:
                logger.error(f"Ошибка при чтении ответа: {ex}")
        
        reply_method = message.reply if is_group else message.answer
        await reply_method(error_msg)


# Регистрация роутеров
dp.include_router(private_router)
dp.include_router(group_router)

# Обработчик состояния выбора модели
@dp.message(Form.model_selection)
async def model_selection_handler(message: types.Message):
    selected_model = message.text
    if selected_model not in PROVIDERS:
        reply_method = message.reply if message.chat.type != ChatType.PRIVATE else message.answer
        await reply_method("❌ Неизвестная модель!", reply_markup=ReplyKeyboardRemove())
        return
    
    is_group = message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
    storage.set_model(message.chat.id, selected_model, is_group)
    
    reply_method = message.reply if is_group else message.answer
    await reply_method(
        f"✅ Модель изменена на {selected_model}",
        reply_markup=ReplyKeyboardRemove()
    )
    await dp.current_state().reset_state()

async def on_startup():
    global bot_username
    bot_username = (await bot.get_me()).username
    logger.info(f"Bot @{bot_username} started")

if __name__ == "__main__":
    dp.startup.register(on_startup)
    try:
        logger.info("Starting bot...")
        dp.run_polling(bot)
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    finally:
        bot.session.close()