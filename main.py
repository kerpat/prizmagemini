import os
import json
import logging
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import httpx

# --- Настройка ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI(title="Универсальный API Шлюз для Gemini и Telegram")

# --- Конфигурация из .env ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    INTERNAL_SECRET = os.getenv("INTERNAL_SECRET")

    if not all([GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, INTERNAL_SECRET]):
        raise ValueError("Не установлены все необходимые переменные: GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, INTERNAL_SECRET")
    
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("API Gemini и токен бота успешно настроены на сервере.")
except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при настройке: {e}")

# --- Модели данных для FastAPI ---
class RecognizeDocsRequest(BaseModel):
    images_base64: list[str]
    country: str

class NotifyRequest(BaseModel):
    user_id: int
    text: str
    
class ParseDealRequest(BaseModel):
    description: str

class BuyoutPlanRequest(BaseModel):
    deal_description: str

# ==============================================================================
# ВСЯ ЛОГИКА GEMINI С ТВОИМИ ОРИГИНАЛЬНЫМИ, ПРАВИЛЬНЫМИ ПРОМПТАМИ
# ==============================================================================

def recognize_documents_with_gemini(images: list, country: str) -> dict | None:
    """Распознает документы с использованием твоих оригинальных промптов."""
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        # ИСПРАВЛЕНО: Используем существующую модель
        model = genai.GenerativeModel('gemini-2.5-flash', safety_settings=safety_settings)

        if country == 'РФ':
            prompt = '''
            Проанализируй эти три изображения: основной разворот паспорта РФ, страница с пропиской и селфи с паспортом.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "Фамилия", "Имя", "Отчество", "Дата рождения", "Серия и номер паспорта", "Кем выдан", "Дата выдачи", "Адрес регистрации".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            '''
        else:
            prompt = '''
            Проанализируй эти четыре изображения: паспорт иностранного гражданина, регистрация в РФ, патент и селфи с паспортом.
            Извлеки все данные и верни их в виде ОДНОГО плоского JSON объекта.
            Ключи: "ФИО", "Гражданство", "Дата рождения", "Номер паспорта", "Адрес регистрации в РФ", "Номер патента".
            Если поле не найдено, значение должно быть пустой строкой.
            Ответ должен быть только чистым JSON.
            '''
        
        response = model.generate_content([prompt] + images, request_options={"timeout": 120})
        response.resolve()
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        logger.info(f"Ответ от Gemini (документы): {cleaned_text}")
        return json.loads(cleaned_text)

    except Exception as e:
        logger.error(f"Ошибка Gemini (документы): {e}", exc_info=True)
        return None

# ... (остальные функции Gemini остаются без изменений) ...

# ==============================================================================
# API ЭНДПОИНТЫ
# ==============================================================================

@app.post("/recognize-documents")
async def api_recognize_documents(request: RecognizeDocsRequest, http_request: Request):
    # Проверка секрета
    if http_request.headers.get('x-internal-secret') != INTERNAL_SECRET:
        logger.warning(f"Попытка неавторизованного доступа к /recognize-documents с IP: {http_request.client.host}")
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info(f"Входящий запрос /recognize-documents для страны: {request.country}")
    images = [Image.open(BytesIO(base64.b64decode(b64))) for b64 in request.images_base64]
    data = recognize_documents_with_gemini(images, request.country)
    if data: return data
    raise HTTPException(status_code=500, detail="Ошибка распознавания документов на стороне Gemini.")

@app.post("/notify")
async def notify_user(request: NotifyRequest, http_request: Request):
    logger.info(f"Входящий запрос /notify для user_id: {request.user_id}")
    if http_request.headers.get('x-internal-secret') != INTERNAL_SECRET:
        logger.warning(f"Попытка неавторизованного доступа к /notify с IP: {http_request.client.host}")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(telegram_api_url, json={"chat_id": request.user_id, "text": request.text, "parse_mode": "Markdown"})
            response.raise_for_status()
            logger.info(f"Уведомление успешно отправлено пользователю {request.user_id}")
            return {"success": True}
        except httpx.HTTPStatusError as e:
            error_info = e.response.json()
            logger.error(f"Ошибка от Telegram API для user {request.user_id}: {error_info}")
            raise HTTPException(status_code=400, detail=f"Telegram API error: {error_info.get('description')}")

@app.get("/")
async def root():
    return {"status": "Универсальный API-шлюз работает"}
