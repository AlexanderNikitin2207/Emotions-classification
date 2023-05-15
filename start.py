import os
import requests
import telebot
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from dotenv import load_dotenv
import torch

load_dotenv()

bot_token = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(bot_token)

def image_processing(image_path, path_type):
    image = Image.open(requests.get(image_path, stream=True).raw) if path_type else Image.open(image_path)

    repo_name = "Mauregato/vit-base-patch16-224-finetuned-on-all-affectnet_short"
    image_processor = AutoImageProcessor.from_pretrained(repo_name)
    model = AutoModelForImageClassification.from_pretrained(repo_name)

    encoding = image_processor(image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    emotion = model.config.id2label[predicted_class_idx]

    return emotion

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'Привет! Чтобы начать работать, отправь одно или несколько изображений или ссылки на изображения')

@bot.message_handler(content_types=['text'])
def handle_text(message):
    try:
        text = message.text
        for line in text.split('\n'):
            if line.startswith('http') and (line.endswith('.png') or line.endswith('.jpg') or line.endswith('.jpeg') or line.endswith('.webp')):
                result = line + '\n' + image_processing(line, 1)
                bot.reply_to(message, result)
            else:
                bot.reply_to(message, f"Неверный URL: {line}")
    except Exception as e:
        bot.reply_to(message.chat.id, f"Ошибка: {str(e)}")

@bot.message_handler(content_types=['photo'])
def image_handler(message):
    file_id = message.photo[-1].file_id
    file = bot.download_file(bot.get_file(file_id).file_path)
    with open('image.jpg', 'wb') as f:
        f.write(file)
    result = image_processing('image.jpg', 0)
    bot.reply_to(message, result)

@bot.message_handler(content_types=['document'])
def multi_image_handler(message):
    if message.document.mime_type.startswith('image'):
        file_info = bot.get_file(message.document.file_id)
        file = bot.download_file(file_info.file_path)
        with open('image.jpg', 'wb') as f:
            f.write(file)
        result = image_processing('image.jpg', 0)
        bot.reply_to(message, result)
    else:
        bot.reply_to(message, 'Простите, бот умеет обрабатывать только изображения.')

# Webhook Configuration
WEBHOOK_URL_BASE = f"https://angdl.ru"
WEBHOOK_URL_PATH = "/telegram-bot"

# Set webhook
bot.remove_webhook()
bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)

# Start Flask server
from flask import Flask, request, abort
app = Flask(__name__)

@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        abort(403)

@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()  # Start the server.