import getpass
import os

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat import GigaChatEmbeddings
from AI.create_data import create_data_chroma_db
from AI.search_data import search_data_chroma_db
from AI.mood_analyzer import check_mood

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# модель GigaChat
def get_giga_chat_llm():
    # инициализация GigaChat
    if "GIGACHAT_CREDENTIALS" not in os.environ:
        os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API:NmNiYjRkYjYtM2RmYy00MDRiLTk0ZjktMjRmMmUwNWM3NjFmOmE1N2Q2ODlhLWRhNmYtNDZhZS04YWUwLTk4Nzg1MWQ2Y2VkNQ== ")

    try:
        giga_chat = GigaChat(verify_ssl_certs=False)  # Отключение проверки сертификатов для тестирования
        return giga_chat
    except Exception as e:
        print(f"Error initializing GigaChat: {e}")
        return None

# создаём объекты моделей
embeddings = GigaChatEmbeddings(verify=False)    # модель для векторизации
llm = get_giga_chat_llm()               # генеративный ИИ

@app.route('/api/v1/create', methods=['POST'])
def api_create():
    if embeddings is None:
        return jsonify({"error": "Embeddings initialization failed"}), 500

    create_data_chroma_db(embeddings,
                          "/Users/Sasha/Desktop/AIDemo/info.txt",
                          "./market_chroma_db")
    create_response = {"result": "Success!"}

    return jsonify(create_response)

@app.route('/api/v1/search', methods=['POST'])
def api_search():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    # Получаем текст из запроса
    question = data['text']
    answer = search_data_chroma_db(llm, embeddings, question, "./market_chroma_db")
    answer_response = {"answer": answer}

    return jsonify(answer_response)

@app.route('/api/v1/mood', methods=['POST'])
def api_mood():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    # Получаем текст из запроса
    question = data['text']

    answer = check_mood(llm, embeddings, question)

    answer_response = {"answer": answer}

    return jsonify(answer_response)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    
    if 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    # Basic response logic
    message = data['message'].lower()
    if "группа" in message:
        response_message = "Наша группа - это группа поддержки!"
    elif "привет" in message:
        response_message = "Привет! Как я могу помочь?"
    else:
        response_message = f"Вы сказали: {message}"
    
    return jsonify({'reply': response_message})

if __name__ == '__main__':
    app.run(debug=True, port=5000)