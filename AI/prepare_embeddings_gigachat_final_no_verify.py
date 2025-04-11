from langchain_gigachat import GigaChatEmbeddings

def get_embeddings():
    # Создание векторных представлений с использованием Gigachat
    try:
        embeddings = GigaChatEmbeddings(verify=True)  # Включение проверки сертификатов
        return embeddings
    except Exception as e:
        print(f"Error initializing GigaChatEmbeddings: {e}")
        return None
