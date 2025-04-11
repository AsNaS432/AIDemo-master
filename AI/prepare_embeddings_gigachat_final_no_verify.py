from langchain_gigachat import GigaChatEmbeddings

def get_embeddings():
    # Создание векторных представлений с использованием Gigachat
    embeddings = GigaChatEmbeddings(verify=False)  # Отключение проверки сертификатов
    return embeddings
