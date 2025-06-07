import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import os 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import ollama

# Dataset from: https://www.kaggle.com/datasets/jensenbaxter/10dataset-text-document-classification?resource=download
# Articles directory only features business sub-directory from Kaggle dataset

"""
References and useful articles: 
https://dev.to/mohsin_rashid_13537f11a91/rag-with-ollama-1049
https://docs.trychroma.com/docs/run-chroma/persistent-client
https://medium.com/@kbdhunga/an-overview-of-chromadb-the-vector-database-206437541bdd
https://medium.com/@laurentkubaski/ollama-chat-endpoint-parameters-21a7ac1252e5
https://www.youtube.com/watch?v=g8fMRuGR5z0
Understanding Naive RAG: https://www.youtube.com/watch?v=ea2W8IogX80
"""

embedding_model = OllamaEmbeddingFunction(
    url='http://127.0.0.1:11434',
    model_name="nomic-embed-text"
)

client = chromadb.PersistentClient(path="chroma_persistent")
collection_name = "articles_collection"
collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_model) 

dir_list = os.listdir("articles")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

def create_and_embed_chunks():
    texts = []
    for article in dir_list:
        with open(os.path.join("articles",article)) as f: 
            text = f.read()
            texts.append(text_splitter.split_text(text))
    for t in texts:
        docs = text_splitter.create_documents(t)
        for doc in docs: 
            uuid_name = uuid.uuid1()
            collection.add(ids=[str(uuid_name)], documents=doc.page_content) 
query = input(">>>")

closest = collection.query(query_texts=[query], n_results = 5)

response = ollama.chat(
    model = "llama3.1",
    messages=[{
        "role": "system",
        "content": closest["documents"][0][0]
    },
    {
        "role": "system",
        "content": closest["documents"][0][1]
    },
    {
        "role": "system",
        "content": closest["documents"][0][2]
    },
    {
        "role": "system",
        "content": closest["documents"][0][3]
    },
    {
        "role": "system",
        "content": closest["documents"][0][4]
    },
    {
        "role": "user",
        "content": query
    }
    ]
)

print(response["message"]["content"])
