import sys
sys.path.append("..")
import os.path
from tqdm import tqdm

import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader

from backend.config import DATA_DIR, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

def upload_embeddings():
    # Create client
    client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )

    # create collection
    vectors_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE,
    )

    # Load data
    file_path = os.path.join(DATA_DIR, 'bigBasketProducts.csv')

    loader = CSVLoader(file_path=file_path, source_column=('rating'), csv_args={'delimiter': ','}, autodetect_encoding=True)
    documents = loader.load()
    documents = documents[:100]

    docs = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )

    # create vector store
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    # add documents to vector store
    vector_store.add_texts(
        texts=docs,
        metadatas=metadatas,
        ids=[str(i) for i in tqdm(range(len(docs)))],
    )


if __name__ == '__main__':
    upload_embeddings()