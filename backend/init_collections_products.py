import sys
sys.path.append("..")
import os.path
from tqdm import tqdm

import qdrant_client
from qdrant_client import models
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader

from backend.config import DATA_DIR, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, TEXT_FIELD_NAME, EMBEDDINGS_MODEL

def upload_embeddings():
    # Create client
    client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )

    client.set_model(EMBEDDINGS_MODEL)

    # Load data
    payload_path = os.path.join(DATA_DIR, 'bigBasketProducts.csv')

    loader = CSVLoader(file_path=payload_path, source_column=('rating'), csv_args={'delimiter': ','}, autodetect_encoding=True)
    documents = loader.load()
    documents = documents[:100]

    docs = [doc.page_content for doc in documents]
    payload = [doc.payload for doc in documents]

    # Create collection
    vectors_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE,
    )

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
        # Quantization is optional, but it can significantly reduce the memory usage
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )
    )

    # Create a payload index for text field.
    # This index enables text search by the TEXT_FIELD_NAME field.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=TEXT_FIELD_NAME,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )
    )

    client.add(
        collection_name=COLLECTION_NAME,
        documents=docs,
        metadata=payload,
        ids=tqdm(range(len(payload))),
        parallel=0,
    )


if __name__ == '__main__':
    upload_embeddings()