import time
import json
import pandas as pd
from typing import List

from qdrant_client import QdrantClient
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI

from backend.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY


class NeuralSearcher:

    def __init__(self, collection_name: str):
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=OpenAIEmbeddings(),
        )
        self.llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    def search(self, question: str, num_results: int, filter_: dict = None) -> dict:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={'k':num_results}),
            return_source_documents=True,
            reduce_k_below_max_tokens=True
        )
        start_time = time.time()
        res = qa({"question": question})
        print(f"Search took {time.time() - start_time} seconds")

        ret = {}
        ret['answer'] = res['answer']

        srcs = [json.loads(row.page_content) for row in res['source_documents']]

        df = pd.DataFrame(srcs)
        df = df.fillna('null')
        # df.set_index('product', inplace=True)

        df1 = df[['product','brand', 'sale_price', 'rating', 'description']]

        # Remove duplicates
        df1 = df1.drop_duplicates()

        ret['products'] = df1.to_dict(orient='records')
        return ret