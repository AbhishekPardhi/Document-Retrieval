import time
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

    def search(self, question: str, filter_: dict = None) -> List[dict]:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={'k':2}),
            return_source_documents=True
        )
        start_time = time.time()
        res = qa({"question": question})
        print(f"Search took {time.time() - start_time} seconds")

        ret = {}
        ret['answer'] = res['answer']

        srcs = [row.page_content for row in res['source_documents']]

        dicts = []
        for src in srcs:
            key_value = src.split("\n")
            dict = {}
            for v in key_value:
                aux = v.split(": ")
                dict[aux[0]] = aux[1]
            dicts.append(dict)
        ret['source_documents'] = dicts

        return [ret]