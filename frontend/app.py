import sys
sys.path.append("..")

import os
import json
import time
import pandas as pd
import streamlit as st
from streamlit_extras.mention import mention
from dotenv import load_dotenv

import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv('../.env')

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

K=4

# neural_searcher = NeuralSearcher(collection_name=COLLECTION_NAME)

@st.cache_resource
def Retriever():
    global K
    client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    ) 

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    llm = OpenAI(openai_api_key=OPENAI_API_KEY)

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'k':K}),
        return_source_documents=True,
        reduce_k_below_max_tokens=True
    )

    return qa

@st.cache_data
def search(_retriever, user_question):
    res = _retriever({"question": user_question})
    return res

def display_data(res):
    srcs = [json.loads(row.page_content) for row in res['source_documents']]

    df = pd.DataFrame(srcs)
    # df.set_index('product', inplace=True)

    df1 = df[['product','brand', 'sale_price', 'rating', 'description']]

    # Remove duplicates
    df1 = df1.drop_duplicates()

    st.dataframe(
        df1,
        column_config={
            "product": st.column_config.Column(
                "Product Name",
                width="medium"
            ),
            "brand": st.column_config.Column(
                "Brand",
                width="medium"
            ),
            "sale_price": st.column_config.NumberColumn(
                "Sale Price",
                help="The price of the product in USD",
                min_value=0,
                max_value=1000,
                format="₹%f",
            ),
            "rating": st.column_config.NumberColumn(
                "Rating",
                help="Rating of the product",
                format="%f ⭐",
            ),
            "description": "Description",
        },
        hide_index=True,
    )

def main():
    global K
    st.set_page_config(
        page_title="BigBasket Products",
        page_icon="🧺",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:

        st.markdown(
            """This is an app interface for [Document-Retrieval](https://python.langchain.com/docs/modules/data_connection/) on :green[BigBasketProducts.csv] using :blue[Streamlit].
            This Query Engine uses :blue[Qdrant] for vector DB & :blue[LangChain] for performing semantic search.  
            [![VectorDB: Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-blue)](https://github.com/qdrant/qdrant)
            [![Embeddings: OpenAI](https://img.shields.io/badge/Embeddings-OpenAI-blue)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
            [![Retriever: LanghChain](https://img.shields.io/badge/Retriever-LanghChain-blue)](https://github.com/langchain-ai/langchain)
            [![UI: streamlit](https://img.shields.io/badge/UI-streamlit-blue)](https://streamlit.io/)
            """
        )

        mention(
            label="Document-Retrieval",
            icon="github",  # GitHub is also featured!
            url="https://github.com/AbhishekPardhi/Document-Retrieval",
        )

        # st.write('---')

        st.subheader('Parameters')
        K = st.slider('K', 1, 10, 4, help='Sets max number of products  \nthat can be retrieved')
        


    st.header('BigBasket Products',divider=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    retriever = Retriever()

    with st.chat_message("assistant"):
        st.write("Hello 👋\n\n I am here to help you choose the product that you wanna buy!")

    if prompt:=st.chat_input("Say something"):

        with st.chat_message("user"):
            st.markdown(f'{prompt}')

        # Recommend me some product using which I can make bread

        res = search(retriever, prompt)
        answer = res['answer']
        # answer = prompt

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Simulate stream of response with milliseconds delay
            for chunk in answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # Dsiplay product details
            display_data(res)



if __name__ == "__main__":
    main()