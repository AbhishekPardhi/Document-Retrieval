# import streamlit as st
# import pandas as pd

# st.title('Chaabi App')


# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache_data
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text('Done! (using st.cache_data)')

import sys
sys.path.append("..")

import time
import random
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from backend.neural_searcher import NeuralSearcher
# from backend.config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY

import qdrant_client
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv('../.env')

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# neural_searcher = NeuralSearcher(collection_name=COLLECTION_NAME)

@st.cache_resource
def Retriever():
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

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    return qa

@st.cache_data
def search(_retriever, user_question):
    return _retriever.run(user_question)

def main():
    st.set_page_config(
        page_title="BigBasket Products",
        page_icon="🧺",
        layout="centered",
        initial_sidebar_state="expanded",
    )

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
        # st.write(neural_searcher.search(text=user_question))

        with st.chat_message("user"):
            st.markdown(f'{prompt}')
        # for hit in neural_searcher.search(text=user_question):
        #     st.write(hit)
        #     st.write('---')

        # Recommend me some product using which I can make bread

        answer = search(retriever, prompt)
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



if __name__ == "__main__":
    main()