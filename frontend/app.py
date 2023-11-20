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
# from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv('../.env')

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

K=2

#############################################################################################################
#############################################################################################################

@st.cache_resource(show_spinner=False)
def PROMPT():
    prompt_template = '''
    About: You are a Product Recommendation Agent who gets his context from the retrieved descriptions of the products that matches best with the User's query. User is a human who, as a customer, wants to buy a product from this application.
    Answer my questions based on your knowledge and our older conversation.

    Given below is the summary of conversation between you (AI) and the human:
    Context: {chat_history}

    Now use this summary of previous conversations and the retrieved descriptions of products to answer the following question:
    Question: {question}

    Note:
    1. After answering the question, do remember what you answered and add it to the summary of conversation. While summarizing, mention about what is written in About section only once.
    2. If you do not know the answer to a question, just say "I don't know" in a polite manner.
    '''

    return PromptTemplate(
        template=prompt_template, input_variables=["chat_history", "question"]
    )

@st.cache_resource(show_spinner=False)
def load_model():
    return ChatOpenAI(model_name='gpt-3.5-turbo')

llm = load_model()

def memory():
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    return st.session_state.memory


# @st.cache_resource(show_spinner=False)
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

    retriever = vector_store.as_retriever(search_kwargs={'k':K})

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory(),
        return_source_documents=True,
        reduce_k_below_max_tokens=True
    )

    return chain

@st.cache_data(show_spinner=False)
def search(_retriever, user_question):
    prompt = PROMPT().format(question=user_question, chat_history=memory().load_memory_variables({})['chat_history'][0].content)
    # res = _retriever({"question": user_question})
    st.session_state.last_prompt = prompt
    res = _retriever({"question": prompt})
    return res

#############################################################################################################
#############################################################################################################

def init():
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
        K = st.slider('K', 1, 10, K, help='Sets max number of products  \nthat can be retrieved')

        st.write('---')

        st.subheader('Debugging Tools')

        col1, col2 = st.columns([1,1])
        if 'debugging_output' not in st.session_state:
            st.session_state.debugging_output = ''
        with col1:
            if st.button('Examine Memory'):
                mem = memory().load_memory_variables({})['chat_history'][0].content
                st.session_state.debugging_output = mem if mem!='' else 'Memory is empty'
        with col2:
            if st.button('Show Prompt'):
                if 'last_prompt' not in st.session_state:
                    st.session_state.last_prompt = 'No prompt generated yet'
                st.session_state.debugging_output = st.session_state.last_prompt
        st.write(st.session_state.debugging_output)

        
    st.header('BigBasket Products',divider=True)

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

    init()

    # Initialize chat history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello 👋\n\n I am here to help you choose the product that you wanna buy!"}
        ]
    
    # Display chat messages from history on app rerun
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    retriever = Retriever()

    # with st.chat_message("assistant"):
    #     st.write("Hello 👋\n\n I am here to help you choose the product that you wanna buy!")

    if prompt:=st.chat_input("Say something"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = search(retriever, prompt)
                answer = res['answer']
                # st.write(response.response)
                message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(message) # Add response to message history

                # Display assistant response in chat message container
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

        # with st.chat_message("user"):
        #     st.markdown(f'{prompt}')

        # # Recommend me some product using which I can make bread

        # res = search(retriever, prompt)
        # answer = res['answer']
        # # answer = prompt

        # # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        #     message_placeholder = st.empty()
        #     full_response = ""

        #     # Simulate stream of response with milliseconds delay
        #     for chunk in answer.split():
        #         full_response += chunk + " "
        #         time.sleep(0.05)
        #         # Add a blinking cursor to simulate typing
        #         message_placeholder.markdown(full_response + "▌")
        #     message_placeholder.markdown(full_response)

        #     # Dsiplay product details
        #     display_data(res)



if __name__ == "__main__":
    main()