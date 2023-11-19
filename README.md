# Document-Retrieval (Qdrant+LangChain+FastAPI)
This repo is an implementation of [Document Retrieval](https://python.langchain.com/docs/modules/data_connection/) as part of the Assignment of chaabi for SDE-1 role - Placements'23. This Query Engine uses [Qdrant](https://github.com/qdrant/qdrant) for vector database, [LangChain](https://github.com/langchain-ai/langchain) for performing semantic search and [Streamlit](https://streamlit.io/) for user-interface. The DB is hosted on Qdrant Cloud as a cluster with a collection of vectors. I've used [OpenAIEmbeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) for embedding the vectors.

## DocRetrieval in action

![demo](data/chat-demo.gif)

## API
I've used FastAPI for the retreival service which can be ran with curl command as follows:
```
curl -X 'GET' \
  'http://127.0.0.1:8000/api/search?q=Suggest%20me%20some%20hair%20care%20products&num_results=2' \
  -H 'accept: application/json'
```
Response:
```
{
  "result": {
    "answer": " Suggested hair care products include the Daily Use Hair Conditioner - For Men and the Hairfall Control Shampoo.\n",
    "products": [
      {
        "product": "Daily Use Hair Conditioner - For Men",
        "brand": "USTRAA",
        "sale_price": 153.23,
        "rating": 3.8,
        "description": "Using conditioner after shampoo keeps your hair soft and manageable. This daily use conditioner contains Vitamin E which helps to increase blood flow to the scalp and promote hair health and quality. It also has Water Hyssop (Brahmi) which is one of best-known Ayurvedic ingredients that strengthen hair follicles and checks split ends. Due to the presence of wheat gram, it is enriched with Vitamin B which is a great conditioner for hair. It has Evening Primrose which nourishes the scalp and promotes healthy hair growth."
      },
      {
        "product": "Hairfall Control Shampoo",
        "brand": "Aroma Magic ",
        "sale_price": 182.75,
        "rating": 4.1,
        "description": "This Hairfall control shampoo is 100% free of parabens, petrochemicals, phthalates, sulphates, toxic ingredients, artificial colouring and fragrances. Enriched with natural oils and essential oils of clary sage and lavender, it helps balance the pH levels of the scalp and strengthen hair strands. The plant juice of aloe vera and reetha deeply cleans and conditions the hair from the root to the tips. The extracts of thyme, peppermint and natural vitamins increase blood circulation to the scalp, thereby promoting hair growth and shine. Tocopherol in its formulation hydrates hair fibres while restoring natural moisture levels of the scalp, giving you back the control over your luscious hair."
      }
    ]
  }
}
```

## Setup
Install all required modules
```
pip install -r requirements.txt
```
Setup .env file at root directory of this repo having following fields:
```
QDRANT_URL="your-qdrant-url"
QDRANT_API_KEY="your-qdrant-api-key"
COLLECTION_NAME="big-basket-products"
OPENAI_API_KEY="your-openai-api-key"
FILE_PATH="data/bigBasketProducts.csv"
```

### Create Collection
Create collection and Upload embeddings. (Do this only if you want to change the DB)
```
cd backend
python init_collections_products.py
```

### Perform Search
Search for the best match vectors stored in DB.
```
python service.py
```
Now open http://127.0.0.1:8000/docs in your browser and put the value of `q` (question) in query.

![demo](data/fast-api.png)

## Running Streamlit App
You can also use the web interface for running the application.
```
cd frontend
streamlit run app.py
```

### Setting Search Parameter (K)
You can set the search parameter `K` that controls the maximum number of products the retrieval chain can fetch.

![demo](data/chat-parameters.gif)

### Sorting Results
After getting the products, you can sort them on the basis of Sale Price and Rating.

![demo](data/chat-sort.gif)

## Colab Notebook
Complete Code (dev) for performing DocRetrieval over bigBasketProducts.csv could be found in [Colab Notebook](https://colab.research.google.com/github/AbhishekPardhi/Document-Retrieval/blob/main/test.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbhishekPardhi/Document-Retrieval/blob/main/test.ipynb)

## Limitations and Future Scope
Currently the chain of chats are independent of each other. This can be easily changed by using previous responses as a context while retrieving results for the current question. But this will also increase the number of tokens that will be used for performing search operation, and hence the cost of each search.