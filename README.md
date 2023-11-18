# Document-Retrieval (Qdrant+LangChain+FastAPI)
This repo is an implementation of [Document Retrieval](https://python.langchain.com/docs/modules/data_connection/) as part of the Assignment of chaabi for SDE-1 role - Placements'23. This Query Engine uses [Qdrant](https://github.com/qdrant/qdrant) for vector database, [LangChain](https://github.com/langchain-ai/langchain) for performing semantic search and [Streamlit](https://streamlit.io/) for user-interface. The DB is hosted on Qdrant Cloud as a cluster with a collection of vectors. I've used [OpenAIEmbeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) for embedding the vectors.

## DocRetrieval in action

![demo](data/chat-demo.gif)

## API
I've used FastAPI for the retreival service which can be ran with curl command as follows:
```
curl -X 'GET' \
  'http://127.0.0.1:8000/api/search?q=I%20want%20to%20buy%20shampoo' \
  -H 'accept: application/json'
```
Response:
```
{
  "result": [
    {
      "answer": " You can buy the Biotin & Collagen Volumizing Hair Shampoo + Biotin & Collagen Hair Conditioner from the brand StBotanica.\n",
      "source_documents": [
        {
          "index": "65",
          "product": "Aqua Halo Rejuvenating Conditioner",
          "category": "Beauty & Hygiene",
          "sub_category": "Hair Care",
          "brand": "Azafran",
          "sale_price": "168.75",
          "market_price": "225",
          "type": "Shampoo & Conditioner",
          "rating": "1",
          "description": "This Aqua Halo Rejuvenating Conditioner is an organic conditioner, which is specially formulated with moisturizing actives to add vibrancy and vitality to your hair. This natural hair conditioner provides long-lasting conditioning and colour protection. It controls unruly, dry and brittle hair and it is formulated for all hair types."
        },
        {
          "index": "9",
          "product": "Biotin & Collagen Volumizing Hair Shampoo + Biotin & Collagen Hair Conditioner",
          "category": "Beauty & Hygiene",
          "sub_category": "Hair Care",
          "brand": "StBotanica",
          "sale_price": "1098",
          "market_price": "1098",
          "type": "Shampoo & Conditioner",
          "rating": "3.5",
          "description": "An exclusive blend with Vitamin B7 Biotin, Hydrolyzed collagen, Oat Extract along with premium & organic cold-pressed ingredients helps to infuse nutrients into every strand and creates the appearance of thicker, fuller healthier looking hair. This powerful formula helps volumize even the skinniest strands into fuller and more abundant looking locks. It is safe for color-treated hair and safe for all hair types. St Botanica Biotin & Collagen Hair Conditioner has been specially formulated to repair dry & damaged hair for full, thick, voluminous, shiny & healthy looking hair! The amazing hair conditioner ingredients include Biotin, Hydrolyzed Collagen, Pro-Vitamin B5, Vitamin E, & Hydrolyzed Silk Proteins for glistening looking hair. Biotin and Collagen, infused with most efficacious natural extracts not only promotes healthy hair growth but also prevents hair dryness, increases the elasticity of the hair cortex, thereby strengthening hair, minimizing hair breakage and helping hair grow longer, healthier and thicker. PLUMP IT UP"
        }
      ]
    }
  ]
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