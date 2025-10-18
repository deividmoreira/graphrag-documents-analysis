# Document processing routines and embedding generation helpers

# Efficient vector similarity search
import faiss

# Numerical helper library
import numpy as np

# Streamlit secrets storage
import streamlit as st

# OpenAI client for API calls
from openai import OpenAI

# Recursive text splitter to chunk long documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Generates embeddings and completions using the OpenAI API
class OpenAIEmbedding:

    # Initialize the client with the API key
    def __init__(self, api_key):

        # Instantiate the OpenAI client
       self.client = OpenAI(api_key = api_key)

    # Generate embeddings for the provided text
    def embed_documents(self, documents, model = "text-embedding-3-small", batch_size = 32):

        # Normalize new lines
        documents = documents.replace("\n", " ")

        # Request embeddings from the API
        response = self.client.embeddings.create(input = [documents], model = model)

        # Extract embeddings from the response
        embeddings = [data.embedding for data in response.data]

        # Return a numpy array
        return np.array(embeddings)

    # Issue a completion call with the configured model
    def completion(self, prompt, model = "gpt-4o-mini", max_tokens = 150, temperature = 0.3):

        # Request the completion
        response = self.client.chat.completions.create(model = model,
                                                       messages = prompt,
                                                       max_tokens = max_tokens,
                                                       temperature = temperature,
                                                       n = 1)

        # Return the textual content
        return response.choices[0].message.content

# Document processor that produces splits, embeddings, and vector indexes
class DocumentProcessor:

    # Initialize the processor with the embedding model identifier
    def __init__(self, model = "text-embedding-3-small"):

        # Store the embedding model name
        self.model = model

        # Configure the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

        # Instantiate the OpenAI embedding helper with the Streamlit secret
        self.openai_model = OpenAIEmbedding(api_key = st.secrets["API_KEY"])

    # Split documents into chunks and compute embeddings
    def process_documents(self, documents):

        # Create document splits
        splits = self.text_splitter.split_documents(documents)

        # Collect embeddings for each chunk
        embeddings = []

        # Generate embeddings chunk by chunk
        for chunk in splits:
            embedd = self.openai_model.embed_documents(chunk.page_content, model = self.model)
            embeddings.extend(embedd)

        # Convert the list to a numpy array
        embedding_array = np.array(embeddings, dtype = "float32")

        # Determine embedding dimensionality
        dimension = embedding_array.shape[1]

        # Build an in-memory FAISS index
        vector_store = faiss.IndexFlatL2(dimension)

        # Populate the index
        vector_store.add(embedding_array)
        
        # Store the original splits for later retrieval
        self.documents = splits

        # Return the splits, vector store, embedding helper, and processed documents
        return splits, vector_store, self.openai_model, self.documents

