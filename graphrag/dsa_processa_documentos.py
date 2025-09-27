# Projeto 5 - Grafo de Conhecimento com GraphRAG Para Aplicação de Análise de Contratos com IA
# Este módulo é usado para processar dados de texto e gerar embeddings

# Biblioteca para busca eficiente por similaridade de vetores
import faiss

# Biblioteca NumPy para operações matemáticas com vetores
import numpy as np

# Biblioteca Streamlit para criar aplicações web
import streamlit as st

# Cliente OpenAI para interação com a API
from openai import OpenAI

# Divisor de texto recursivo para dividir documentos grandes em partes menores
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Classe para gerar embeddings de documentos usando a API OpenAI
class OpenAIEmbedding:

    # Inicializa a classe com a chave da API OpenAI
    def __init__(self, api_key):

        # Cria o cliente OpenAI usando a chave fornecida
        self.client = OpenAI(api_key = api_key)

    # Método para gerar embeddings dos documentos usando modelo OpenAI
    def embed_documents(self, documents, model = "text-embedding-3-small", batch_size = 32):

        # Remove quebras de linha dos documentos
        documents = documents.replace("\n", " ")

        # Faz uma solicitação para gerar embeddings usando a API OpenAI
        response = self.client.embeddings.create(input = [documents], model = model)

        # Extrai os embeddings da resposta da API
        embeddings = [data.embedding for data in response.data]

        # Retorna os embeddings como um array NumPy
        return np.array(embeddings)

    # Método para obter uma resposta de completude usando o modelo GPT-4o-mini
    def completion(self, prompt, model = "gpt-4o-mini", max_tokens = 150, temperature = 0.3):

        # Solicita resposta ao modelo GPT
        response = self.client.chat.completions.create(model = model,
                                                       messages = prompt,
                                                       max_tokens = max_tokens,
                                                       temperature = temperature,
                                                       n = 1)

        # Retorna o conteúdo da primeira resposta gerada
        return response.choices[0].message.content

# Classe para processar documentos e calcular embeddings
class DocumentProcessor:

    # Inicializa a classe com o modelo de embeddings a ser utilizado
    def __init__(self, model = "text-embedding-3-small"):

        # Armazena o modelo escolhido
        self.model = model

        # Inicializa o divisor de texto com parâmetros específicos para chunking
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

        # Instancia o modelo OpenAI usando a chave secreta armazenada no Streamlit
        self.openai_model = OpenAIEmbedding(api_key = st.secrets["API_KEY"])

    # Método para dividir documentos em partes e calcular embeddings
    def process_documents(self, documents):

        # Divide documentos grandes em pequenos pedaços (chunks)
        splits = self.text_splitter.split_documents(documents)

        # Inicializa lista vazia para armazenar embeddings
        embeddings = []

        # Gera embeddings para cada chunk do documento
        for chunk in splits:
            embedd = self.openai_model.embed_documents(chunk.page_content, model = self.model)
            embeddings.extend(embedd)

        # Converte a lista de embeddings em um array NumPy
        embedding_array = np.array(embeddings, dtype = "float32")

        # Determina a dimensão dos embeddings
        dimension = embedding_array.shape[1]

        # Inicializa o armazenamento vetorial usando FAISS
        vector_store = faiss.IndexFlatL2(dimension)

        # Adiciona os embeddings ao armazenamento vetorial
        vector_store.add(embedding_array)
        
        # Armazena as divisões originais dos documentos
        self.documents = splits

        # Retorna os splits, armazenamento vetorial, modelo OpenAI e documentos processados
        return splits, vector_store, self.openai_model, self.documents



