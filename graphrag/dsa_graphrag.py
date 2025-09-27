# Projeto 5 - Grafo de Conhecimento com GraphRAG Para Aplicação de Análise de Contratos com IA
# Este módulo une os demais módulos criando assim o GraphRAG

# Imports
import numpy as np
from graphrag.dsa_processa_documentos import DocumentProcessor  # Classe para processar documentos e gerar embeddings
from graphrag.dsa_knowledgegraph import knowledgeGraph          # Classe para construção do grafo de conhecimento
from graphrag.dsa_queryengine import QueryEngine                # Classe para execução de consultas no grafo e vetor de embeddings

# Classe GraphRAG, que estende o DocumentProcessor e combina a criação de grafo de conhecimento e consulta
class GraphRAG(DocumentProcessor):

    # Inicializa a classe e seus componentes
    def __init__(self):

        # Instância do DocumentProcessor para dividir documentos e gerar embeddings
        self.dsa_processa_documentos = DocumentProcessor()
        
        # Modelo OpenAI inicializado no DocumentProcessor, utilizado para gerar embeddings e respostas
        self.openai = DocumentProcessor().openai_model
        
        # Instância do KnowledgeGraph, que armazena o grafo de conhecimento com entidades e conceitos
        self.knowledge_graph = knowledgeGraph(openai_model = self.openai)
        
        # Inicializa o QueryEngine como None, pois será definido após o processamento dos documentos
        self.query_engine = None
    
    # Função para processar documentos, criando embeddings, grafo de conhecimento e um motor de consulta
    def process_documents(self, documents):

        # Usa o DocumentProcessor para dividir e gerar embeddings dos documentos
        splits, vector_store, openai_model, documents = self.dsa_processa_documentos.process_documents(documents) 
        
        # Constrói o grafo de conhecimento com os pedaços (splits) dos documentos
        self.knowledge_graph.build_graph(splits)
        
        # Inicializa o QueryEngine, que permite fazer consultas no grafo e no vetor de embeddings
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.openai, documents)

    # Função para fazer uma consulta no grafo e vetor de embeddings
    def query(self, query: str):
        
        # Executa a consulta usando o QueryEngine e retorna a resposta
        response, traversal_path, filtered_content = self.query_engine.query(query)

        return response






