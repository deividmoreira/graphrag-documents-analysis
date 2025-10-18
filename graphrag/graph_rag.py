# Componente central que integra processamento de documentos, grafo de conhecimento e motor de consulta

# Imports
from graphrag.document_processor import DocumentProcessor  # Classe para processar documentos e gerar embeddings
from graphrag.knowledge_graph import KnowledgeGraph        # Classe para construção do grafo de conhecimento
from graphrag.query_engine import QueryEngine              # Classe para execução de consultas no grafo e vetor de embeddings

# Classe GraphRAG que combina processamento de documentos, grafo e motor de consulta
class GraphRAG:

    # Inicializa a classe e seus componentes
    def __init__(self):

        # Instância do DocumentProcessor para dividir documentos e gerar embeddings
        self.document_processor = DocumentProcessor()
        
        # Modelo OpenAI inicializado no DocumentProcessor, utilizado para gerar embeddings e respostas
        self.openai = self.document_processor.openai_model
        
        # Instância do KnowledgeGraph, que armazena o grafo de conhecimento com entidades e conceitos
        self.knowledge_graph = KnowledgeGraph(openai_model = self.openai)
        
        # Inicializa o QueryEngine como None, pois será definido após o processamento dos documentos
        self.query_engine = None
    
    # Função para processar documentos, criando embeddings, grafo de conhecimento e um motor de consulta
    def process_documents(self, documents):

        # Usa o DocumentProcessor para dividir e gerar embeddings dos documentos
        splits, vector_store, _, documents = self.document_processor.process_documents(documents) 
        
        # Constrói o grafo de conhecimento com os pedaços (splits) dos documentos
        self.knowledge_graph.build_graph(splits)
        
        # Inicializa o QueryEngine, que permite fazer consultas no grafo e no vetor de embeddings
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.openai, documents)

    # Função para fazer uma consulta no grafo e vetor de embeddings
    def query(self, query: str):
        
        # Executa a consulta usando o QueryEngine e retorna a resposta
        response, traversal_path, filtered_content = self.query_engine.query(query)

        return response


