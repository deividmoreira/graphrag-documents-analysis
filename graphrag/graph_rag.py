# Central component that wires the document processor, knowledge graph, and query engine

# Imports
from graphrag.document_processor import DocumentProcessor  # Handles document chunking and embeddings
from graphrag.knowledge_graph import KnowledgeGraph        # Builds the knowledge graph representation
from graphrag.query_engine import QueryEngine              # Executes hybrid queries over graph and vector store

# GraphRAG class that combines document processing, graph construction, and querying
class GraphRAG:

    # Initialize dependencies
    def __init__(self):

        # Chunk documents and create embeddings
        self.document_processor = DocumentProcessor()
        
        # Shared OpenAI model used for embeddings and completions
        self.openai = self.document_processor.openai_model
        
        # Build the knowledge graph that stores entities and concepts
        self.knowledge_graph = KnowledgeGraph(openai_model = self.openai)
        
        # Query engine is created after documents are processed
        self.query_engine = None
    
    # Process documents to generate embeddings, knowledge graph, and query engine
    def process_documents(self, documents):

        # Chunk documents and populate the vector index
        splits, vector_store, _, documents = self.document_processor.process_documents(documents) 
        
        # Build the knowledge graph from the document splits
        self.knowledge_graph.build_graph(splits)
        
        # Instantiate the hybrid query engine
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.openai, documents)

    # Query the graph/vector store hybrid engine
    def query(self, query: str):
        
        # Execute the query and return only the answer text
        response, traversal_path, filtered_content = self.query_engine.query(query)

        return response

