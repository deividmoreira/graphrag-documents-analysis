# Projeto 5 - Grafo de Conhecimento com GraphRAG Para Aplicação de Análise de Contratos com IA
# Este módulo é usado para gerar o grafo de conhecimento que será usado como contexto para o LLM responder a query do usuário

# Importa o módulo para manipulação de arquivos temporários
import tempfile

# Importa a biblioteca nltk para processamento de linguagem natural
import nltk

# Importa o módulo numpy para manipulação de arrays e operações numéricas
import numpy as np

# Importa a biblioteca networkx para criação e manipulação de grafos
import networkx as nx

# Importa o lematizador WordNet do nltk
from nltk.stem import WordNetLemmatizer

# Importa a função para calcular similaridade de cosseno entre vetores
from sklearn.metrics.pairwise import cosine_similarity

# Importa ferramentas para execução paralela de tarefas
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importa a biblioteca pydantic para validação de dados
from pydantic import BaseModel, Field

# Importa o tqdm para exibição de barras de progresso
from tqdm import tqdm

# Importa tipos úteis para anotações de funções
from typing import List, Tuple, Dict

# Tenta carregar o corpus WordNet para lematização
try:
    from nltk.corpus import wordnet as wn
# Baixa o corpus WordNet caso não esteja disponível
except LookupError:
    print("WordNet não encontrado. Baixando...")
    nltk.download('wordnet')

# Verifica se o corpus WordNet foi baixado, caso contrário, baixa novamente
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Define um modelo de dados para representar uma lista de conceitos
class Concepts(BaseModel):

    # Lista de conceitos descrita com um campo do pydantic
    concepts_list: List[str] = Field(description = "Lista de conceitos")

# Define uma classe para construção e manipulação de um grafo de conhecimento
class knowledgeGraph:

    # Inicializa o grafo de conhecimento 
    def __init__(self, openai_model):

        # Armazena o modelo OpenAI para geração de embeddings e respostas
        self.OpenAIModel = openai_model

        # Cria um grafo vazio usando a biblioteca networkx
        self.graph = nx.Graph()

        # Inicializa um lematizador do WordNet
        self.lemmatizer = WordNetLemmatizer()

        # Cria um cache para conceitos já processados
        self.concept_cache = {}

        # Define o limite de similaridade para criação de arestas
        self.edges_threshold = 0.8

    # Função para construir o grafo com base em documentos divididos
    def build_graph(self, splits):

        # Adiciona nós ao grafo baseados nos documentos
        self._add_nodes(splits)

        # Extrai conceitos dos documentos para criar conexões no grafo
        self._extract_concepts(splits)

        # Cria embeddings para os documentos e os armazena
        embeddings = self._create_embeddings(splits, self.OpenAIModel)

        # Adiciona arestas entre os nós com base na similaridade entre as embeddings
        self._add_edges(embeddings)

    # Adiciona nós ao grafo a partir de divisões dos documentos
    def _add_nodes(self, splits):

        # Loop
        for i, split in enumerate(splits):

            # Cada nó contém o conteúdo do documento associado
            self.graph.add_node(i, content = split.page_content)

    # Cria embeddings para cada documento dividido usando o LLM
    def _create_embeddings(self, splits, openai_model):
        
        # Cria a lista
        embeddings = []
        
        # Loop
        for split in splits:
            
            # Gera embeddings para o conteúdo do documento
            embedd = self.OpenAIModel.embed_documents(split.page_content)
            embeddings.extend(embedd)

        # Converte os embeddings para um array numpy
        embeddings = np.array(embeddings, dtype = "float32")
        
        # Garante que os embeddings têm a forma correta
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        return embeddings

    # Calcula a similaridade de cosseno entre as embeddings
    def _compute_similarities(self, embeddings):
        return cosine_similarity(embeddings)

    # Extrai entidades nomeadas de um conteúdo textual
    def named_entities(self, content):
        
        # Define um prompt para extrair entidades do texto
        prompt = [
            {
                "role": "system",
                "content": f"Dado o conteúdo: {content}, extraia entidades nomeadas do conteúdo."
            }
        ]
        
        # Gera a resposta do modelo OpenAI com as entidades
        response = self.OpenAIModel.completion(prompt = prompt)
        
        # Retorna a resposta como uma string
        named_entities = response
        
        return named_entities

    # Extrai conceitos e entidades gerais de um conteúdo textual
    def _extra_concepts_and_entities(self, content):
        
        # Verifica se os conceitos já estão no cache
        if content in self.concept_cache:
            return self.concept_cache[content]

        # Extrai entidades nomeadas do conteúdo
        named_entities = [self.named_entities(content)]
        
        # Define um prompt para extrair conceitos gerais
        prompts = (
            f"Extraia os principais conceitos (excluindo entidades nomeadas) do texto a seguir:\n\n"
            f"{content}\n\n"
            f"Principais conceitos:"
        )
        
        # Gera a resposta com os conceitos gerais
        response = self.OpenAIModel.completion(prompt = [{"role": "user", "content": prompts}], temperature = 0.4)
        
        # Divide os conceitos em uma lista a partir da resposta
        general_concepts = response.strip().split(', ')
        
        # Combina entidades nomeadas e conceitos gerais em uma lista única
        all_concepts = list(set(named_entities + general_concepts))
        
        # Armazena os conceitos no cache
        self.concept_cache[content] = all_concepts

        return all_concepts

    # Extrai conceitos para cada divisão do documento e adiciona ao grafo
    def _extract_concepts(self, splits):

        # Trabalha com ThreadPoolExecutor para otimizar a performance
        with ThreadPoolExecutor() as executor:
            
            # Associa cada divisão a uma tarefa assíncrona de extração de conceitos
            future_to_node = {
                executor.submit(self._extra_concepts_and_entities, split.page_content): i
                for i, split in enumerate(splits)
            }
            
            # Loop para extração de conceitos e entidades
            for future in tqdm(as_completed(future_to_node), total = len(splits), desc = "Extraindo conceitos e entidades"):
                
                # Obtém o índice do nó associado à tarefa concluída
                node = future_to_node[future]
                
                # Recupera os conceitos processados
                concepts = future.result()
                
                # Adiciona os conceitos ao nó correspondente no grafo
                self.graph.nodes[node]['concepts'] = concepts

    # Adiciona arestas entre os nós com base na similaridade e conceitos compartilhados
    def _add_edges(self, embeddings):
        
        # Calcula a matriz de similaridade entre as embeddings
        similarity_matrix = self._compute_similarities(embeddings)
        
        # Obtém o número total de nós no grafo
        num_nodes = len(self.graph.nodes)

        # Loop para adicionar as arestas entre nós
        for node1 in tqdm(range(num_nodes), desc = "Adicionando arestas"):
            for node2 in range(node1 + 1, num_nodes):
                try:
                    # Recupera a pontuação de similaridade entre os nós
                    similarity_score = similarity_matrix[node1][node2]
                except IndexError:
                    continue

                # Compara o score de similaridade com o limite definido
                if similarity_score > self.edges_threshold:
                    
                    # Identifica conceitos compartilhados entre os dois nós
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(self.graph.nodes[node2]['concepts'])
                    
                    # Calcula o peso da aresta com base na similaridade e nos conceitos
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    
                    # Adiciona a aresta ao grafo
                    self.graph.add_edge(node1, 
                                        node2, 
                                        weight = edge_weight, 
                                        similarity = similarity_score,
                                        shared_concepts = list(shared_concepts))

    # Calcula o peso de uma aresta com base na similaridade e nos conceitos compartilhados
    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha = 0.7, beta = 0.3):
        
        # Determina o número máximo de conceitos compartilhados possíveis
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        
        # Normaliza os conceitos compartilhados pelo máximo possível
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        
        # Calcula o peso final da aresta
        return alpha * similarity_score + beta * normalized_shared_concepts

    # Lematiza conceitos para normalizar variações de palavras
    def _lemmatize_concepts(self, concept):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])




