# Projeto 5 - Grafo de Conhecimento com GraphRAG Para Aplicação de Análise de Contratos com IA
# Este módulo é usado no motor de consultas

# Imports
import heapq                          # Para gerenciamento de fila de prioridade
import numpy as np                    # Para manipulação de arrays e cálculos numéricos
import streamlit as st                # Para criação de aplicativos web interativos
from typing import List, Tuple, Dict  # Para anotações de tipos

# Classe para verificar se o contexto contém uma resposta completa para uma query
class AnswerCheck:

    # Construtor
    def __init__(self, openai_model, model = "gpt-4o-mini"):

        # Nome do modelo utilizado
        self.model = model  

        # Modelo OpenAI para geração de respostas
        self.OpenAIModel = openai_model  

    # Função para verificar se o contexto oferece uma resposta completa para a query
    def check_answer(self, query, context):

        # Prompt para instruir o modelo a verificar a completude do contexto
        prompt = [
            {"role": "system", 
             "content": f"Dada a query {query} e o contexto:{context}, diga se o contexto fornece uma resposta completa. Sim ou Não. Se sim, forneça a resposta."
            }
        ]
        
        # Aplica o LLM
        response = self.OpenAIModel.completion(prompt = prompt)
        
        # Limpa a resposta retornada e verifica se o contexto é completo
        text_response = response.replace("Sim, o contexto fornece uma resposta completa.", "")

        # Verifica se a resposta contém a confirmação
        is_complete = "Yes" in text_response 

        # Armazena a resposta se completa 
        answer = text_response if is_complete else None  
        
        # Retorna se o contexto é completo e a resposta gerada
        return is_complete, answer  

# Classe para executar a consulta no grafo de conhecimento e no índice vetorial
class QueryEngine:
    
    # Método construtor
    def __init__(self, 
                 vector_store,                            # Índice vetorial 
                 knowledge_graph,                         # Grafo de conhecimento 
                 openai_model,                            # Modelo OpenAI utilizado
                 documents,                               # Documentos processados
                 model_name = 'distilbert-base-uncased',  # Nome do modelo de embedding
                 model = "gpt-4o-mini"):                  # Nome do LLM utilizado
        
        self.vector_store = vector_store                                   # Armazena o índice vetorial
        self.knowledge_graph = knowledge_graph                             # Grafo de conhecimento
        self.documents = documents                                         # Lista de documentos processados
        self.openai_model = openai_model                                   # Modelo OpenAI para respostas
        self.answer_check = AnswerCheck(openai_model = self.openai_model)  # Instância para verificar respostas
        self.model = model                                                 # Nome do LLM utilizado
        self.max_content_length = 4000                                     # Limite máximo de caracteres no contexto

    # Função para gerar embeddings para o texto
    def get_embedding(self, text):

        # Remove quebras de linha do texto
        text = text.replace("\n", " ")  

        # Gera embeddings para o texto
        embeddings = self.openai_model.embed_documents(text)  

        # Retorna os embeddings gerados
        return embeddings  

    # Função para gerar uma resposta baseada na query e contexto fornecidos
    def generate_answer(self, query, context):

        # Prompt inicial para verificar a completude do contexto
        prompt = [
            {"role": "system", 
             "content": f"Dada a query {query} e o contexto: {context}, forneça a resposta completa se o contexto for suficiente."
            }
        ]
        
        # Aplica o LLM
        response = self.openai_model.completion(prompt = prompt)
        
        # Verifica se o contexto é suficiente para responder
        if "Yes" in response:

            # Prompt final para gerar a resposta completa
            prompt = [
                {"role": "system", 
                 "content": f"Dada a query {query} e o contexto: {context}, apenas responda a pergunta."
                }
            ]

            # Aplica o LLM
            final_response = self.openai_model.completion(prompt=prompt, temperature=0.3)

            # Retorna a resposta final
            return final_response.replace("Sim, o contexto fornece uma resposta completa.", "")  
        
        return response.replace("Sim, o contexto fornece uma resposta completa.", "")  

    # Expande o contexto buscando nós relevantes no grafo
    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:

        expanded_context = ""     # Inicializa o contexto expandido
        traversal_path = []       # Caminho percorrido no grafo
        visited_concepts = set()  # Conjuntos de conceitos já visitados
        filtered_content = {}     # Conteúdo filtrado dos nós visitados
        final_answer = ""         # Resposta final gerada
        priority_queue = []       # Fila de prioridade para busca no grafo
        distances = {}            # Distâncias acumuladas para cada nó

        # Adiciona documentos relevantes à fila de prioridade com base na similaridade
        for doc in relevant_docs:

            # Busca vetorial para retornar índices e score de similaridade entre as embeddings
            similarity_score, indices = self.vector_store.search(self.get_embedding(doc.page_content), k=1)

            # Conteúdo do nó mais próximo
            closest_node_content = [self.documents[i] for i in indices[0]]  
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content[0].page_content)
            
            # Calcula prioridade com base na similaridade
            priority = 1 / (similarity_score if similarity_score != 0 else 1e-10)  

            # Adiciona à fila
            heapq.heappush(priority_queue, (priority, closest_node))  

            # Atualiza distância do nó
            distances[closest_node] = priority  

        # Processa a fila de prioridade até encontrar um contexto completo
        while priority_queue:

            # Remove o nó de maior prioridade
            current_priority, current_node = heapq.heappop(priority_queue)  

            # Ignora se a prioridade for maior que a distância registrada
            if current_priority > distances.get(current_node, float('inf')):
                continue  

            # Se não for o nó atual, criamos o caminho do grafo
            if current_node not in traversal_path:

                # Adiciona nó ao caminho percorrido
                traversal_path.append(current_node)  

                # Conteúdo do nó
                node_content = self.knowledge_graph.graph.nodes[current_node]['content'] 

                # Conceitos do nó 
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']  

                # Adiciona conteúdo filtrado
                filtered_content[current_node] = node_content  

                # Atualiza o contexto expandido
                expanded_context += "\n" + node_content if expanded_context else node_content  

                # Verifica se o contexto é suficiente para responder a query
                is_complete, answer = self.answer_check.check_answer(query, expanded_context)

                # Define a resposta final
                if is_complete:
                    final_answer = answer  
                    break

                # Expande o contexto buscando conceitos e nós vizinhos
                node_concepts_set = set(self.knowledge_graph._lemmatize_concepts(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):

                    # Atualiza conceitos visitados
                    visited_concepts.update(node_concepts_set)  

                    # Loop
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):

                        # Dados da aresta
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]  

                        # Peso da aresta
                        edge_weight = edge_data['weight']  

                        # Calcula nova distância
                        distance = current_priority + (1 / edge_weight)  

                        # Verifica a distância do nó vizinho
                        if distance < distances.get(neighbor, float('inf')):

                            # Atualiza a distância do vizinho
                            distances[neighbor] = distance  

                            # Adiciona à fila
                            heapq.heappush(priority_queue, (distance, neighbor))  

                            # Atualiza o contexto expandido
                            neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                            neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']
                            filtered_content[neighbor] = neighbor_content

                            # Adiciona conteúdo do vizinho
                            expanded_context += "\n" + neighbor_content  

                            # Verifica novamente se o contexto é suficiente
                            is_complete, answer = self.answer_check.check_answer(query, expanded_context)
                            if is_complete:
                                final_answer = answer
                                break

                            neighbor_concepts_set = set(self.knowledge_graph._lemmatize_concepts(c) for c in neighbor_concepts)

                            # Atualiza conceitos visitados
                            if not neighbor_concepts_set.issubset(visited_concepts):
                                visited_concepts.update(neighbor_concepts_set)  

                if final_answer:
                    break

        # Caso nenhum contexto completo seja encontrado, gera uma resposta com o contexto expandido
        if not final_answer:
            final_answer = self.generate_answer(query, expanded_context)

        return expanded_context, traversal_path, filtered_content, final_answer

    # Função para realizar uma consulta usando a query
    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:

        # Recupera documentos relevantes
        relevant_docs = self._retrieve_relevant_documents(query)  

        # Expande o contexto
        expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)  

        # Se não houver resposta final, gera uma resposta completa com o contexto expandido
        if not final_answer:

            prompt = [
                {"role": "system", 
                 "content": f"Dada a pergunta e o contexto: {expanded_context}, apenas responda a pergunta."
                },
                {"role": "user", "content": f"Aqui está a minha pergunta: {query}"}
            ]

            # Aplica o LLM e obtém a resposta
            response = self.openai_model.completion(prompt = prompt, temperature = 0.3, max_tokens = 500)
            final_answer = response.replace("Sim, o contexto fornece uma resposta completa.", "")  

        # Retorna a resposta final, caminho e conteúdo filtrado
        return final_answer, traversal_path, filtered_content  
    
    # Função para recuperar documentos relevantes com base na query
    def _retrieve_relevant_documents(self, query: str):

        # Gera embedding para a query
        query_embedding = self.get_embedding(query)  

        # Busca os documentos mais próximos
        distance, indices = self.vector_store.search(query_embedding, k = 5)  

        # Recupera os documentos relevantes
        relevant_docs = [self.documents[i] for i in indices[0]] 

        # Retorna os documentos relevantes 
        return relevant_docs  






