# Query engine that blends the knowledge graph with the vector index

# Imports
import heapq                          # Priority queue management
from typing import List, Tuple, Dict  # Type hints

# Response prefix returned by the LLM when the context is sufficient
COMPLETENESS_PREFIX = "Yes, the context provides a complete answer."

# Helper class that checks whether the context already contains a full answer
class AnswerCheck:

    # Initialize the helper with the OpenAI model
    def __init__(self, openai_model, model = "gpt-4o-mini"):

        # Model identifier (kept for compatibility)
        self.model = model  

        # OpenAI client used to run prompts
        self.openai_model = openai_model  

    # Determine whether the provided context fully answers the query
    def check_answer(self, query, context):

        # Prompt instructing the model to state if the context is sufficient
        prompt = [
            {"role": "system", 
             "content": (
                 f"Given the query '{query}' and the following context: {context}. "
                 "Answer Yes or No; if Yes, provide the complete answer."
             )
            }
        ]
        
        # Run the completion
        response = self.openai_model.completion(prompt = prompt)

        # Normalize the response text
        normalized_response = response.strip()
        
        # Determine if the context is sufficient
        is_complete = normalized_response.startswith("Yes")

        # Extract the answer when available
        answer = normalized_response.replace(COMPLETENESS_PREFIX, "").strip() if is_complete else None  
        
        # Return the completeness flag and the answer text
        return is_complete, answer  

# Query engine that orchestrates graph and vector search
class QueryEngine:
    
    # Initialize the engine
    def __init__(self, 
                 vector_store,                            # Vector index
                 knowledge_graph,                         # Knowledge graph
                 openai_model,                            # OpenAI model instance
                 documents,                               # Processed documents
                 model_name = 'distilbert-base-uncased',  # Embedding identifier (kept for compatibility)
                 model = "gpt-4o-mini"):                  # LLM identifier
        
        self.vector_store = vector_store                                   # Vector index
        self.knowledge_graph = knowledge_graph                             # Knowledge graph structure
        self.documents = documents                                         # Stored document chunks
        self.openai_model = openai_model                                   # OpenAI client for answers
        self.answer_check = AnswerCheck(openai_model = self.openai_model)  # Completeness checker
        self.model = model                                                 # LLM identifier
        self.max_content_length = 4000                                     # Context length safeguard

    # Generate embeddings for a text query
    def get_embedding(self, text):

        # Normalize line breaks
        text = text.replace("\n", " ")  

        # Create embeddings
        embeddings = self.openai_model.embed_documents(text)  

        # Return the embedding vector
        return embeddings  

    # Generate an answer based on the current context
    def generate_answer(self, query, context):

        # Ask the model to confirm the context sufficiency
        prompt = [
            {"role": "system", 
             "content": (
                 f"Given the query '{query}' and the context: {context}, "
                 "confirm if the context is sufficient and provide the complete answer when it is."
             )
            }
        ]
        
        # Run the request
        response = self.openai_model.completion(prompt = prompt)
        
        # If the context is enough, request a direct answer
        if "Yes" in response:

            # Prompt to return just the answer
            prompt = [
                {"role": "system", 
                 "content": (
                     f"Given the query '{query}' and the context: {context}, answer the question directly."
                 )
                }
            ]

            # Fetch the final answer
            final_response = self.openai_model.completion(prompt=prompt, temperature=0.3)

            # Remove the completeness prefix if present
            return final_response.replace(COMPLETENESS_PREFIX, "")  
        
        return response.replace(COMPLETENESS_PREFIX, "")  

    # Expand the context by traversing relevant graph nodes
    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:

        expanded_context = ""     # Aggregated textual context
        traversal_path = []       # Sequence of visited nodes
        visited_concepts = set()  # Concepts already used
        filtered_content = {}     # Filtered node content
        final_answer = ""         # Final answer placeholder
        priority_queue = []       # Priority queue for traversal order
        distances = {}            # Accumulated distance per node

        # Seed the queue with the most relevant documents
        for doc in relevant_docs:

            # Vector search for the closest chunk
            similarity_score, indices = self.vector_store.search(self.get_embedding(doc.page_content), k=1)

            # Locate the closest graph node
            closest_node_content = [self.documents[i] for i in indices[0]]  
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content[0].page_content)
            
            # Invert similarity to turn into a priority
            priority = 1 / (similarity_score if similarity_score != 0 else 1e-10)  

            # Push into the queue
            heapq.heappush(priority_queue, (priority, closest_node))  

            # Track the best distance found so far
            distances[closest_node] = priority  

        # Continue exploring until the answer is complete
        while priority_queue:

            # Pop the node with the highest priority
            current_priority, current_node = heapq.heappop(priority_queue)  

            # Skip if we already found a better path
            if current_priority > distances.get(current_node, float('inf')):
                continue  

            # Only visit each node once
            if current_node not in traversal_path:

                # Record traversal order
                traversal_path.append(current_node)  

                # Node content and concepts
                node_content = self.knowledge_graph.graph.nodes[current_node]['content'] 
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']  

                # Snapshot of the node content
                filtered_content[current_node] = node_content  

                # Append to the context block
                expanded_context += "\n" + node_content if expanded_context else node_content  

                # Check if the context already answers the question
                is_complete, answer = self.answer_check.check_answer(query, expanded_context)

                # Exit early when we have a complete answer
                if is_complete:
                    final_answer = answer  
                    break

                # Expand traversal using connected nodes
                node_concepts_set = set(self.knowledge_graph._lemmatize_concepts(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):

                    # Register concepts that have been covered
                    visited_concepts.update(node_concepts_set)  

                    # Explore neighbors
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):

                        # Edge metadata
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]  

                        # Edge weight
                        edge_weight = edge_data['weight']  

                        # Update distance based on the edge
                        distance = current_priority + (1 / edge_weight)  

                        # Check if this path is better
                        if distance < distances.get(neighbor, float('inf')):

                            # Record the best distance
                            distances[neighbor] = distance  

                            # Queue the neighbor for exploration
                            heapq.heappush(priority_queue, (distance, neighbor))  

                            # Expand the context with neighbor content
                            neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                            neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']
                            filtered_content[neighbor] = neighbor_content

                            # Append neighbor text
                            expanded_context += "\n" + neighbor_content  

                            # Check again if we already have the answer
                            is_complete, answer = self.answer_check.check_answer(query, expanded_context)
                            if is_complete:
                                final_answer = answer
                                break

                            neighbor_concepts_set = set(self.knowledge_graph._lemmatize_concepts(c) for c in neighbor_concepts)

                            # Mark newly visited concepts
                            if not neighbor_concepts_set.issubset(visited_concepts):
                                visited_concepts.update(neighbor_concepts_set)  

                if final_answer:
                    break

        # Fall back to generating an answer if none was found during traversal
        if not final_answer:
            final_answer = self.generate_answer(query, expanded_context)

        return expanded_context, traversal_path, filtered_content, final_answer

    # Execute the full query pipeline
    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:

        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_documents(query)  

        # Expand graph context
        expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query, relevant_docs)  

        # If no final answer was produced, request one explicitly
        if not final_answer:

            prompt = [
                {"role": "system", 
                 "content": (
                     f"Given the question and the context: {expanded_context}, answer the question directly."
                 )
                },
                {"role": "user", "content": f"Here is the question: {query}"}
            ]

            # Request the final answer
            response = self.openai_model.completion(prompt = prompt, temperature = 0.3, max_tokens = 500)
            final_answer = response.replace(COMPLETENESS_PREFIX, "")  

        # Return the answer, traversal path, and filtered content
        return final_answer, traversal_path, filtered_content  
    
    # Retrieve relevant documents from the vector index
    def _retrieve_relevant_documents(self, query: str):

        # Encode the query
        query_embedding = self.get_embedding(query)  

        # Search for the top matches
        distance, indices = self.vector_store.search(query_embedding, k = 5)  

        # Collect the document splits
        relevant_docs = [self.documents[i] for i in indices[0]] 

        # Return the list of relevant documents
        return relevant_docs  




