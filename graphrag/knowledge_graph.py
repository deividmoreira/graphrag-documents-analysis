# Builds the knowledge graph used as context for LLM answers

# Natural language processing utilities
import nltk

# Array and numerical operations
import numpy as np

# Graph construction and analysis
import networkx as nx

# WordNet lemmatizer
from nltk.stem import WordNetLemmatizer

# Cosine similarity helper
from sklearn.metrics.pairwise import cosine_similarity

# Parallel execution utilities
from concurrent.futures import ThreadPoolExecutor, as_completed

# Pydantic for data validation
from pydantic import BaseModel, Field

# Progress bar helper
from tqdm import tqdm

# Typing helpers
from typing import List, Tuple, Dict

# Attempt to load the WordNet corpus for lemmatization
try:
    from nltk.corpus import wordnet as wn
# Download the corpus if it is not present
except LookupError:
    print("WordNet not found. Downloading...")
    nltk.download('wordnet')

# Double-check that the corpus is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Pydantic model that represents a list of concepts
class Concepts(BaseModel):

    # Concept list field
    concepts_list: List[str] = Field(description = "List of concepts")

# Knowledge graph builder and helper functions
class KnowledgeGraph:

    # Initialize the knowledge graph with an OpenAI model
    def __init__(self, openai_model):

        # Store the OpenAI model used for embeddings and completions
        self.openai_model = openai_model

        # Initialize an empty graph
        self.graph = nx.Graph()

        # WordNet lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Cache concepts that were already extracted
        self.concept_cache = {}

        # Similarity threshold used when connecting nodes
        self.edges_threshold = 0.8

    # Build the knowledge graph from the document splits
    def build_graph(self, splits):

        # Register nodes for each split
        self._add_nodes(splits)

        # Extract concepts that will drive the edges
        self._extract_concepts(splits)

        # Create embeddings that back the similarity scores
        embeddings = self._create_embeddings(splits)

        # Connect nodes that are similar enough
        self._add_edges(embeddings)

    # Add one node per document split
    def _add_nodes(self, splits):

        # Iterate over document splits
        for i, split in enumerate(splits):

            # Store the chunk content as node metadata
            self.graph.add_node(i, content = split.page_content)

    # Create embeddings for each split
    def _create_embeddings(self, splits):
        
        # Embedding accumulator
        embeddings = []
        
        # Generate embeddings per chunk
        for split in splits:
            
            # Encode the chunk content
            embedd = self.openai_model.embed_documents(split.page_content)
            embeddings.extend(embedd)

        # Convert the list into a numpy array
        embeddings = np.array(embeddings, dtype = "float32")
        
        # Ensure the shape is two-dimensional
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        return embeddings

    # Compute cosine similarity between embeddings
    def _compute_similarities(self, embeddings):
        return cosine_similarity(embeddings)

    # Extract named entities from the content
    def named_entities(self, content):
        
        # Prompt the LLM for entity extraction
        prompt = [
            {
                "role": "system",
                "content": f"Given the following content: {content}, extract the named entities."
            }
        ]
        
        # Request the entities from the LLM
        response = self.openai_model.completion(prompt = prompt)
        
        # Return the entities string
        named_entities = response
        
        return named_entities

    # Extract concepts and entities from a text chunk
    def _extra_concepts_and_entities(self, content):
        
        # Check if the content is already cached
        if content in self.concept_cache:
            return self.concept_cache[content]

        # Retrieve named entities
        named_entities = [self.named_entities(content)]
        
        # Prompt to obtain key concepts
        prompts = (
            f"Extract the main concepts (excluding named entities) from the text below:\n\n"
            f"{content}\n\n"
            f"Key concepts:"
        )
        
        # Get the concept list
        response = self.openai_model.completion(prompt = [{"role": "user", "content": prompts}], temperature = 0.4)
        
        # Convert the response into a list
        general_concepts = response.strip().split(', ')
        
        # Combine named entities and generic concepts
        all_concepts = list(set(named_entities + general_concepts))
        
        # Cache the result
        self.concept_cache[content] = all_concepts

        return all_concepts

    # Extract concepts for every split and store them in the graph
    def _extract_concepts(self, splits):

        # Use a thread pool for faster extraction
        with ThreadPoolExecutor() as executor:
            
            # Map each split to an asynchronous task
            future_to_node = {
                executor.submit(self._extra_concepts_and_entities, split.page_content): i
                for i, split in enumerate(splits)
            }
            
            # Collect results as they complete
            for future in tqdm(as_completed(future_to_node), total = len(splits), desc = "Extracting concepts and entities"):
                
                # Get the node identifier
                node = future_to_node[future]
                
                # Retrieve the concept list
                concepts = future.result()
                
                # Attach concepts to the node metadata
                self.graph.nodes[node]['concepts'] = concepts

    # Add edges based on similarity scores and shared concepts
    def _add_edges(self, embeddings):
        
        # Compute the similarity matrix
        similarity_matrix = self._compute_similarities(embeddings)
        
        # Total number of nodes in the graph
        num_nodes = len(self.graph.nodes)

        # Iterate over all node pairs
        for node1 in tqdm(range(num_nodes), desc = "Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                try:
                    # Fetch similarity score between nodes
                    similarity_score = similarity_matrix[node1][node2]
                except IndexError:
                    continue

                # Check if the similarity is above threshold
                if similarity_score > self.edges_threshold:
                    
                    # Concepts shared by both nodes
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(self.graph.nodes[node2]['concepts'])
                    
                    # Compute edge weight based on similarity and shared concepts
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    
                    # Create the edge with metadata
                    self.graph.add_edge(node1, 
                                        node2, 
                                        weight = edge_weight, 
                                        similarity = similarity_score,
                                        shared_concepts = list(shared_concepts))

    # Compute edge weight using similarity and shared concepts
    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha = 0.7, beta = 0.3):
        
        # Maximum possible shared concepts
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        
        # Normalize shared concepts
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        
        # Final weight calculation
        return alpha * similarity_score + beta * normalized_shared_concepts

    # Lemmatize concept text to normalize variants
    def _lemmatize_concepts(self, concept):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])
