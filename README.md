# GraphRAG Contract Analyzer

Aplicativo de referência que combina RAG (Retrieval-Augmented Generation) com grafos de conhecimento para responder perguntas sobre contratos em PDF. Desenvolvido para servir como base a outros profissionais que desejam construir soluções de análise documental usando Streamlit, LangChain, NetworkX, FAISS e OpenAI.

## Visão Geral

- Upload de contratos em PDF com extração automática de até 20 páginas.
- Geração de chunks, embeddings vetoriais e grafo de conhecimento em paralelo.
- Motor de consulta híbrido (vetorial + grafo) com verificação de completude de resposta.
- Interface Streamlit otimizada para interação tipo chat.
- Estrutura modular para facilitar reuso e experimentação.

## Arquitetura em alto nível

1. **Ingestão de documentos** (`graphrag/document_processor.py`)  
   - Carrega PDFs com `PyPDFLoader`.  
   - Realiza chunking com `RecursiveCharacterTextSplitter`.  
   - Gera embeddings via API OpenAI e indexa no FAISS.
2. **Grafo de conhecimento** (`graphrag/knowledge_graph.py`)  
   - Extrai conceitos e entidades com auxílio do LLM.  
   - Constrói um grafo NetworkX ponderado por similaridade e conceitos compartilhados.
3. **Motor de consulta** (`graphrag/query_engine.py`)  
   - Recupera chunks relevantes pelo índice vetorial.  
   - Expande contexto percorrendo o grafo e valida se a resposta é completa.  
   - Gera a resposta final usando o LLM.
4. **Interface Streamlit** (`app.py`)  
   - Controle de estado da sessão e execução assíncrona com `ThreadPoolExecutor`.  
   - Exibição do histórico de perguntas e respostas com `streamlit-chat`.

## Pré-requisitos

- Python 3.12+
- Chave de API da OpenAI com acesso a modelos de embedding e chat
- Ambiente Conda ou virtualenv (recomendado)

## Configuração do ambiente

```bash
conda create --name graphrag-contracts python=3.12
conda activate graphrag-contracts
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Configure o segredo do Streamlit adicionando o arquivo `.streamlit/secrets.toml` na raiz do projeto:

```toml
[general]
API_KEY = "sua-chave-da-openai"
```

## Execução

```bash
streamlit run app.py
```

Após o upload de um PDF, faça perguntas na caixa de texto. O histórico de interação fica registrado no painel principal.

## Estrutura do projeto

```
.
├── app.py                       # Interface Streamlit
├── graphrag
│   ├── document_processor.py    # Pipeline de chunking + embeddings + FAISS
│   ├── graph_rag.py             # Orquestração do fluxo GraphRAG
│   ├── knowledge_graph.py       # Construção do grafo de conhecimento
│   └── query_engine.py          # Estratégia de consulta híbrida
├── requirements.txt
├── README.md
└── LEIAME.txt                   # Guia rápido em português
```

## Personalização

- Ajuste `chunk_size` e `chunk_overlap` em `DocumentProcessor` para adequar o tamanho dos trechos.
- Modifique `edges_threshold` em `KnowledgeGraph` para controlar a densidade do grafo.
- Troque os modelos de embedding ou chat atualizando os parâmetros em `OpenAIEmbedding`.
- Adapte o limite de `self.max_content_length` no `QueryEngine` para limitar o contexto enviado ao LLM.

## Próximos passos sugeridos

1. Adicionar armazenamento persistente das embeddings (FAISS on disk).
2. Implementar autenticação na interface Streamlit.
3. Criar suíte de testes automatizados para validação das etapas críticas.

---

Sinta-se à vontade para adaptar este projeto e mencioná-lo em apresentações ou posts técnicos. Ele foi estruturado para acelerar provas de conceito de GraphRAG aplicadas a análises documentais.
