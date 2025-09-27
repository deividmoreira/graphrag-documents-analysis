# Projeto 5 - Grafo de Conhecimento com GraphRAG Para Aplicação de Análise de Contratos com IA

# Importação para criação de arquivos temporários
import tempfile

# Framework Streamlit para interface do usuário
import streamlit as st

# Loader de documentos PDF
from langchain_community.document_loaders import PyPDFLoader

# Biblioteca GraphRAG para análise com grafos
from graphrag.dsa_graphrag import GraphRAG

# Componente chat para Streamlit
from streamlit_chat import message

# Execução concorrente para melhorar desempenho
from concurrent.futures import ThreadPoolExecutor

# Função para carregar contratos usando PyPDFLoader
def dsa_carrega_contrato(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents[:20]

# Função para realizar consultas com GraphRAG em documentos carregados
def dsa_query_graph_rag(documents, query):
    graph_rag = GraphRAG()
    graph_rag.process_documents(documents)
    return graph_rag.query(query)

# Função principal que define o aplicativo Streamlit
def main():

    # Configuração da página no Streamlit
    st.set_page_config(page_title="Data Science Academy", page_icon=":100:", layout="wide")

    # Título e subtítulo do projeto na interface
    st.markdown("<h1 style='text-align: center;'>🧠 DSA RAG Projeto 5</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Grafo de Conhecimento com GraphRAG Para Aplicação de Análise de Contratos com IA</h4>", unsafe_allow_html=True)

    # Barra lateral com instruções de uso
    st.sidebar.title("📌 Instruções de Uso")
    st.sidebar.write("""
    1. Faça o upload de um contrato em PDF.
    2. Aguarde o processamento.
    3. Pergunte algo sobre o contrato.
    4. Receba respostas contextualizadas pela IA.
    5. IA Generativa comete erros. SEMPRE verifique as respostas.
    """)

    # Informações adicionais sobre uso
    st.sidebar.info("💡 Dica: perguntas específicas geram melhores respostas.")

    # Botão de suporte ao usuário
    if st.sidebar.button("Suporte"):
        st.sidebar.write("Dúvidas? Envie um e-mail para: suporte@datascienceacademy.com.br")

    # Inicialização do estado da sessão Streamlit
    if 'ready' not in st.session_state:
        st.session_state['ready'] = False
    if 'documents' not in st.session_state:
        st.session_state['documents'] = None
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Divisão visual no Streamlit
    st.divider()

    # Seção para upload de arquivo PDF
    st.subheader("📤 Upload do Arquivo")
    uploaded_file = st.file_uploader("Envie aqui seu contrato em PDF", type="pdf")

    # Processamento após o upload do arquivo
    if uploaded_file is not None:
        with st.spinner("🔍 Processando o documento..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Carregamento concorrente dos documentos
            with ThreadPoolExecutor() as executor:
                future = executor.submit(dsa_carrega_contrato, tmp_file_path)
                st.session_state['documents'] = future.result()

            st.session_state['ready'] = True

    st.divider()

    # Seção para envio de consultas após documentos carregados
    if st.session_state['ready'] and st.session_state['documents']:
        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key = 'dsa_form', clear_on_submit = True):
                query = st.text_input("💬 Pergunte algo sobre o contrato:", key = 'input')
                submit_button = st.form_submit_button(label = '🚀 Enviar')

            # Execução da consulta usando GraphRAG
            if submit_button and query:
                with st.spinner("🤖 A IA Está Processando Sua Consulta. Seja Paciente e Aguarde..."):
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(dsa_query_graph_rag, st.session_state['documents'], query)
                        output = future.result()

                    # Processamento e exibição da resposta
                    if output is not None:
                        if hasattr(output, 'content'):
                            response_text = output.content
                        elif hasattr(output, 'text'):
                            response_text = output.text
                        else:
                            response_text = output

                        st.session_state.past.append(query)
                        st.session_state.generated.append(response_text)

        # Exibição das interações anteriores
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

# Execução da função principal
if __name__ == '__main__':
    main()



