# Streamlit app for contract analysis powered by GraphRAG

# Utilities to manage temporary files
import tempfile

# Streamlit framework for the user interface
import streamlit as st

# PDF document loader
from langchain_community.document_loaders import PyPDFLoader

# GraphRAG orchestration layer
from graphrag.graph_rag import GraphRAG

# Chat component for Streamlit
from streamlit_chat import message

# Concurrency utilities
from concurrent.futures import ThreadPoolExecutor

# Load a contract with PyPDFLoader
def load_contract(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents[:20]

# Execute a GraphRAG query on the uploaded documents
def query_graph_rag(documents, query):
    graph_rag = GraphRAG()
    graph_rag.process_documents(documents)
    return graph_rag.query(query)

# Entry point for the Streamlit application
def main():

    # Global Streamlit page configuration
    st.set_page_config(page_title="GraphRAG Contract Analyzer", page_icon=":100:", layout="wide")

    # Title and subtitle
    st.markdown("<h1 style='text-align: center;'>🧠 GraphRAG Contract Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Build knowledge graphs and answer questions about PDF contracts with AI</h4>", unsafe_allow_html=True)

    # Sidebar instructions
    st.sidebar.title("📌 Usage Instructions")
    st.sidebar.write("""
    1. Upload a contract in PDF format.
    2. Wait for the processing to finish.
    3. Ask a question about the contract.
    4. Review the AI-powered answer that references the document.
    5. Generative AI can be wrong. Always validate the output.
    """)

    # Additional hint
    st.sidebar.info("💡 Tip: Specific questions lead to more precise answers.")

    # Session state initialization
    if 'ready' not in st.session_state:
        st.session_state['ready'] = False
    if 'documents' not in st.session_state:
        st.session_state['documents'] = None
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Horizontal separator
    st.divider()

    # File upload
    st.subheader("📤 Upload Contract")
    uploaded_file = st.file_uploader("Upload your contract in PDF format", type="pdf")

    # Processing pipeline once the file is available
    if uploaded_file is not None:
        with st.spinner("🔍 Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Concurrent load of the document set
            with ThreadPoolExecutor() as executor:
                future = executor.submit(load_contract, tmp_file_path)
                st.session_state['documents'] = future.result()

            st.session_state['ready'] = True

    st.divider()

    # Query section
    if st.session_state['ready'] and st.session_state['documents']:
        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key = 'query_form', clear_on_submit = True):
                query = st.text_input("💬 Ask something about the contract:", key = 'input')
                submit_button = st.form_submit_button(label = '🚀 Send')

            # Execute the GraphRAG query
            if submit_button and query:
                with st.spinner("🤖 The AI is processing your request. Please wait..."):
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(query_graph_rag, st.session_state['documents'], query)
                        output = future.result()

                    # Handle the response payload
                    if output is not None:
                        if hasattr(output, 'content'):
                            response_text = output.content
                        elif hasattr(output, 'text'):
                            response_text = output.text
                        else:
                            response_text = output

                        st.session_state.past.append(query)
                        st.session_state.generated.append(response_text)

        # Display chat history
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

# Entrypoint
if __name__ == '__main__':
    main()
