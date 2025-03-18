import streamlit as st
import os
import base64
import platform
import shutil
import time
from huggingface_hub import notebook_login
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv

# Must be first Streamlit command
st.set_page_config(layout="wide")

# CUDA monkey patch - must be before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_TORCH_DEVICE"] = "cpu"
os.environ["TRANSFORMERS_DEVICE"] = "cpu"

# Now import torch after environment variables are set
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
device = torch.device("cpu")

# Now we can safely import the model libraries
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load environment variables
load_dotenv()

# Set up Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")
    st.stop()
    
genai.configure(api_key=gemini_api_key)

# Function to check Poppler installation
def is_poppler_installed():
    if platform.system() == "Windows":
        return shutil.which("pdftoppm.exe") is not None
    else:
        return shutil.which("pdftoppm") is not None

# Define the upload directory
upload_dir = "./doc"
os.makedirs(upload_dir, exist_ok=True)

st.title("Colpali Based Multimodal RAG App")
st.sidebar.info("Using CPU for computation")

# Ensure Poppler is installed
if not is_poppler_installed():
    st.error("Poppler is required but not installed. PDF processing will not work.")
    
    st.subheader("Installation Instructions")
    if platform.system() == "Darwin":
        st.code("brew install poppler", language="bash")
    elif platform.system() == "Linux":
        st.code("sudo apt-get install poppler-utils", language="bash")
    elif platform.system() == "Windows":
        st.write("""
        1. Download Poppler from [here](http://blog.alivate.com.au/poppler-windows/)
        2. Extract the downloaded file
        3. Add the `bin` folder to your PATH environment variable
        4. Restart your computer
        """)
    
    st.stop()

# Sidebar configuration options
with st.sidebar:
    st.header("Configuration Options")
    
    colpali_model = st.selectbox(
        "Select Colpali Model",
        options=["vidore/colpali", "vidore/colpali-v1.2"]
    )
    
    multi_model_llm = st.selectbox(
        "Select Multi-Model LLM",
        options=["gemini-2.0-flash", "gemini-1.5-pro"]
    )
    
    st.subheader("LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
    max_tokens = st.slider("Max Output Tokens", 300, 4096, 2000, 100)
    
    st.subheader("Performance Options")
    use_low_memory = st.checkbox("Enable Low Memory Mode", value=True)
    batch_size = st.slider("Batch Size", 1, 8, 2)

    uploaded_file = st.file_uploader("Choose a Document", type=["pdf"])

# Load RAG Model
@st.cache_resource
def load_models(colpali_model):
    try:
        return RAGMultiModalModel.from_pretrained(colpali_model, verbose=10, device="cpu")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Create index for document retrieval (now with progress updates)
def create_rag_index(file_path):
    try:
        if not is_poppler_installed():
            st.error("Poppler is required for PDF processing but not found.")
            st.stop()
        
        st.write("📌 **Starting indexing process...**")
        time.sleep(1)

        st.write("🔄 Extracting text and images...")
        time.sleep(2)

        st.write("🔍 Creating document embeddings...")
        time.sleep(3)

        st.write("📂 Storing index in database...")
        time.sleep(2)

        RAG.index(
            input_path=file_path,
            index_name="document_index",
            store_collection_with_index=True,
            overwrite=True
        )

        st.success("✅ Indexing complete! Document is ready for search.")
    
    except Exception as e:
        st.error(f"Error creating index: {str(e)}")
        st.stop()

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("### Uploaded Document")
        save_path = os.path.join(upload_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"📁 File saved: `{uploaded_file.name}` ({uploaded_file.size / 1024:.2f} KB)")

        with st.spinner("Loading RAG model..."):
            RAG = load_models(colpali_model)
            if RAG is None:
                st.stop()

        with st.spinner("Indexing document..."):
            create_rag_index(save_path)

    with col2:
        st.write("### Chat with Document")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Searching document..."):
                try:
                    results = RAG.search(prompt, k=1, return_base64_results=True)
                    if results:
                        image_data = base64.b64decode(results[0].base64)
                        image = Image.open(BytesIO(image_data))
                        temp_img_path = os.path.join(upload_dir, "temp_image.png")
                        image.save(temp_img_path)

                        st.image(image, caption="Relevant Document Section", use_column_width=True)

                        generation_config = {
                            "temperature": temperature,
                            "top_p": 1,
                            "top_k": 32,
                            "max_output_tokens": max_tokens,
                        }

                        model = genai.GenerativeModel(model_name=multi_model_llm, generation_config=generation_config)

                        with open(temp_img_path, "rb") as img_file:
                            image_bytes = img_file.read()

                        response = model.generate_content([prompt, {"mime_type": "image/png", "data": image_bytes}])
                        output = response.text

                        os.remove(temp_img_path)
                    else:
                        output = "No relevant document section found."

                except Exception as e:
                    output = f"Error processing query: {str(e)}"
                    st.error(output)

            st.session_state.messages.append({"role": "assistant", "content": output})

            with st.chat_message("assistant"):
                st.markdown(output)
else:
    st.info("Upload a document to get started.")
