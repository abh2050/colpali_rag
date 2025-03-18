import streamlit as st
import os
import base64
import platform
import subprocess
from huggingface_hub import notebook_login
from PIL import Image
from io import BytesIO
import re
import google.generativeai as genai
from dotenv import load_dotenv
import sys
import shutil

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

# Check for poppler installation
def is_poppler_installed():
    # Check if pdftoppm is available (part of poppler)
    if platform.system() == "Windows":
        return shutil.which("pdftoppm.exe") is not None
    else:
        return shutil.which("pdftoppm") is not None

# Define the upload directory
upload_dir = "./doc"
os.makedirs(upload_dir, exist_ok=True)

st.title("Colpali Based Multimodal RAG App")
st.sidebar.info("Using CPU for computation")

# Check for Poppler before proceeding
if not is_poppler_installed():
    st.error("Poppler is required but not installed. PDF processing will not work.")
    
    # Show installation instructions based on OS
    st.subheader("Installation Instructions")
    
    if platform.system() == "Darwin":  # macOS
        st.code("brew install poppler", language="bash")
        st.write("After installing, restart this application.")
    elif platform.system() == "Linux":
        st.code("sudo apt-get install poppler-utils", language="bash")
        st.write("After installing, restart this application.")
    elif platform.system() == "Windows":
        st.write("""
        1. Download Poppler from [here](http://blog.alivate.com.au/poppler-windows/)
        2. Extract the downloaded file
        3. Add the `bin` folder to your PATH environment variable
        4. Restart your computer
        5. Rerun this application
        """)
    
    st.stop()

# Create sidebar for configuration options
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
    
    # LLM generation settings
    st.subheader("LLM Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.1,
                           help="Higher = more creative, Lower = more deterministic")
    max_tokens = st.slider("Max Output Tokens", min_value=300, max_value=4096, value=2000, step=100,
                          help="Maximum length of generated response")
    
    # Performance options
    st.subheader("Performance Options")
    use_low_memory = st.checkbox("Enable Low Memory Mode", value=True, 
                                help="Reduces memory usage but may be slower")
    batch_size = st.slider("Batch Size", min_value=1, max_value=8, value=2,
                            help="Lower for less memory usage, higher for more speed")

    uploaded_file = st.file_uploader("Choose a Document", type=["pdf"])

# Main content layout
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("### Uploaded Document")
        save_path = os.path.join(upload_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved: {uploaded_file.name}")

        @st.cache_resource
        def load_models(colpali_model):
            try:
                RAG = RAGMultiModalModel.from_pretrained(
                    colpali_model,
                    verbose=10,
                    device="cpu"
                )
                return RAG
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.error("Try using a smaller model or enabling low memory mode.")
                return None

        with st.spinner("Loading RAG model..."):
            RAG = load_models(colpali_model)
            if RAG is None:
                st.stop()

        @st.cache_data
        def create_rag_index(image_path):
            try:
                # Double-check poppler is available before indexing
                if not is_poppler_installed():
                    st.error("Poppler is required for PDF processing but not found.")
                    st.stop()
                    
                RAG.index(
                    input_path=image_path,
                    index_name="image_index",
                    store_collection_with_index=True,
                    overwrite=True
                )
            except Exception as e:
                st.error(f"Error creating index: {str(e)}")
                st.stop()

        with st.spinner("Creating document index..."):
            create_rag_index(save_path)

    with col2:
        text_query = st.text_input("Enter your text query")
        
        if st.button("Search and Extract Text"):
            if text_query:
                with st.spinner("Processing your query..."):
                    results = RAG.search(text_query, k=1, return_base64_results=True)
                    image_data = base64.b64decode(results[0].base64)
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption="Result Image", use_column_width=True)

                    if multi_model_llm.startswith("gemini"):
                        temp_img_path = os.path.join(upload_dir, "temp_image.png")
                        image.save(temp_img_path)

                        generation_config = {
                            "temperature": temperature,
                            "top_p": 1,
                            "top_k": 32,
                            "max_output_tokens": max_tokens,
                        }

                        model = genai.GenerativeModel(
                            model_name=multi_model_llm,
                            generation_config=generation_config
                        )

                        with open(temp_img_path, "rb") as img_file:
                            image_bytes = img_file.read()

                        try:
                            response = model.generate_content(
                                [
                                    text_query,
                                    {"mime_type": "image/png", "data": image_bytes}
                                ]
                            )
                            output = response.text
                        except Exception as e:
                            output = f"Error generating response: {str(e)}"
                            st.error(output)

                        os.remove(temp_img_path)
                    else:
                        output = f"Model {multi_model_llm} integration not yet implemented."

                    st.subheader("Query with LLM Model")
                    st.markdown(output, unsafe_allow_html=True)
            else:
                st.warning("Please enter a query.")
else:
    st.info("Upload a document to get started.")
