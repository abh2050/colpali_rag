# Colpali-Based Multimodal RAG App
![](https://assets.zilliz.com/large_Col_Pali_Visual_Retriever_with_Col_BERT_strategy_d258c79a05.png)

## Overview
This Streamlit application implements a **Multimodal Retrieval-Augmented Generation (RAG) system** using the Colpali model. The app allows users to upload documents, index them for retrieval, and interact with the content using Large Language Models (LLMs) such as Gemini 2.0.

## Features
- **Document Upload & Processing**: Supports PDF documents for indexing and retrieval.
- **RAG Model Integration**: Uses `byaldi` for retrieval-augmented generation.
- **Multi-Model LLM Support**: Select between `gemini-2.0-flash` and `gemini-1.5-pro`.
- **Configurable Settings**: Adjust temperature, max tokens, batch size, and memory usage.
- **Image-Based Search**: Extracts images and text from documents to generate AI responses.
- **Streamlit-Based UI**: Interactive web interface with a sidebar for customization.

## ColPali: Efficient Document Retrieval with Vision Language Models
https://colab.research.google.com/github/abh2050/colpali_rag/blob/main/colpali.ipynb

### Abstract:
Documents are visually rich structures that convey information through text, as well as tables, figures, page layouts, or fonts. While modern document retrieval systems exhibit strong performance on query-to-text matching, they struggle to exploit visual cues efficiently, hindering their performance on practical document retrieval applications such as Retrieval Augmented Generation. To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark ViDoRe, composed of various page-level retrieving tasks spanning multiple domains, languages, and settings. The inherent shortcomings of modern systems motivate the introduction of a new retrieval model architecture, ColPali, which leverages the document understanding capabilities of recent Vision Language Models to produce high-quality contextualized embeddings solely from images of document pages. Combined with a late interaction matching mechanism, ColPali largely outperforms modern document retrieval pipelines while being drastically faster and end-to-end trainable.

[Read the full paper](https://arxiv.org/abs/2407.01449)

### Distinctions from Traditional RAG Systems:
- **Visual Context Utilization:** Unlike traditional RAG systems that primarily rely on extracted text, ColPali processes entire document pages as images, capturing both textual content and visual elements like layouts, tables, and figures. This holistic approach enables a more comprehensive understanding of the document's information.
- **Simplified Processing Pipeline:** By eliminating the need for OCR and complex text extraction processes, ColPali streamlines the document indexing workflow, resulting in faster and more efficient retrieval operations.
- **Enhanced Retrieval Performance:** Leveraging Vision Language Models (VLMs), ColPali generates high-quality embeddings that improve retrieval accuracy, especially in scenarios where visual context is crucial for understanding the content.

In summary, ColPali's innovative use of VLMs for direct image-based document processing sets it apart from traditional text-centric RAG systems, offering improved efficiency and effectiveness in retrieving information from visually rich documents.

## Installation
Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install streamlit huggingface_hub byaldi transformers Pillow torch python-dotenv google-generativeai poppler-utils
```

For **Windows** users, install `poppler-utils` manually:
1. Download from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
2. Extract the files
3. Add the `bin` directory to your system `PATH`
4. Restart your computer

## Usage
1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <repo-url>
   cd <project-folder>
   ```
2. Create a `.env` file in the project root and add your Gemini API key:
   ```plaintext
   GEMINI_API_KEY=your-api-key-here
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Upload a document and start interacting with it.

## Configuration Options
- **Model Selection**: Choose between `vidore/colpali` or `vidore/colpali-v1.2`.
- **LLM Settings**: Adjust temperature (0.0 - 1.0) and max output tokens (300 - 4096).
- **Performance Optimization**: Enable low-memory mode and configure batch size (1-8).

## Architecture
The application follows this workflow:
1. **Document Upload**: Users upload a PDF file.
2. **Index Creation**: The document is processed and indexed for retrieval.
3. **Query Processing**: User inputs a query; relevant document sections are retrieved.
4. **Response Generation**: The retrieved content is used by Gemini LLM to generate an answer.
5. **Image Handling**: If relevant images are found, they are displayed along with the response.

## Dependencies
- `streamlit` - Web UI framework
- `huggingface_hub` - Model repository access
- `byaldi` - RAG model for multimodal retrieval
- `transformers` - Hugging Face library for LLMs
- `Pillow` - Image processing
- `torch` - Deep learning framework
- `python-dotenv` - Environment variable management
- `google-generativeai` - Gemini model integration
- `poppler-utils` - PDF processing (required for text/image extraction)

## Troubleshooting
- **Gemini API Key Not Found**: Ensure `.env` file contains `GEMINI_API_KEY`.
- **Poppler Not Installed**: Follow the installation instructions above.
- **Model Load Failure**: Ensure `byaldi` and `transformers` are installed.
- **High Memory Usage**: Enable *Low Memory Mode* in the sidebar.

## License
This project is released under the MIT License.

