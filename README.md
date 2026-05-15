# Healthcare RAG — Question Answering & Summarization System

A robust Retrieval-Augmented Generation (RAG) system designed for healthcare clinical guidelines. This application allows users to query medical documents and receive evidence-grounded answers with full citations and grounding validation.

## Features

- **Evidence-Grounded QA**: Answers are strictly based on the provided clinical guidelines.
- **Summarization Mode**: Condenses long-form medical text into actionable summaries.
- **Grounding Validation**: Automatically evaluates responses for hallucinations and ensures claims are supported by source passages.
- **Source Citations**: Every answer includes references to the specific documents and sections used.
- **Vector Search**: Uses FAISS and Sentence-Transformers for efficient document retrieval.
- **Gemini Integration**: Powered by Google Gemini for high-quality language generation.

## Project Structure

- `app.py`: Streamlit-based UI for the application.
- `pipeline.py`: Orchestrator for the RAG flow (Ingest -> Retrieve -> Generate -> Validate).
- `ingestion/`: Modules for document loading, chunking, and embedding.
- `retrieval/`: Vector store management using FAISS.
- `generation/`: Prompt engineering and LLM interaction.
- `validation/`: Grounding validation logic.
- `data/sample_docs/`: Sample clinical guideline documents.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd healthcare-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Testing

The project includes a suite of tests for the pipeline:
```bash
pytest tests/
```

## License

MIT
