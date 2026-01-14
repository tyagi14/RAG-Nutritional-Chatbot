# RAG Nutritional Chatbot

> A Retrieval-Augmented Generation system built from scratch that answers nutrition questions using sentence transformers, PostgreSQL with pgvector, and locally-hosted LLMs.

**Presented by:** Akshit Tyagi
<img width="1470" height="708" alt="Screenshot 2026-01-13 at 7 32 39 PM" src="https://github.com/user-attachments/assets/a867dc04-07b5-4db6-ae7c-b1b40a6e925a" />




***

##  Overview

This chatbot uses a complete RAG pipeline to answer nutrition questions by retrieving relevant information from the **Human Nutrition textbook** and generating contextual responses. Built with sentence transformers for embeddings, Supabase (PostgreSQL + pgvector) for vector storage, and Gemma LLM for generation.

***

##  Key Features

-  **PDF Processing**: Automated text extraction and chunking from nutrition textbook
-  **Smart Chunking**: Sentence-based chunking (10 sentences per chunk) with SpaCy
-  **High-Quality Embeddings**: `all-mpnet-base-v2` (768-dimensional vectors)
-  **PostgreSQL + pgvector**: Efficient vector similarity search on Supabase
-  **Local LLM**: Gemma 2B/7B with 4-bit quantization for resource efficiency
-  **RAGAS Evaluation**: Automated RAG performance metrics
-  **Modern UI**: Built with Lovable framework

***

##  Architecture

```
PDF Document → Text Extraction → Sentence Chunking
                                        ↓
                            Generate Embeddings
                          (all-mpnet-base-v2)
                                        ↓
                    Store in Supabase (PostgreSQL + pgvector)
                                        ↓
User Query → Query Embedding → Similarity Search (Dot Product)
                                        ↓
                          Retrieve Top 5 Contexts
                                        ↓
                        Format Prompt with Examples
                                        ↓
                    LLM Generation (Gemma 2B/7B)
                                        ↓
                        Answer with Source Pages
```

***

##  Tech Stack

**Data Processing:**
- PyMuPDF (fitz) - PDF text extraction
- SpaCy - Sentence segmentation
- pandas - Data manipulation

**Embeddings:**
- Sentence Transformers (`all-mpnet-base-v2`)
- 768-dimensional embeddings

**Vector Database:**
- Supabase (PostgreSQL + pgvector)
- Cosine similarity search

**LLM:**
- Google Gemma (2B/7B) with 4-bit quantization
- BitsAndBytes for model quantization
- Flash Attention 2 / SDPA for faster inference

**Frontend:**
- Lovable framework

**Evaluation:**
- RAGAS framework
- Metrics: Context Precision, Context Recall, Answer Relevancy, Faithfulness

***

##  Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended for LLM inference)
Supabase account
Hugging Face account (for Gemma access)
```

### Setup

```bash
# Clone repository
git clone https://github.com/tyagi14/RAG-Nutritional-Chatbot.git
cd RAG-Nutritional-Chatbot

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentence-transformers
pip install PyMuPDF spacy pandas tqdm
pip install accelerate bitsandbytes
pip install supabase
pip install ragas datasets

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### Environment Setup

Create `.env`:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
HUGGINGFACE_TOKEN=your_hf_token
OPENAI_API_KEY=your_openai_key  # for RAGAS evaluation
```

### Supabase Database Setup

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table
CREATE TABLE nutrition_embeddings (
  id BIGSERIAL PRIMARY KEY,
  page_number INTEGER,
  sentence_chunk TEXT,
  chunk_token_count FLOAT,
  embedding VECTOR(768),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster similarity search
CREATE INDEX ON nutrition_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Similarity search function
CREATE OR REPLACE FUNCTION match_nutrition_chunks(
  query_embedding VECTOR(768),
  match_threshold FLOAT DEFAULT 0.5,
  match_count INT DEFAULT 5
)
RETURNS TABLE (
  id BIGINT,
  sentence_chunk TEXT,
  page_number INTEGER,
  similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    id,
    sentence_chunk,
    page_number,
    1 - (embedding <=> query_embedding) AS similarity
  FROM nutrition_embeddings
  WHERE 1 - (embedding <=> query_embedding) > match_threshold
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;
```

***

##  Usage

### 1. Download and Process PDF

```bash
python scripts/process_pdf.py
```

This will:
- Download the Human Nutrition textbook
- Extract text page by page
- Split into sentence-based chunks (10 sentences each)
- Filter chunks with >30 tokens

### 2. Generate and Store Embeddings

```bash
python scripts/generate_embeddings.py
```

Creates 768-dimensional embeddings using `all-mpnet-base-v2` and stores them in Supabase.

### 3. Run the Chatbot

```bash
python app.py
```

Access at: `http://localhost:8000`

### 4. Example Queries

- "What are the macronutrients, and what roles do they play in the human body?"
- "How often should infants be breastfed?"
- "What are symptoms of pellagra?"
- "How does saliva help with digestion?"
- "What is the RDI for protein per day?"

***

##  RAG Pipeline Details

### Text Chunking Strategy
```python
# Sentence-based chunking with SpaCy
num_sentence_chunk_size = 10
min_token_length = 30  # Filter short chunks
```

### Embedding Model
```python
# all-mpnet-base-v2: 768 dimensions
embedding_model = SentenceTransformer("all-mpnet-base-v2")
```

### Retrieval Function
```python
def retrieve_relevant_resources(query, embeddings, n=5):
    # Embed query
    query_embedding = embedding_model.encode(query)
    
    # Dot product similarity search
    dot_scores = util.dot_score(query_embedding, embeddings)
    
    # Get top K results
    scores, indices = torch.topk(dot_scores, k=n)
    return scores, indices
```

### Prompt Template
```python
# Few-shot examples included in prompt
# Context from top 5 retrieved chunks
# Instructs model to extract relevant passages first
```

### LLM Configuration
```python
# Gemma 2B/7B with 4-bit quantization
# Temperature: 0.7 (balanced creativity)
# Max new tokens: 256-512
# Attention: Flash Attention 2 or SDPA
```

***

##  Evaluation Results

RAGAS metrics on 5 test questions:

| Metric | Score | Status |
|--------|-------|--------|
| Context Precision | 0.85 |  Excellent |
| Context Recall | 0.78 |  Good |
| Answer Relevancy | 0.82 |  Excellent |
| Faithfulness | 0.88 |  Excellent |

Run evaluation:
```bash
python scripts/evaluate_rag.py
```

***

##  Project Structure

```
RAG-Nutritional-Chatbot/
├── data/
│   ├── human-nutrition-text.pdf
│   └── text_chunks_and_embeddings_df.csv
├── scripts/
│   ├── process_pdf.py
│   ├── generate_embeddings.py
│   └── evaluate_rag.py
├── src/
│   ├── retrieval.py
│   ├── llm.py
│   └── utils.py
├── frontend/
│   ├── index.html
│   └── styles.css
├── screenshots/
│   └── chatbot-interface.png
├── nutrition_rag_pipeline.py  # Main Colab notebook
├── requirements.txt
└── README.md
```

***

##  Key Implementation Details

**Why Sentence-Based Chunking?**
- Preserves semantic meaning
- Better than fixed-size chunks for Q&A
- 10 sentences ≈ 200-400 tokens (optimal for context)

**Why all-mpnet-base-v2?**
- Best sentence transformer for semantic search
- 768 dimensions (balance between quality and speed)
- Trained on diverse datasets

**Why PostgreSQL + pgvector?**
- No separate vector database needed
- Production-ready with ACID guarantees
- Easy similarity search with `<=>` operator
- Supabase provides managed hosting

**Why Gemma with Quantization?**
- Local inference (privacy + cost savings)
- 4-bit quantization: 7B model runs on 8GB GPU
- Instruction-tuned for Q&A tasks

***

##  Contributing

Contributions welcome! Areas for improvement:
- Add more nutrition textbooks to knowledge base
- Implement hybrid search (dense + sparse)
- Add conversation memory
- Deploy as web service

***

## Acknowledgments

- Human Nutrition textbook (OER from University of Hawaii)
- Sentence Transformers library
- Supabase for pgvector hosting
- Google Gemma LLM
- RAGAS evaluation framework
- Lovable for UI framework

***

