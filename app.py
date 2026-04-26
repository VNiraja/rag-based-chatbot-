"""
╔══════════════════════════════════════════════════════════════════════╗
║           RAG DOCUMENT Q&A  —  app.py                               ║
║   LangChain 1.x  +  FAISS  +  Groq  +  Streamlit                   ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS RAG? (Retrieval-Augmented Generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: LLMs like LLaMA don't know about YOUR documents.
         If you just ask "what does my contract say?", it will guess.

Solution (RAG):
  1. We pre-read your documents and cut them into small pieces (chunks)
  2. Each chunk is converted to a vector (a list of numbers = its "meaning")
  3. All vectors are stored in FAISS (a fast similarity search database)
  4. When you ask a question:
       a) Convert question to a vector
       b) FAISS finds the 4 chunks most similar to that vector
       c) We give THOSE chunks + your question to the LLM
       d) LLM answers using only real content from your document

  Result: Accurate, grounded answers — no hallucination!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL PIPELINE FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Your PDF/DOCX/TXT]
        │
        ▼
  [Loader] ──── reads raw text from file
        │
        ▼
  [Text Splitter] ──── cuts into 800-char overlapping chunks
        │
        ▼
  [HuggingFace Embeddings] ──── converts each chunk to a 384-number vector
        │
        ▼
  [FAISS Vector Store] ──── stores all vectors for fast search
        │
        ▼  (at query time)
  [User Question] ──── also converted to a vector
        │
        ▼
  [FAISS Retriever] ──── finds 4 most similar chunk-vectors
        │
        ▼
  [Prompt Template] ──── fills in: context = chunks, question = your question
        │
        ▼
  [Groq LLM] ──── reads the filled prompt and generates an answer
        │
        ▼
  [Answer + Source citations] ──── shown in Streamlit chat UI

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: CORRECT pip PACKAGE NAMES (vs Python import names)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pip install NAME          | from import_name import ...
  ──────────────────────────┼──────────────────────────────────────
  langchain                 | from langchain_core.* import ...
  langchain-community       | from langchain_community.* import ...
  langchain-groq            | from langchain_groq import ...
  langchain-huggingface     | from langchain_huggingface import ...
  langchain-text-splitters  | from langchain_text_splitters import ...
  faiss-cpu                 | from langchain_community.vectorstores import FAISS
  sentence-transformers     | (used internally by HuggingFaceEmbeddings)
  pymupdf                   | from langchain_community.document_loaders import PyMuPDFLoader
  docx2txt                  | from langchain_community.document_loaders import Docx2txtLoader

  ❌ WRONG (causes your error):
     pip install langchain_community.document_loaders   ← this is an import PATH, not a package

  ✅ CORRECT:
     pip install langchain-community                    ← install the package
     from langchain_community.document_loaders import PyMuPDFLoader  ← then import from it
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS — each one explained
# ══════════════════════════════════════════════════════════════════════════════

import os
import tempfile
import streamlit as st


# ── 1. DOCUMENT LOADERS ───────────────────────────────────────────────────────
# These classes know how to open specific file formats and extract plain text.
# pip package: langchain-community
from langchain_community.document_loaders import (
    PyMuPDFLoader,   # Opens PDFs using PyMuPDF library. Returns one Document per page.
    Docx2txtLoader,  # Opens .docx Word files. Extracts all text as a single Document.
    TextLoader,      # Opens plain .txt files. Returns file content as one Document.
)
# A "Document" in LangChain = object with:
#   .page_content  →  the actual text string
#   .metadata      →  dict like {"source": "myfile.pdf", "page": 3}


# ── 2. TEXT SPLITTER ──────────────────────────────────────────────────────────
# pip package: langchain-text-splitters  (separate package in LangChain 1.x!)
# OLD (LangChain 0.x, BROKEN in 1.x):  from langchain.text_splitter import ...
# NEW (LangChain 1.x, CORRECT):        from langchain_text_splitters import ...
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Why split documents?
#   - LLMs have token limits (can't read 100 pages at once)
#   - We only want to send RELEVANT parts to the LLM, not everything
#   - Smaller chunks = more precise retrieval
#
# RecursiveCharacterTextSplitter is smart: it tries to split on:
#   "\n\n" (paragraphs) first → then "\n" (lines) → then ". " (sentences)
#   → then " " (words) → then "" (characters) as last resort
# This keeps chunks at natural boundaries (end of sentence/paragraph)


# ── 3. EMBEDDINGS ─────────────────────────────────────────────────────────────
# pip package: langchain-huggingface
# OLD (deprecated): from langchain_community.embeddings import HuggingFaceEmbeddings
# NEW (correct):    from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# What is an embedding?
#   Text → a list of numbers (vector) that captures its MEANING.
#   "The dog ran fast" → [0.23, -0.11, 0.87, ..., 0.44]  (384 numbers)
#   "The puppy sprinted" → [0.24, -0.10, 0.85, ..., 0.43]  (very similar!)
#   "Stock market crash" → [0.91, 0.33, -0.55, ..., -0.12] (very different)
#
# We use "all-MiniLM-L6-v2" — a free, fast, 80MB model that runs on CPU.
# No API key needed. Downloads once from HuggingFace Hub, then cached locally.


# ── 4. VECTOR STORE (FAISS) ───────────────────────────────────────────────────
# pip package: faiss-cpu  (the actual FAISS library)
#              langchain-community  (the LangChain wrapper around it)
from langchain_community.vectorstores import FAISS
# What is FAISS?
#   Facebook AI Similarity Search — a library that stores vectors and finds
#   the most "similar" ones to a query vector EXTREMELY fast (even with millions).
#
# How similarity works:
#   Each chunk has a vector. When you ask a question, it also gets a vector.
#   FAISS finds the N chunk-vectors that are mathematically closest to the
#   question-vector. "Closest" = most similar in meaning.
#
# Example:
#   You ask: "What is the refund policy?"
#   Your question vector will be closest to the chunk that says
#   "Refunds are processed within 7 business days..." — even if you used
#   different words like "policy" vs "rules".


# ── 5. LLM (GROQ) ─────────────────────────────────────────────────────────────
# pip package: langchain-groq
# OLD (broken in 1.x): from langchain.chat_models import ChatGroq
# NEW (correct):       from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
# Groq = a free AI API that runs open-source models at ~300 tokens/second.
# We use LLaMA-3.3-70B — a 70 billion parameter model, very smart.
# Free tier: 14,400 requests/day. Get key at https://console.groq.com


# ── 6. PROMPT TEMPLATE ────────────────────────────────────────────────────────
# pip package: langchain (core is bundled here)
# OLD (broken): from langchain.prompts import PromptTemplate
# NEW (correct): from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# A PromptTemplate is a string with placeholders like {context} and {question}.
# We fill in the placeholders before sending to the LLM.
# This lets us inject retrieved document chunks as context.


# ── 7. LCEL PIPELINE COMPONENTS ───────────────────────────────────────────────
# OLD (broken in 1.x): from langchain.chains import RetrievalQA
# NEW (correct, LangChain Expression Language):
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# LangChain 1.x replaced the old "chain" classes with LCEL (LangChain Expression Language).
# Instead of RetrievalQA.from_chain_type(...), we chain steps with the | (pipe) operator:
#   retriever | prompt | llm | output_parser
# This is more flexible, transparent, and easier to debug.


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# The HuggingFace embedding model to use (free, runs locally on CPU)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The Groq model. LLaMA 3.3 70B = best free option for quality answers.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Chunk size in characters. 800 chars ≈ 150-200 words.
# Smaller = more precise retrieval but less context per chunk.
# Larger = more context per chunk but less precise retrieval.
CHUNK_SIZE = 800

# Overlap between consecutive chunks in characters.
# Why overlap? If a sentence spans the boundary of two chunks,
# overlap ensures it's fully captured in at least one chunk.
CHUNK_OVERLAP = 150

# Number of chunks to retrieve per question.
# More chunks = more context for the LLM, but prompt gets longer.
TOP_K = 4


# ══════════════════════════════════════════════════════════════════════════════
# THE RAG PROMPT
# This is the exact text sent to the LLM for every question.
# ══════════════════════════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE = """You are a helpful and precise document assistant.
Your job is to answer questions based ONLY on the document context provided below.

Rules:
- Only use information from the context. Do not use outside knowledge.
- If the answer is not in the context, say exactly: "I couldn't find that in the uploaded documents."
- Be concise and specific. Quote the document when helpful.
- Mention which part of the document your answer comes from if possible.

━━━━━━━━━━━━━━━━━━━━━
DOCUMENT CONTEXT:
{context}
━━━━━━━━━━━━━━━━━━━━━

QUESTION: {question}

ANSWER:"""

# Create a ChatPromptTemplate from the string above.
# HumanMessage means this entire prompt is sent as the "user" turn.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("human", RAG_PROMPT_TEMPLATE)
])


# ══════════════════════════════════════════════════════════════════════════════
# STEP-BY-STEP FUNCTIONS
# Each function does ONE thing in the pipeline.
# ══════════════════════════════════════════════════════════════════════════════

def load_document(file_path: str, file_name: str) -> list:
    """
    STEP 1: Load a file and return a list of LangChain Document objects.

    Input:  file_path (str) — path to temp file on disk
            file_name (str) — original filename (for metadata/citations)

    Output: list of Document objects, each with .page_content and .metadata

    How it works:
    - We look at the file extension to pick the right loader.
    - PyMuPDFLoader reads PDFs page by page → returns one Document per page.
    - Docx2txtLoader strips Word formatting → returns all text as one Document.
    - TextLoader reads the file as-is → one Document.
    """
    ext = file_name.lower().rsplit(".", 1)[-1]  # get extension: "pdf", "docx", "txt"

    if ext == "pdf":
        loader = PyMuPDFLoader(file_path)
        # PyMuPDF = "MuPDF" library binding for Python.
        # It extracts text from each PDF page, preserving layout better than pdfplumber.
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
        # docx2txt strips .docx XML structure and returns clean plain text.
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
        # Simple file read. encoding="utf-8" handles international characters.
    else:
        raise ValueError(f"Unsupported format: .{ext} — use PDF, DOCX, or TXT")

    documents = loader.load()

    # Add original filename to every document's metadata.
    # This lets us show "Source: report.pdf" in citations later.
    for doc in documents:
        doc.metadata["source"] = file_name

    return documents


def split_into_chunks(documents: list) -> list:
    """
    STEP 2: Split documents into smaller, overlapping chunks.

    Why? A 30-page PDF might be 60,000 characters.
    We can't send all 60,000 chars to the LLM every time.
    Instead we find only the 4 most relevant 800-char pieces.

    Visual example with chunk_size=20, overlap=5:
      Original: "The quick brown fox jumps over the lazy dog"
      Chunk 1:  "The quick brown fox j"
      Chunk 2:  "n fox jumps over the "   ← starts 5 chars before chunk 1 ends
      Chunk 3:  "r the lazy dog"

    The overlap (5 chars) ensures "fox jumps" isn't lost at a boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,         # target chunk length in characters
        chunk_overlap=CHUNK_OVERLAP,   # overlap between adjacent chunks
        separators=["\n\n", "\n", ". ", " ", ""],
        # Tries these separators in order.
        # Prefers splitting on blank lines (paragraphs), then single newlines,
        # then sentence endings, then spaces, then characters as last resort.
    )

    chunks = splitter.split_documents(documents)
    # Each chunk is still a Document object with:
    #   .page_content  → the 800-char text slice
    #   .metadata      → inherited from parent + chunk location info

    return chunks


@st.cache_resource(show_spinner="⚙️ Loading embedding model (one-time, ~80MB)...")
def load_embedding_model():
    """
    STEP 3a: Load the embedding model ONCE and cache it permanently.

    @st.cache_resource — Streamlit decorator that:
    - Runs this function only ONCE per app session (not on every page rerun)
    - Stores the result in memory
    - Returns the cached result on subsequent calls
    This is critical because loading an 80MB model takes 5-10 seconds.

    What is the model?
    - "all-MiniLM-L6-v2" by SentenceTransformers / HuggingFace
    - "MiniLM" = Mini Language Model (compact version)
    - "L6" = 6 transformer layers (fast, runs on CPU)
    - "v2" = version 2
    - Output: 384-dimensional vectors (384 numbers per text)
    - Quality: very good for English semantic similarity
    - Size: ~80MB, downloads once, then cached at ~/.cache/huggingface/
    - Cost: FREE, runs entirely on your machine, no API key needed
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},          # use CPU (works everywhere)
        encode_kwargs={"normalize_embeddings": True},
        # normalize_embeddings=True makes all vectors unit length (length=1).
        # This means cosine similarity = dot product, which is faster.
        # It also makes scores more comparable across different texts.
    )


def create_faiss_index(chunks: list, embeddings) -> FAISS:
    """
    STEP 3b: Convert all chunks to vectors and store them in FAISS.

    Input:  chunks (list of Document objects)
            embeddings (HuggingFaceEmbeddings model)

    Output: FAISS vector store (in-memory database of all chunk vectors)

    What happens internally:
    1. For each chunk, call embeddings.embed_query(chunk.page_content)
       → returns a list of 384 floats
    2. All these vectors are stored in a FAISS "flat" index
       (flat = exact search, no approximation, perfect for < 100k chunks)
    3. The original text is stored alongside each vector
       so we can return the text when retrieval finds a match

    Memory estimate: 384 floats × 4 bytes × N chunks
                     For 100 chunks: ~150KB — negligible
    """
    # FAISS.from_documents() does the embedding + indexing in one call
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vectorstore


def build_rag_chain(vectorstore: FAISS, groq_api_key: str):
    """
    STEP 4: Build the complete RAG pipeline using LCEL (LangChain Expression Language).

    OLD WAY (LangChain 0.x — broken in 1.x):
      chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, ...)

    NEW WAY (LangChain 1.x — LCEL with | pipe operator):
      chain = setup | prompt | llm | parser

    The | operator chains steps: output of left becomes input of right.
    This is like Unix pipes: cat file.txt | grep "error" | head -10

    Our pipeline:
      1. RunnableParallel gets question, retrieves context simultaneously
      2. RAG_PROMPT fills in {context} and {question}
      3. ChatGroq sends filled prompt to LLaMA on Groq servers
      4. StrOutputParser extracts the text string from the LLM response object
    """

    # ── The LLM ────────────────────────────────────────────────────────────
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=GROQ_MODEL,
        temperature=0.0,
        # temperature controls randomness:
        #   0.0 = fully deterministic (same question → same answer, factual)
        #   1.0 = very creative/random
        # For document Q&A, we want 0.0 — no hallucination, just facts.
        max_tokens=1024,    # max length of the answer
    )

    # ── The Retriever ──────────────────────────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        # "similarity" = cosine similarity between question vector and chunk vectors.
        # Alternative: "mmr" (Maximal Marginal Relevance) = diverse + relevant chunks.
        search_kwargs={"k": TOP_K},
        # k=4 means: retrieve the 4 most similar chunks for each question
    )

    # ── Helper: format retrieved docs into a single string ─────────────────
    def format_docs(docs):
        """
        Join retrieved Document objects into one big context string.
        Each chunk is separated by a divider so the LLM can tell them apart.
        Also shows which file each chunk came from.
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "document")
            parts.append(f"[Chunk {i} from '{source}']:\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # ── Build the LCEL chain ───────────────────────────────────────────────
    #
    # Step-by-step what this chain does when you call chain.invoke(question):
    #
    # 1. RunnableParallel runs TWO things simultaneously:
    #       "context": retriever | format_docs
    #           → embeds question → FAISS finds top 4 chunks → format_docs joins them
    #       "question": RunnablePassthrough()
    #           → just passes the question string through unchanged
    #
    # 2. RAG_PROMPT takes {"context": "...", "question": "..."}
    #       → fills the template → creates a ChatMessage ready for the LLM
    #
    # 3. llm sends the filled prompt to Groq API
    #       → returns an AIMessage object with .content = answer text
    #
    # 4. StrOutputParser extracts .content from the AIMessage
    #       → returns a plain Python string
    #
    chain = (
        RunnableParallel({
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Also build a retriever reference so we can show sources in the UI
    return chain, retriever


def process_files(uploaded_files: list, embeddings) -> tuple:
    """
    STEP 5: Orchestrate the full document ingestion pipeline.

    Processes all uploaded files, returns (vectorstore, chunk_count, file_names).
    Returns (None, 0, []) if nothing could be processed.

    Why use tempfiles?
    Streamlit gives us file content as in-memory bytes (UploadedFile object).
    But LangChain loaders need a real file path on disk (they open the file themselves).
    So we write bytes to a temp file, pass the path to the loader, then delete the temp file.
    """
    all_chunks = []
    loaded_names = []

    for uploaded_file in uploaded_files:
        # Get the file extension to create a properly named temp file
        suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]

        # tempfile.NamedTemporaryFile creates a real file on disk
        # delete=False so it persists after the with block (loader needs it)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())   # write bytes from Streamlit to disk
            tmp_path = tmp.name               # remember the path

        try:
            # Load the file
            docs = load_document(tmp_path, uploaded_file.name)
            # Split into chunks
            chunks = split_into_chunks(docs)
            all_chunks.extend(chunks)
            loaded_names.append(uploaded_file.name)

            st.sidebar.caption(f"✓ {uploaded_file.name}: {len(docs)} pages, {len(chunks)} chunks")

        except Exception as e:
            st.sidebar.warning(f"⚠️ {uploaded_file.name}: {e}")

        finally:
            # Always clean up the temp file, even if loading failed
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not all_chunks:
        return None, 0, []

    # Build the FAISS index from all chunks across all files
    vectorstore = create_faiss_index(all_chunks, embeddings)
    return vectorstore, len(all_chunks), loaded_names


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
# Streamlit works by re-running the ENTIRE script from top to bottom every time
# the user interacts (clicks a button, types text, uploads a file).
# st.session_state is a dict that PERSISTS between these reruns.
# Without session_state, all variables would reset on every interaction!

st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.chat-wrap { max-width: 820px; margin: 0 auto; }

.user-row { display: flex; justify-content: flex-end; margin: 12px 0 4px; }
.user-bubble {
    background: #312e81;
    color: #e0e7ff;
    border-radius: 18px 18px 4px 18px;
    padding: 11px 17px;
    max-width: 72%;
    font-size: 14.5px;
    line-height: 1.65;
}
.bot-row { display: flex; justify-content: flex-start; margin: 4px 0 12px; }
.bot-bubble {
    background: #f5f3ff;
    color: #1e1b4b;
    border: 1px solid #ddd6fe;
    border-radius: 4px 18px 18px 18px;
    padding: 13px 17px;
    max-width: 82%;
    font-size: 14.5px;
    line-height: 1.75;
}
.lbl {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 3px;
    padding-left: 2px;
}
.src-tag {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    margin: 2px 2px 0;
}
.empty-state {
    text-align: center;
    padding: 70px 20px 50px;
    color: #a78bfa;
}
.step-box {
    border-left: 3px solid #7c3aed;
    padding: 8px 14px;
    margin: 6px 0;
    background: #faf5ff;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #4c1d95;
}
.pill {
    display: inline-block;
    background: #ede9fe;
    color: #6d28d9;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state — initialize once ──────────────────────────────────────────
# These are the "global variables" that persist between Streamlit reruns.
for key, default in {
    "chat_history":  [],    # list of {"role": "user"/"bot", "content": str, "sources": list}
    "vectorstore":   None,  # FAISS index (built from uploaded files)
    "rag_chain":     None,  # the LCEL chain (retriever | prompt | llm | parser)
    "retriever":     None,  # separate retriever reference (for source lookup)
    "loaded_files":  [],    # list of successfully loaded file names
    "chunk_count":   0,     # total number of chunks indexed
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("*Ask questions from your documents*")
    st.markdown("---")

    # ── API Key input ─────────────────────────────────────────────────────
    st.markdown("### 🔑 Groq API Key")
    st.caption("Free key → [console.groq.com](https://console.groq.com)")

    groq_key = st.text_input(
        "Groq key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        label_visibility="collapsed",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("✅ Key ready")
    else:
        st.info("Enter key above to enable Q&A")

    st.markdown("---")

    # ── File uploader ─────────────────────────────────────────────────────
    st.markdown("### 📂 Upload Documents")
    st.caption("PDF · DOCX · TXT — multiple files OK")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Process files when new ones are uploaded
    if uploaded_files and groq_key:
        current_names = sorted([f.name for f in uploaded_files])
        already_loaded = sorted(st.session_state.loaded_files)

        if current_names != already_loaded:
            # New files detected — run the full ingestion pipeline
            with st.spinner("📖 Loading, splitting, embedding..."):
                embeddings = load_embedding_model()
                vs, n_chunks, names = process_files(uploaded_files, embeddings)

            if vs:
                # Store everything in session state
                chain, retriever = build_rag_chain(vs, groq_key)
                st.session_state.vectorstore  = vs
                st.session_state.rag_chain    = chain
                st.session_state.retriever    = retriever
                st.session_state.loaded_files = names
                st.session_state.chunk_count  = n_chunks
                st.session_state.chat_history = []  # clear old chat on new docs
                st.success(f"✅ {len(names)} file(s) indexed!")
            else:
                st.error("❌ Could not process files")

    elif uploaded_files and not groq_key:
        st.warning("Enter Groq API key first")

    # ── Index stats ───────────────────────────────────────────────────────
    if st.session_state.loaded_files:
        st.markdown("---")
        st.markdown("### 📊 Index Stats")
        for fname in st.session_state.loaded_files:
            st.markdown(f'<span class="pill">📄 {fname}</span>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="pill">🧩 {st.session_state.chunk_count} chunks stored</span>'
            f'<span class="pill">🔍 Retrieves top {TOP_K}</span>',
            unsafe_allow_html=True,
        )

    # ── Controls ──────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            for k in ["chat_history","vectorstore","rag_chain","retriever","loaded_files","chunk_count"]:
                st.session_state[k] = [] if k in ("chat_history","loaded_files") else (None if k != "chunk_count" else 0)
            st.rerun()

    # ── Pipeline explainer ────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔬 How RAG works (click)"):
        st.markdown("""
**INDEXING** (when you upload):
""")
        for step in [
            "Load file → extract raw text",
            "Split text into 800-char overlapping chunks",
            "Embed each chunk → 384-number vector (MiniLM)",
            "Store all vectors in FAISS index",
        ]:
            st.markdown(f'<div class="step-box">⬇️ {step}</div>', unsafe_allow_html=True)

        st.markdown("**QUERYING** (when you ask):")
        for step in [
            "Embed your question → vector",
            "FAISS finds 4 most similar chunk-vectors",
            "Chunks + question → filled prompt",
            "Groq LLaMA reads prompt → generates answer",
        ]:
            st.markdown(f'<div class="step-box">⬇️ {step}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 DocMind")
if st.session_state.loaded_files:
    files_str = "  ·  ".join(st.session_state.loaded_files)
    st.caption(f"📄 {files_str}  ·  🤖 {GROQ_MODEL}")
else:
    st.caption("Upload documents in the sidebar to begin")

st.markdown("---")

# ── Render chat history ───────────────────────────────────────────────────────
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
<div class="empty-state">
    <div style="font-size:54px; margin-bottom:18px;">📄</div>
    <div style="font-size:20px; font-weight:500; color:#4c1d95;">
        Upload a document, then start asking
    </div>
    <div style="font-size:13.5px; margin-top:10px; color:#7c3aed; opacity:0.75;">
        Summaries · definitions · specific facts · comparisons — anything in your doc
    </div>
</div>""", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="user-row">
  <div>
    <div class="lbl" style="text-align:right">You</div>
    <div class="user-bubble">{msg["content"]}</div>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        # Format answer: newlines → <br>
        answer_html = msg["content"].replace("\n", "<br>")
        # Build source tags
        src_html = ""
        if msg.get("sources"):
            unique = list({s.metadata.get("source","doc") for s in msg["sources"]})
            tags = " ".join(f'<span class="src-tag">📌 {s}</span>' for s in unique)
            src_html = f'<div style="margin-top:10px">{tags}</div>'

        st.markdown(f"""
<div class="bot-row">
  <div>
    <div class="lbl">🧠 DocMind</div>
    <div class="bot-bubble">{answer_html}{src_html}</div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Question input ────────────────────────────────────────────────────────────
st.markdown("---")
with st.form("q_form", clear_on_submit=True):
    col_q, col_btn = st.columns([6, 1])
    with col_q:
        question = st.text_input(
            "Question",
            placeholder="Ask anything about your documents...",
            label_visibility="collapsed",
        )
    with col_btn:
        ask = st.form_submit_button("Ask ↗", use_container_width=True)

# ── Handle submission ─────────────────────────────────────────────────────────
if ask and question.strip():
    if not st.session_state.rag_chain:
        st.error("❌ Upload documents and set Groq API key first.")
    else:
        # Save user question
        st.session_state.chat_history.append({
            "role": "user",
            "content": question.strip(),
            "sources": [],
        })

        # Run the RAG pipeline
        with st.spinner("🔍 Searching your documents..."):
            try:
                # chain.invoke() runs the full pipeline:
                # question → embed → FAISS retrieve → fill prompt → Groq LLM → answer
                answer = st.session_state.rag_chain.invoke(question.strip())

                # Get source documents separately for citations in the UI
                source_docs = st.session_state.retriever.invoke(question.strip())

            except Exception as e:
                answer = f"❌ Error: {str(e)}"
                source_docs = []

        # Save bot answer with sources
        st.session_state.chat_history.append({
            "role":    "bot",
            "content": answer,
            "sources": source_docs,
        })

        # Rerun = refresh the UI to show new messages
        st.rerun()