# 🤖 RAG-Based Agentic Q&A Bot

An **Agentic Retrieval-Augmented Generation (RAG) application** that allows users to upload PDF documents and ask questions about their content using natural language.

The application combines **Google Gemini Embeddings**, **ChromaDB**, **Groq LLM**, **LangChain Agents**, **LangGraph**, and **Streamlit** to build an interactive document question-answering system.

---

## 🚀 Project Overview

Large Language Models do not automatically have access to the contents of a user's private documents.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

The application allows users to:

- 📄 Upload one or multiple PDF documents
- 🔍 Process and index document content
- 🧠 Generate vector embeddings using Gemini
- 🗄️ Store embeddings in ChromaDB
- 💬 Ask questions about uploaded documents
- 🤖 Use a LangChain Agent with a custom retrieval tool
- 🔎 Retrieve relevant document chunks using semantic search
- ⚡ Generate answers using a Groq-hosted LLM
- 🧠 Maintain conversational context using LangGraph

---

## 📸 Application Screenshots

Screenshots of the running Streamlit application are available in the project's:

```text
screenshots/
```

The folder contains screenshots demonstrating:

- Application interface
- PDF document upload
- Document-based question answering

---

## 🏗️ Project Structure

The project is organized as follows:

```text
RAG_pdf_based_bot/
│
├── docs_files/
│   └── Uploaded PDF documents
│
├── screenshots/
│   └── Application screenshots
│
├── 5_RAG_based_Agentic_qna_bot.py
│
└── README.md
```

### Main Application

`5_RAG_based_Agentic_qna_bot.py`

This is the main Streamlit application containing the complete RAG and Agentic workflow.

### `docs_files/`

Used for storing uploaded PDF documents during document processing.

### `screenshots/`

Contains screenshots of the application's user interface and workflow.

---

# 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    │ Upload PDF / Query  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │
                         PDF Upload
                               │
                               ▼
                    ┌─────────────────────┐
                    │    PDF Loader       │
                    │ PyPDFDirectoryLoader│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Chunking     │
                    │ RecursiveCharacter  │
                    │ TextSplitter        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Gemini Embeddings   │
                    │ gemini-embedding-001│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ChromaDB       │
                    │    Vector Store     │
                    └──────────┬──────────┘
                               │
                         Semantic Search
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LangChain Agent   │
                    └──────────┬──────────┘
                               │
                         Tool Calling
                               │
                               ▼
                    ┌─────────────────────┐
                    │ retrieve_context()  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Groq LLM       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Final Answer     │
                    └─────────────────────┘

                         LangGraph
                            │
                            ▼
                   Conversation State
```

---

# 🔄 How It Works

## 1. Upload Documents

The user uploads PDF documents through the Streamlit interface.

```text
PDF
 ↓
Document Loader
```

The application uses `PyPDFDirectoryLoader` to load PDF content.

---

## 2. Split Documents

The extracted text is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

Chunking allows the system to retrieve only the relevant portions of a large document.

```text
Large Document
      ↓
Text Extraction
      ↓
Smaller Chunks
```

---

## 3. Generate Embeddings

Each document chunk is converted into a vector representation using Google's:

```text
gemini-embedding-001
```

These embeddings capture the semantic meaning of the document content.

---

## 4. Store Embeddings

The generated embeddings are stored in:

```text
ChromaDB
```

ChromaDB is used as the vector store for semantic similarity search.

---

## 5. Ask a Question

The user enters a natural-language question.

For example:

```text
What are the primary skills of the applicant?
```

---

## 6. Agent Uses the Retrieval Tool

The LangChain Agent has access to a custom retrieval tool:

```text
retrieve_context()
```

The tool performs semantic search against ChromaDB and retrieves the most relevant document chunks.

```text
User Question
      ↓
LangChain Agent
      ↓
retrieve_context()
      ↓
ChromaDB
      ↓
Relevant Document Chunks
```

---

## 7. Generate the Answer

The retrieved context is provided to the Groq-hosted LLM.

```text
User Question
       +
Retrieved Document Context
       +
Conversation Context
       ↓
     Groq LLM
       ↓
   Final Answer
```

---

# 🤖 Why Agentic RAG?

A traditional RAG pipeline generally follows:

```text
Question
   ↓
Retriever
   ↓
Context
   ↓
LLM
   ↓
Answer
```

This project introduces an **Agent layer** with retrieval exposed as a tool:

```text
Question
   ↓
LangChain Agent
   ↓
Tool Calling
   ↓
retrieve_context()
   ↓
ChromaDB
   ↓
Relevant Context
   ↓
Groq LLM
   ↓
Answer
```

This makes the application a **tool-using Agentic RAG system**.

The agent can decide when it needs to use the document retrieval tool before generating the final response.

---

# 🧠 Conversation Memory

The project uses **LangGraph's `InMemorySaver`** to maintain conversation/checkpoint state during the application session.

This allows follow-up questions to use the context of the ongoing conversation.

Example:

```text
User:
What is RAG?

Assistant:
RAG stands for Retrieval-Augmented Generation...

User:
What are its advantages?

Assistant:
The main advantages are...
```

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web UI |
| LangChain | LLM and agent orchestration |
| LangGraph | Agent state/checkpoint management |
| Google Gemini | Document embeddings |
| ChromaDB | Vector database |
| Groq | LLM inference |
| PyPDF | PDF processing |
| RecursiveCharacterTextSplitter | Text chunking |
| python-dotenv | Environment variable management |

---

# 🎯 Key Features

- 📄 Multiple PDF document support
- 🔍 Semantic document search
- 🧠 Gemini vector embeddings
- 🗄️ ChromaDB vector storage
- 🤖 LangChain Agent
- 🔧 Custom retrieval tool
- ⚡ Groq LLM inference
- 🧠 LangGraph conversation state
- 💬 Conversational Q&A
- 🎨 Streamlit user interface
- 🔐 Environment-based API key management

---

# ⚙️ Requirements

Before running the project, make sure you have:

- Python 3.10+
- Git
- Google Gemini API key
- Groq API key

---

# 🔑 Environment Variables

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

Do **not** commit your `.env` file to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/arbaaz65dac/ai-ml-journey.git
```

## 2. Navigate to the Project

```bash
cd ai-ml-journey/04-AI/projects/RAG_pdf_based_bot
```

## 3. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is maintained at the parent `04-AI` level in your repository, install dependencies from that location instead.

---

# ▶️ Run the Application

From the `RAG_pdf_based_bot` directory:

```bash
streamlit run 5_RAG_based_Agentic_qna_bot.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 💬 Example Workflow

Upload a PDF document and ask:

```text
What are the primary skills mentioned in this document?
```

The application processes the request through:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Gemini Embeddings
 ↓
ChromaDB
 ↓
LangChain Agent
 ↓
Retrieval Tool
 ↓
Relevant Context
 ↓
Groq LLM
 ↓
Answer
```

Users can then continue asking follow-up questions about the uploaded document.

---

# 🚧 Current Limitations

This project is currently designed as a **portfolio and learning project**.

Current limitations include:

- Conversation state uses in-memory storage.
- Vector storage is not configured as a production-managed database.
- Authentication is not implemented.
- Uploaded documents are not isolated by authenticated users.
- Scanned/image-based PDFs may require OCR.
- Production monitoring and evaluation have not yet been implemented.

---

# 🚀 Future Improvements

- [ ] Persistent vector database
- [ ] User authentication
- [ ] User-specific document isolation
- [ ] Persistent conversation history
- [ ] Source/page citations in answers
- [ ] OCR support for scanned PDFs
- [ ] RAG evaluation
- [ ] Retrieval reranking
- [ ] Hybrid search
- [ ] LangSmith observability
- [ ] Docker deployment

---

# 📚 What I Learned

This project provided hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Vector embeddings
- Semantic search
- Vector databases
- LangChain Agents
- Tool calling
- LangGraph
- LLM integration
- Document processing
- Conversational AI
- Streamlit application development
- AI application deployment concepts

---

# 👨‍💻 Author

**Arbaaz Shaikh**

Java Full Stack Developer | GenAI Engineer

### GitHub

https://github.com/arbaaz65dac/ai-ml-journey

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐.
