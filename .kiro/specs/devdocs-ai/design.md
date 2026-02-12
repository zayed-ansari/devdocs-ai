# Design Document: DevDocs AI

## Overview

DevDocs AI is an AI-powered documentation assistant that enables developers to query codebases conversationally using Retrieval-Augmented Generation (RAG). The system combines semantic search over code embeddings with large language model generation to provide accurate, context-aware explanations with precise source code citations.

The architecture follows a modular design with clear separation between ingestion, retrieval, and generation layers. The system uses ChromaDB for vector storage, LangChain for RAG orchestration, and Gemini API for natural language understanding and generation.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│                     (Streamlit Web UI)                       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────┴────────────────────────────────────┐
│                      Backend Layer                           │
│                    (FastAPI Server)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Session    │  │    Query     │  │   Response   │     │
│  │  Manager     │  │  Processor   │  │  Generator   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    Processing Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     Code     │  │   Chunking   │  │  Embedding   │     │
│  │    Parser    │  │    Engine    │  │   Engine     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                      Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   ChromaDB   │  │   Session    │  │     File     │     │
│  │ Vector Store │  │    Store     │  │    Cache     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   External Services                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Gemini API  │  │  GitHub API  │  │    Ollama    │     │
│  │   (Primary)  │  │              │  │  (Fallback)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Principles

1. **Modularity**: Clear separation between parsing, embedding, retrieval, and generation
2. **Scalability**: Stateless backend design enabling horizontal scaling
3. **Resilience**: Fallback mechanisms for external service failures
4. **Performance**: Caching and indexing strategies for sub-3-second responses
5. **Extensibility**: Plugin architecture for adding new language parsers

## Components and Interfaces

### 1. Frontend Layer

#### Streamlit Web UI

**Responsibilities:**
- Render chat interface for user queries and AI responses
- Display code snippets with syntax highlighting
- Manage codebase upload (GitHub URL or file upload)
- Show session management interface
- Display progress indicators for long-running operations

**Key Components:**
- `ChatInterface`: Main conversational UI component
- `CodeViewer`: Syntax-highlighted code display with line numbers
- `SessionSelector`: Dropdown for switching between codebase sessions
- `UploadManager`: File upload and GitHub URL input handler
- `CitationPanel`: Expandable panel showing source code references

**State Management:**
- Session state stored in Streamlit session_state
- Conversation history maintained in memory
- Active session ID tracked for backend API calls

### 2. Backend Layer (FastAPI)

#### Session Manager

**Responsibilities:**
- Create, list, switch, and delete codebase sessions
- Manage session metadata (name, creation date, file count)
- Load and persist session state

**Interface:**
```python
class SessionManager:
    def create_session(name: str, codebase_path: str) -> Session
    def get_session(session_id: str) -> Session
    def list_sessions() -> List[SessionMetadata]
    def delete_session(session_id: str) -> bool
    def switch_session(session_id: str) -> Session
```

**Data Model:**
```python
@dataclass
class Session:
    id: str
    name: str
    created_at: datetime
    vector_store_path: str
    conversation_history: List[Message]
    metadata: SessionMetadata

@dataclass
class SessionMetadata:
    file_count: int
    total_lines: int
    languages: List[str]
    last_accessed: datetime
```

#### Query Processor

**Responsibilities:**
- Accept user queries and generate embeddings
- Retrieve relevant code chunks from vector store
- Rank and filter results by relevance
- Maintain conversation context

**Interface:**
```python
class QueryProcessor:
    def process_query(query: str, session_id: str, history: List[Message]) -> QueryResult
    def generate_query_embedding(query: str) -> np.ndarray
    def retrieve_chunks(embedding: np.ndarray, top_k: int) -> List[CodeChunk]
    def rerank_chunks(chunks: List[CodeChunk], query: str) -> List[CodeChunk]
```

**Data Model:**
```python
@dataclass
class QueryResult:
    query: str
    retrieved_chunks: List[CodeChunk]
    relevance_scores: List[float]
    context_window: str
```

#### Response Generator

**Responsibilities:**
- Format retrieved chunks as LLM context
- Generate responses using Gemini API or Ollama
- Extract and format citations from responses
- Handle LLM errors and fallbacks

**Interface:**
```python
class ResponseGenerator:
    def generate_response(query: str, chunks: List[CodeChunk], history: List[Message]) -> Response
    def format_context(chunks: List[CodeChunk]) -> str
    def extract_citations(response_text: str, chunks: List[CodeChunk]) -> List[Citation]
    def fallback_to_ollama(query: str, context: str) -> str
```

**Data Model:**
```python
@dataclass
class Response:
    text: str
    citations: List[Citation]
    model_used: str
    generation_time: float

@dataclass
class Citation:
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str]
    code_snippet: str
```

### 3. Processing Layer

#### Code Parser

**Responsibilities:**
- Parse uploaded codebases (GitHub repos or local files)
- Extract functions, classes, and modules
- Detect programming language
- Filter out binary files and build artifacts

**Interface:**
```python
class CodeParser:
    def parse_repository(repo_url: str) -> ParsedCodebase
    def parse_local_files(file_path: str) -> ParsedCodebase
    def extract_functions(file_content: str, language: str) -> List[Function]
    def extract_classes(file_content: str, language: str) -> List[Class]
    def detect_language(file_path: str) -> str
```

**Supported Languages:**
- Python: Uses `ast` module for AST parsing
- JavaScript/TypeScript: Uses `esprima` or `tree-sitter`
- Java: Uses `javalang` parser
- C++: Uses `tree-sitter-cpp`
- Go: Uses `tree-sitter-go`
- Rust: Uses `tree-sitter-rust`

**Data Model:**
```python
@dataclass
class ParsedCodebase:
    files: List[SourceFile]
    total_lines: int
    languages: Dict[str, int]  # language -> line count
    
@dataclass
class SourceFile:
    path: str
    language: str
    content: str
    functions: List[Function]
    classes: List[Class]
    imports: List[str]

@dataclass
class Function:
    name: str
    start_line: int
    end_line: int
    signature: str
    docstring: Optional[str]
    body: str
```

#### Chunking Engine

**Responsibilities:**
- Split source files into logical chunks
- Preserve function and class boundaries
- Maintain context (docstrings, comments)
- Handle large functions by splitting at logical boundaries

**Interface:**
```python
class ChunkingEngine:
    def chunk_file(file: SourceFile) -> List[CodeChunk]
    def chunk_function(function: Function) -> List[CodeChunk]
    def split_large_chunk(chunk: str, max_tokens: int) -> List[str]
    def add_context(chunk: str, file: SourceFile) -> str
```

**Chunking Strategy:**
1. **Function-level chunking**: Each function becomes a chunk
2. **Class-level chunking**: Small classes chunked together
3. **Context preservation**: Include docstrings, type hints, and surrounding comments
4. **Overlap**: 50-token overlap between adjacent chunks for continuity

**Data Model:**
```python
@dataclass
class CodeChunk:
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    function_name: Optional[str]
    class_name: Optional[str]
    chunk_type: ChunkType  # FUNCTION, CLASS, MODULE
    metadata: Dict[str, Any]
```

#### Embedding Engine

**Responsibilities:**
- Generate embeddings for code chunks
- Use sentence-transformers for semantic encoding
- Batch process embeddings for efficiency
- Handle embedding failures gracefully

**Interface:**
```python
class EmbeddingEngine:
    def generate_embeddings(chunks: List[CodeChunk]) -> List[np.ndarray]
    def generate_single_embedding(text: str) -> np.ndarray
    def batch_embed(texts: List[str], batch_size: int) -> List[np.ndarray]
```

**Model Configuration:**
- **Primary Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Embedding dimension: 384
  - Fast inference: ~5ms per chunk
  - Good balance of speed and quality
- **Alternative**: `sentence-transformers/all-mpnet-base-v2`
  - Higher quality but slower
  - Use for smaller codebases

### 4. Data Layer

#### ChromaDB Vector Store

**Responsibilities:**
- Store code chunk embeddings
- Perform similarity search
- Manage collections per session
- Persist data to disk

**Schema:**
```python
Collection Schema:
- id: str (chunk_id)
- embedding: List[float] (384-dimensional vector)
- metadata: {
    "file_path": str,
    "start_line": int,
    "end_line": int,
    "language": str,
    "function_name": str,
    "class_name": str,
    "chunk_type": str
  }
- document: str (chunk content)
```

**Operations:**
```python
class VectorStore:
    def create_collection(session_id: str) -> Collection
    def add_chunks(chunks: List[CodeChunk], embeddings: List[np.ndarray])
    def query(embedding: np.ndarray, top_k: int, filter: Dict) -> List[Result]
    def delete_collection(session_id: str)
```

**Indexing Strategy:**
- HNSW (Hierarchical Navigable Small World) index for fast similarity search
- Metadata filtering for language-specific queries
- Periodic index optimization for large collections

#### Session Store

**Responsibilities:**
- Persist session metadata
- Store conversation history
- Track user preferences

**Storage:**
- SQLite database for session metadata
- JSON files for conversation history
- File system for session-specific data

**Schema:**
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP,
    last_accessed TIMESTAMP,
    vector_store_path TEXT,
    file_count INTEGER,
    total_lines INTEGER
);

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp TIMESTAMP,
    role TEXT,  -- 'user' or 'assistant'
    content TEXT,
    citations TEXT,  -- JSON array
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

#### File Cache

**Responsibilities:**
- Cache uploaded files temporarily
- Store cloned GitHub repositories
- Clean up old files

**Implementation:**
- Temporary directory per session
- TTL-based cleanup (24 hours)
- Size limits (500MB per session)

## Data Models

### Core Data Structures

```python
@dataclass
class Message:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    citations: Optional[List[Citation]] = None

@dataclass
class CodeChunk:
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    function_name: Optional[str]
    class_name: Optional[str]
    embedding: Optional[np.ndarray] = None

@dataclass
class Citation:
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str]
    code_snippet: str
    relevance_score: float

@dataclass
class QueryContext:
    query: str
    session_id: str
    conversation_history: List[Message]
    retrieved_chunks: List[CodeChunk]
    max_context_tokens: int = 4000
```

## Data Flow

### 1. Codebase Ingestion Flow

```
User Upload → Code Parser → Chunking Engine → Embedding Engine → Vector Store
     │              │              │                  │                │
     │              │              │                  │                │
  GitHub URL    Extract AST    Split into        Generate         Store in
  or Zip File   Functions/     Logical          Embeddings       ChromaDB
                Classes        Chunks           (384-dim)        Collection
```

**Detailed Steps:**

1. **Upload**: User provides GitHub URL or uploads zip file
2. **Clone/Extract**: System clones repo or extracts files to temp directory
3. **Parse**: Code Parser processes each file:
   - Detect language from file extension
   - Parse AST to extract functions and classes
   - Extract docstrings and comments
4. **Chunk**: Chunking Engine creates logical chunks:
   - One chunk per function (if < 1000 tokens)
   - Split large functions at logical boundaries
   - Add context (imports, class definition)
5. **Embed**: Embedding Engine generates vectors:
   - Batch process chunks (batch_size=32)
   - Generate 384-dimensional embeddings
6. **Store**: Save to ChromaDB:
   - Create collection for session
   - Store embeddings with metadata
   - Build HNSW index

**Performance:**
- 100,000 LOC processed in ~5 minutes
- Parallel processing for multiple files
- Progress updates every 10% completion

### 2. Query Processing Flow

```
User Query → Query Processor → Vector Store → Response Generator → UI Display
     │              │                │               │                  │
     │              │                │               │                  │
  Natural      Generate Query    Similarity      Format Context    Render with
  Language     Embedding         Search          + LLM Call        Syntax
  Question     (384-dim)         (top-5)         (Gemini API)      Highlighting
```

**Detailed Steps:**

1. **Query Input**: User types question in chat interface
2. **Embedding**: Generate query embedding using same model
3. **Retrieval**: ChromaDB similarity search:
   - Find top-5 most similar chunks
   - Filter by relevance threshold (> 0.6)
   - Apply metadata filters if specified
4. **Context Formation**: Format retrieved chunks:
   - Add file paths and line numbers
   - Include function signatures
   - Limit total context to 4000 tokens
5. **LLM Generation**: Call Gemini API:
   - System prompt: "You are a code documentation assistant..."
   - User prompt: Query + formatted context
   - Temperature: 0.3 (more deterministic)
6. **Citation Extraction**: Parse LLM response:
   - Identify code references
   - Map to original chunks
   - Format with file:line notation
7. **Display**: Render in Streamlit:
   - Show response text
   - Display citations as expandable panels
   - Syntax highlight code snippets

**Performance:**
- Query embedding: ~10ms
- Vector search: ~50ms
- LLM generation: ~1-2 seconds
- Total: < 3 seconds

### 3. Conversation Context Management

```
Query N → Retrieve History → Append to Context → LLM Generation → Update History
              (Last 10)          (+ New Query)                        (+ Response)
```

**Context Window Management:**
- Keep last 10 conversation turns
- Summarize older context if needed
- Prioritize recent queries for relevance

## API Endpoints

### REST API (FastAPI)

```python
# Session Management
POST   /api/sessions                    # Create new session
GET    /api/sessions                    # List all sessions
GET    /api/sessions/{session_id}       # Get session details
DELETE /api/sessions/{session_id}       # Delete session
PUT    /api/sessions/{session_id}/switch # Switch active session

# Codebase Ingestion
POST   /api/ingest/github               # Ingest from GitHub URL
POST   /api/ingest/upload               # Upload local files
GET    /api/ingest/status/{task_id}     # Check ingestion progress

# Query Processing
POST   /api/query                       # Submit query
GET    /api/query/history/{session_id}  # Get conversation history

# Code Retrieval
GET    /api/code/{session_id}/file      # Get full file content
GET    /api/code/{session_id}/function  # Get function definition
POST   /api/code/search                 # Semantic code search
```

### Request/Response Examples

**Create Session:**
```json
POST /api/sessions
{
  "name": "my-project",
  "source_type": "github",
  "source_url": "https://github.com/user/repo"
}

Response:
{
  "session_id": "sess_abc123",
  "name": "my-project",
  "status": "processing",
  "task_id": "task_xyz789"
}
```

**Submit Query:**
```json
POST /api/query
{
  "session_id": "sess_abc123",
  "query": "How does the authentication system work?",
  "include_history": true
}

Response:
{
  "response": "The authentication system uses JWT tokens...",
  "citations": [
    {
      "file_path": "src/auth/jwt.py",
      "start_line": 45,
      "end_line": 67,
      "function_name": "generate_token",
      "code_snippet": "def generate_token(user_id: str) -> str:\n    ..."
    }
  ],
  "model_used": "gemini-pro",
  "generation_time": 1.23
}
```

## Database Schema

### SQLite Schema

```sql
-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vector_store_path TEXT NOT NULL,
    file_count INTEGER DEFAULT 0,
    total_lines INTEGER DEFAULT 0,
    languages TEXT  -- JSON array
);

-- Conversations table
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations TEXT,  -- JSON array of citations
    model_used TEXT,
    generation_time FLOAT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Ingestion tasks table
CREATE TABLE ingestion_tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    progress INTEGER DEFAULT 0,
    total_files INTEGER,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_sessions_last_accessed ON sessions(last_accessed);
CREATE INDEX idx_conversations_session ON conversations(session_id, timestamp);
CREATE INDEX idx_tasks_status ON ingestion_tasks(status);
```

### ChromaDB Collections

Each session has its own collection:

```python
collection_name = f"session_{session_id}"

# Metadata stored per chunk
metadata = {
    "file_path": str,
    "start_line": int,
    "end_line": int,
    "language": str,
    "function_name": str,
    "class_name": str,
    "chunk_type": str,  # "function", "class", "module"
    "complexity": int,  # cyclomatic complexity
    "doc_length": int   # docstring length
}
```

## Deployment Strategy

### Docker Containerization

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose ports
EXPOSE 8000 8501

# Start services
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port 8501"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  devdocs-ai:
    build: .
    ports:
      - "8000:8000"  # FastAPI
      - "8501:8501"  # Streamlit
    volumes:
      - ./data:/app/data
      - ./sessions:/app/sessions
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - CHROMA_PERSIST_DIR=/app/data/chroma
      - SESSION_DB_PATH=/app/data/sessions.db
    restart: unless-stopped
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-api-key"
export GITHUB_TOKEN="your-github-token"

# Run backend
uvicorn api.main:app --reload --port 8000

# Run frontend (separate terminal)
streamlit run ui/app.py --server.port 8501
```

### Cloud Deployment Options

**Option 1: Single VM (Hackathon Demo)**
- Deploy on Google Cloud Compute Engine or AWS EC2
- Instance type: 4 vCPU, 16GB RAM
- Docker Compose for orchestration
- Nginx reverse proxy for SSL

**Option 2: Kubernetes (Production)**
- Separate pods for API and UI
- Persistent volumes for ChromaDB and SQLite
- Horizontal pod autoscaling based on CPU
- Load balancer for traffic distribution

**Option 3: Serverless (Cost-Optimized)**
- Cloud Run for API (scales to zero)
- Cloud Storage for vector store
- Cloud SQL for session database
- Static hosting for Streamlit (if possible)

### Environment Configuration

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str
    github_token: Optional[str] = None
    
    # Database
    chroma_persist_dir: str = "./data/chroma"
    session_db_path: str = "./data/sessions.db"
    
    # Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gemini-pro"
    llm_temperature: float = 0.3
    
    # Performance
    max_chunk_tokens: int = 1000
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.6
    max_context_tokens: int = 4000
    
    # Limits
    max_upload_size_mb: int = 500
    max_codebase_lines: int = 100000
    max_concurrent_users: int = 10
    
    class Config:
        env_file = ".env"
```

## Scalability Considerations

### Performance Optimization

**1. Caching Strategy**
- Cache frequently accessed code chunks in Redis
- Cache query embeddings for repeated queries
- LRU cache for session metadata

**2. Batch Processing**
- Batch embed chunks during ingestion (batch_size=32)
- Parallel file parsing using multiprocessing
- Async I/O for API calls

**3. Index Optimization**
- Use HNSW index for fast similarity search
- Periodic index rebuilding for large collections
- Metadata filtering to reduce search space

**4. Database Optimization**
- Index on session_id and timestamp
- Partition conversations table by session
- Archive old sessions to cold storage

### Horizontal Scaling

**Stateless Backend:**
- API servers don't hold session state
- Session data stored in shared database
- Vector stores on shared file system or object storage

**Load Balancing:**
- Round-robin distribution across API instances
- Sticky sessions for WebSocket connections
- Health checks for automatic failover

**Resource Limits:**
- Rate limiting per user (10 queries/minute)
- Queue system for ingestion tasks
- Circuit breaker for external API calls

### Monitoring and Observability

**Metrics to Track:**
- Query latency (p50, p95, p99)
- Ingestion throughput (files/second)
- Vector store size and query performance
- LLM API usage and costs
- Error rates by component

**Logging:**
- Structured logging (JSON format)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Correlation IDs for request tracing

**Alerting:**
- High error rates (> 5%)
- Slow queries (> 5 seconds)
- API quota exhaustion
- Disk space warnings



## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property Reflection

After analyzing all acceptance criteria, I identified several areas where properties can be consolidated:

**Redundancy Analysis:**
- Properties 1.3 and 6.1 both test multi-language support → Consolidated into Property 1
- Properties 2.1 and 6.4 both test syntax boundary preservation → Consolidated into Property 2
- Properties 2.5 and 5.2 both test metadata completeness → Consolidated into Property 4
- Properties 5.1, 5.2, and 5.3 all test citation format → Consolidated into Property 8

This reflection ensures each property provides unique validation value without logical redundancy.

### Core Properties

**Property 1: Multi-language parsing completeness**
*For any* codebase containing files in Python, JavaScript, TypeScript, Java, C++, Go, or Rust, the parser should successfully extract functions, classes, and modules from all supported file types, and the ingestion summary should list all detected languages.
**Validates: Requirements 1.3, 6.1, 6.2**

**Property 2: Chunk boundary preservation**
*For any* source file with functions and classes, all generated chunks should preserve complete function and class boundaries without breaking syntactic validity, and chunks should include associated docstrings and comments.
**Validates: Requirements 2.1, 2.2, 6.4**

**Property 3: Large chunk splitting**
*For any* code chunk exceeding 1000 tokens, the chunking engine should split it at logical boundaries (statement or expression boundaries) such that all resulting sub-chunks are syntactically valid in their language.
**Validates: Requirements 2.3**

**Property 4: Embedding storage completeness**
*For any* code chunk that is embedded and stored, the vector store entry should contain all required metadata fields: file_path, start_line, end_line, language, and function_name (when applicable), and the embedding should be a 384-dimensional vector.
**Validates: Requirements 2.4, 2.5**

**Property 5: File filtering correctness**
*For any* codebase upload containing binary files, build artifacts (node_modules, target, dist, build), and dependency directories, the parser should exclude all such files from processing, and the ingestion summary should only count source code files.
**Validates: Requirements 1.4**

**Property 6: Upload size validation**
*For any* file upload, if the total size exceeds 500MB, the system should reject the upload with a clear error message; if the size is within limits, the upload should be accepted and processed.
**Validates: Requirements 1.2**

**Property 7: Query embedding generation**
*For any* user query string, the RAG system should generate a 384-dimensional embedding vector using the same model as code chunks.
**Validates: Requirements 3.1**

**Property 8: Citation format completeness**
*For any* response that references code chunks, all citations should include file_path, start_line, end_line, and when the chunk is a function or class, the citation should also include the function_name or class_name.
**Validates: Requirements 5.1, 5.2, 5.3**

**Property 9: Citation context window**
*For any* cited code section, when displayed to the user, the system should include exactly 5 lines of context before the start_line and 5 lines after the end_line (or fewer if at file boundaries).
**Validates: Requirements 5.5**

**Property 10: Conversation history truncation**
*For any* session with more than 10 conversation exchanges (20 messages), the system should maintain only the most recent 10 exchanges when processing new queries, discarding older messages.
**Validates: Requirements 4.4**

**Property 11: Response grounding**
*For any* generated response containing code references or explanations, all specific code mentions (function names, variable names, code snippets) should exist in the retrieved chunks provided as context to the LLM.
**Validates: Requirements 7.2**

**Property 12: Top-k retrieval consistency**
*For any* query embedding, the vector store should return at most 5 chunks (or fewer if the total collection has fewer than 5 chunks), ordered by similarity score in descending order.
**Validates: Requirements 3.2**

**Property 13: Similarity threshold filtering**
*For any* query where all chunks have similarity scores below 0.6, the system should return zero results and the response should explicitly state that no relevant code was found.
**Validates: Requirements 3.5**

**Property 14: Session persistence**
*For any* created session, after the session is created, querying the session store should return the session with its vector_store_path and conversation_history intact, and the vector store collection should exist and be queryable.
**Validates: Requirements 8.2**

**Property 15: Session deletion completeness**
*For any* session that is deleted, subsequent queries for that session should return "not found", the vector store collection should be removed, and all associated conversation history should be deleted.
**Validates: Requirements 8.5**

**Property 16: Session metadata completeness**
*For any* list of sessions, each session entry should include name, created_at, file_count, total_lines, and languages fields.
**Validates: Requirements 8.3**

**Property 17: Embedding failure resilience**
*For any* batch of chunks where some chunks fail to generate embeddings, the system should log errors for failed chunks and successfully process and store all chunks that generated embeddings successfully.
**Validates: Requirements 10.3**

**Property 18: Ingestion error reporting**
*For any* codebase ingestion that fails (invalid URL, network error, parsing error), the system should return an error response with a specific error message indicating the failure type and reason.
**Validates: Requirements 10.1**

**Property 19: Progress indicator emission**
*For any* long-running ingestion operation, the system should emit progress updates at least every 10% of completion, and each update should include the current progress percentage and files processed count.
**Validates: Requirements 10.5**

**Property 20: Syntax highlighting markers**
*For any* response containing code snippets, the formatted output should include language-specific syntax highlighting markers (markdown code fences with language identifiers).
**Validates: Requirements 4.5**

**Property 21: Context inclusion in LLM prompts**
*For any* query processing, the prompt sent to the LLM should include all retrieved code chunks with their file paths and line numbers as context.
**Validates: Requirements 4.2**

**Property 22: Conversation history availability**
*For any* follow-up query in a session, the query processor should have access to all previous messages in the conversation history (up to the 10-message limit).
**Validates: Requirements 4.3**

**Property 23: Session creation with naming**
*For any* session creation request with a valid name, the system should create a new session with that name and return a unique session_id.
**Validates: Requirements 8.1**

### Edge Case Properties

**Property 24: Empty codebase handling**
*For any* codebase with zero supported source files, the ingestion should complete successfully with file_count=0 and queries should return "no code indexed" messages.
**Validates: Requirements 1.5**

**Property 25: Minimum chunk size**
*For any* source file with very small functions (< 50 tokens), the chunking engine should still create valid chunks, potentially combining multiple small functions into a single chunk while preserving boundaries.
**Validates: Requirements 2.1**

### Example-Based Tests

The following scenarios should be tested with specific examples rather than property-based tests:

**Example 1: Private repository authentication**
When attempting to clone a private GitHub repository without credentials, the system should prompt for authentication.
**Validates: Requirements 10.2**

**Example 2: LLM API unavailability**
When the Gemini API returns a 503 error, the system should display "The AI service is temporarily unavailable. Please try again in a moment."
**Validates: Requirements 10.4**

**Example 3: Unanswerable query**
When a user asks "How does the payment processing work?" on a codebase with no payment-related code, the system should respond "I couldn't find any relevant code in this codebase to answer your question about payment processing."
**Validates: Requirements 7.3**

## Error Handling

### Error Categories

**1. Ingestion Errors**
- **Invalid GitHub URL**: Return 400 with message "Invalid GitHub repository URL format"
- **Repository not found**: Return 404 with message "Repository not found or not accessible"
- **Clone failure**: Return 500 with message "Failed to clone repository: {error_details}"
- **File size exceeded**: Return 413 with message "Upload size exceeds 500MB limit"
- **No supported files**: Return 400 with message "No supported source files found in codebase"

**2. Parsing Errors**
- **Syntax errors in source**: Log warning, skip file, continue processing
- **Unsupported language**: Skip file, log info message
- **Encoding errors**: Try UTF-8, fallback to latin-1, skip if both fail

**3. Embedding Errors**
- **Model loading failure**: Return 500 with message "Failed to load embedding model"
- **Individual chunk failure**: Log error, continue with remaining chunks
- **Batch timeout**: Retry with smaller batch size, fail after 3 attempts

**4. Query Errors**
- **Empty query**: Return 400 with message "Query cannot be empty"
- **Session not found**: Return 404 with message "Session not found"
- **No results found**: Return 200 with message "No relevant code found for your query"

**5. LLM Errors**
- **API unavailable**: Return 503 with message "AI service temporarily unavailable"
- **Rate limit exceeded**: Return 429 with message "Rate limit exceeded, please try again in {seconds} seconds"
- **Context too large**: Truncate context, retry with reduced chunks
- **Timeout**: Return 504 with message "Request timed out, please try again"

**6. Storage Errors**
- **Disk full**: Return 507 with message "Storage capacity exceeded"
- **Vector store corruption**: Attempt recovery, recreate collection if needed
- **Database lock**: Retry with exponential backoff (max 3 attempts)

### Error Recovery Strategies

**Graceful Degradation:**
- If Gemini API fails, attempt fallback to Ollama (if configured)
- If syntax highlighting fails, return plain text code
- If citation extraction fails, return response without citations

**Retry Logic:**
- Network errors: Exponential backoff (1s, 2s, 4s)
- Rate limits: Wait for specified duration
- Transient failures: Max 3 retries

**User Feedback:**
- All errors include actionable messages
- Progress indicators show current step during failures
- Logs include correlation IDs for debugging

## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

**Unit Tests** focus on:
- Specific examples demonstrating correct behavior
- Edge cases (empty files, single-line functions, special characters)
- Error conditions (invalid inputs, API failures, resource exhaustion)
- Integration points between components

**Property-Based Tests** focus on:
- Universal properties that hold for all inputs
- Comprehensive input coverage through randomization
- Invariants that must be maintained
- Round-trip properties (parse → chunk → embed → retrieve)

### Property-Based Testing Configuration

**Framework**: Use `hypothesis` for Python property-based testing

**Configuration:**
- Minimum 100 iterations per property test
- Each test tagged with: `# Feature: devdocs-ai, Property {N}: {property_text}`
- Seed-based reproducibility for failed tests
- Shrinking enabled to find minimal failing examples

**Example Property Test Structure:**
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=10000))
def test_property_7_query_embedding_generation(query: str):
    """
    Feature: devdocs-ai, Property 7: Query embedding generation
    For any user query string, the RAG system should generate 
    a 384-dimensional embedding vector.
    """
    embedding = embedding_engine.generate_query_embedding(query)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32
```

### Test Coverage Requirements

**Component-Level Tests:**
- Code Parser: 90% coverage
- Chunking Engine: 95% coverage (critical for correctness)
- Embedding Engine: 85% coverage
- Query Processor: 90% coverage
- Response Generator: 85% coverage

**Integration Tests:**
- End-to-end ingestion flow
- End-to-end query flow
- Session management lifecycle
- Error recovery scenarios

**Performance Tests:**
- Ingestion throughput (target: 100k LOC in 5 minutes)
- Query latency (target: p95 < 3 seconds)
- Concurrent user load (target: 10 users)
- Memory usage under load

### Test Data Strategy

**Synthetic Codebases:**
- Generate code in all supported languages
- Include edge cases (empty files, large files, nested structures)
- Vary complexity (simple scripts to complex frameworks)

**Real-World Codebases:**
- Test with popular open-source projects
- Include polyglot repositories
- Test with different project structures

**Query Datasets:**
- Common developer questions
- Ambiguous queries
- Domain-specific terminology
- Edge cases (empty, very long, special characters)

### Continuous Testing

**Pre-commit Hooks:**
- Run unit tests
- Check code formatting
- Validate type hints

**CI/CD Pipeline:**
- Run full test suite on every PR
- Property tests with 100 iterations
- Integration tests with real LLM API
- Performance benchmarks on standard codebases

**Monitoring in Production:**
- Track query success rates
- Monitor error rates by type
- Alert on performance degradation
- Log failed property violations
