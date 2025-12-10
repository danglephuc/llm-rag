# LLM RAG Project

A NestJS REST API application that implements Retrieval-Augmented Generation (RAG) using:
- **RakutenAI-2.0-mini-instruct** (GGUF format) for CPU-based LLM inference via `node-llama-cpp`
- **Faiss** vector store for efficient similarity search via LangChain
- **@xenova/transformers** for local embeddings

## Features

- Pure Node.js implementation (no Python required)
- CPU-based inference for both LLM and embeddings
- **Server-Sent Events (SSE) API** for streaming chat responses
- REST API endpoints for health checks and warm-up
- Automatic document loading and chunking
- Persistent Faiss index for faster subsequent loads
- Configurable via environment variables
- Lazy initialization with on-demand warm-up for cold start optimization

## Prerequisites

- Node.js 18.x or higher
- Sufficient RAM (at least 2GB free for model loading)
- GGUF model file (see Setup section)
- **Build tools** (required for `node-llama-cpp`):
  - `cmake` (version 3.19 or higher - **important!**)
  - `build-essential` (includes gcc, g++, make)

### Installing Build Tools

**On Ubuntu/Debian (WSL2/Linux):**

CMake 3.19 or higher is required. The default Ubuntu repositories may have an older version. To install a newer version:

**Option 1: Install from Kitware APT repository (Recommended)**

Follow the official instructions from [Kitware APT Repository](https://apt.kitware.com/):

```bash
# Install prerequisites (if using minimal Ubuntu/Docker image)
sudo apt-get update
sudo apt-get install -y ca-certificates gpg wget build-essential

# Remove old CMake if installed
sudo apt-get remove --purge --auto-remove cmake

# Obtain Kitware signing key
test -f /usr/share/doc/kitware-archive-keyring/copyright || \
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

# Add repository based on your Ubuntu version
# For Ubuntu 24.04 (Noble Numbat):
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

# For Ubuntu 22.04 (Jammy Jellyfish):
# echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

# For Ubuntu 20.04 (Focal Fossa):
# echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

sudo apt-get update

# Remove manually obtained key to make room for package
test -f /usr/share/doc/kitware-archive-keyring/copyright || \
sudo rm /usr/share/keyrings/kitware-archive-keyring.gpg

# Install keyring package and CMake
sudo apt-get install -y kitware-archive-keyring
sudo apt-get install -y cmake
```

**Note:** Uncomment the repository line that matches your Ubuntu version. Check your version with `lsb_release -cs`.

**Option 2: Install from snap (if snap is available)**
```bash
sudo apt-get install -y build-essential
sudo snap install cmake --classic
```

**Option 3: Build from source (if other options don't work)**
```bash
sudo apt-get install -y build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0.tar.gz
tar -xzf cmake-3.28.0.tar.gz
cd cmake-3.28.0
./bootstrap
make -j$(nproc)
sudo make install
```

After installation, verify the version:
```bash
cmake --version
```

**On macOS:**
```bash
brew install cmake
```

**On Windows:**
Install [CMake](https://cmake.org/download/) and ensure it's in your PATH.

## Installation

1. Clone or navigate to the project directory:
```bash
cd llm-rag
```

2. Install dependencies:
```bash
npm install
```

**Note:** If you see errors about cmake not found during `npm install`, make sure you've installed the build tools above. The `node-llama-cpp` package will build from source if prebuilt binaries aren't available for your platform.

3. Download the RakutenAI-2.0-mini-instruct GGUF model:
   - Visit: https://huggingface.co/mmnga/RakutenAI-2.0-mini-instruct-gguf
   - Download a quantized version (recommended: `Q4_K_M` at 936MB or `Q4_0` at 890MB)
   - Place the model file in the `models/` directory

4. Create a `.env` file (optional, defaults are provided):
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
MODEL_PATH=./models/rakutenai-2.0-mini-instruct-q4_k_m.gguf
EMBEDDING_MODEL=Xenova/all-MiniLM-L6-v2
KNOWLEDGE_BASE_DIR=./knowledge-base
FAISS_INDEX_PATH=./faiss-store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3
MAX_TOKENS=512
TEMPERATURE=0.7
```

5. Add documents to the knowledge base:
   - Place `.txt` or `.md` files in the `knowledge-base/` directory
   - The system will automatically load and index them on first run

## Usage

### Starting the Server

Start the NestJS application:
```bash
# Development mode (with hot reload)
npm run start:dev

# Production mode
npm start
```

The server will start on `http://localhost:3000` (or the port specified in `PORT` environment variable).

### API Endpoints

#### 1. Health Check
Check if the RAG system is initialized and ready:
```bash
GET /health
```

**Response (200 OK - Ready):**
```json
{
  "status": "healthy",
  "rag": {
    "initialized": true,
    "ready": true
  },
  "timestamp": "2025-12-10T14:30:00.000Z"
}
```

**Response (503 Service Unavailable - Not Ready):**
```json
{
  "status": "unhealthy",
  "rag": {
    "initialized": false,
    "ready": false
  },
  "timestamp": "2025-12-10T14:30:00.000Z"
}
```

#### 2. Warm-up Endpoint
Initialize the RAG system on demand (useful for cold start optimization):
```bash
POST /warmup
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "RAG system initialized successfully",
  "timestamp": "2025-12-10T14:30:00.000Z"
}
```

If already initialized:
```json
{
  "status": "success",
  "message": "RAG system is already initialized",
  "timestamp": "2025-12-10T14:30:00.000Z"
}
```

#### 3. Chat Stream (SSE)
Stream chat responses using Server-Sent Events:
```bash
POST /chat/stream
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "topK": 3  // optional, defaults to configured TOP_K
}
```

**Response:** Server-Sent Events stream with the following event types:

- `chunk`: Token chunk from the streaming response
  ```
  data: {"type":"chunk","content":"Paris"}
  ```

- `done`: Stream completed
  ```
  data: {"type":"done","message":"Stream completed"}
  ```

- `error`: Error occurred
  ```
  data: {"type":"error","message":"Error description"}
  ```

**Example using curl:**
```bash
curl -N -X POST http://localhost:3000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

**Example using JavaScript (EventSource for GET or fetch for POST):**
```javascript
// Using fetch for POST with SSE
const response = await fetch('http://localhost:3000/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'What is the capital of France?' })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'chunk') {
        process.stdout.write(data.content);
      } else if (data.type === 'done') {
        console.log('\nStream completed');
      } else if (data.type === 'error') {
        console.error('Error:', data.message);
      }
    }
  }
}
```

### Initialization Flow

The RAG system uses **lazy initialization**:
1. The server starts immediately without loading models
2. On first chat request or when calling `/warmup`, the system will:
   - Load the GGUF model (first time may take a moment)
   - Initialize the embedding model
   - Load or create the Faiss vector store
   - If no index exists, it will process documents from `knowledge-base/`
3. Subsequent requests use the initialized system

This allows the server to start quickly and initialize on-demand, which is ideal for serverless or containerized deployments.

## Project Structure

```
llm-rag/
├── src/
│   ├── main.ts               # NestJS application entry point
│   ├── app.module.ts         # Root application module
│   ├── app.controller.ts     # Root controller
│   ├── app.service.ts        # Root service
│   ├── index.js              # Legacy CLI entry point (optional)
│   ├── llm/
│   │   ├── modelLoader.js    # Model loading with node-llama-cpp
│   │   └── llamaAdapter.js   # LangChain adapter with streaming support
│   ├── rag/
│   │   ├── rag.service.ts    # RAG system management service
│   │   ├── rag.controller.ts # Warm-up endpoint controller
│   │   ├── rag.module.ts     # RAG module
│   │   ├── vectorStore.js    # Faiss vector store management
│   │   ├── embeddings.js     # Custom embeddings with @xenova/transformers
│   │   └── documentLoader.js # Document loading and chunking
│   ├── chat/
│   │   ├── chat.controller.ts # SSE chat endpoint controller
│   │   ├── chat.module.ts     # Chat module
│   │   └── chatHandler.js     # RAG pipeline orchestration with streaming
│   └── health/
│       ├── health.controller.ts # Health check endpoint
│       └── health.module.ts     # Health module
├── knowledge-base/           # Place your documents here (.txt, .md)
├── faiss-store/              # Generated Faiss index (auto-created)
├── models/                   # Place GGUF model file here
├── dist/                     # Compiled output (generated)
└── package.json
```

## Configuration

All configuration is done via environment variables (see `.env.example`):

- `PORT`: Server port (default: 3000)
- `MODEL_PATH`: Path to the GGUF model file
- `EMBEDDING_MODEL`: Embedding model name (default: Xenova/all-MiniLM-L6-v2)
- `KNOWLEDGE_BASE_DIR`: Directory containing knowledge base documents
- `FAISS_INDEX_PATH`: Path to save/load Faiss index
- `CHUNK_SIZE`: Document chunk size in characters (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K`: Number of documents to retrieve for RAG (default: 3)
- `MAX_TOKENS`: Maximum tokens to generate (default: 512)
- `TEMPERATURE`: Generation temperature (default: 0.7)

## How It Works

1. **Document Processing**: Documents from `knowledge-base/` are loaded and split into chunks
2. **Embedding Generation**: Each chunk is converted to a vector using the embedding model
3. **Vector Store**: Embeddings are stored in a Faiss index for fast similarity search
4. **Query Processing**: User queries are embedded and used to find relevant document chunks
5. **Response Generation**: Retrieved context is combined with the query and sent to the LLM

## Model Information

- **LLM**: RakutenAI-2.0-mini-instruct (GGUF format)
  - Source: https://huggingface.co/mmnga/RakutenAI-2.0-mini-instruct-gguf
  - Size: ~936MB (Q4_K_M quantization)
  - Languages: Japanese, English
  - Architecture: LLaMA-based

- **Embeddings**: Xenova/all-MiniLM-L6-v2
  - Small, efficient model for CPU inference
  - 384-dimensional embeddings

## Troubleshooting

**CMake version too old (requires 3.19+)**: 
- The default Ubuntu repositories may have CMake 3.16.x which is too old
- Follow the "Installing Build Tools" section above to install CMake 3.19 or higher
- After upgrading, run `npm install` again

**cmake not found during npm install**: 
- Install cmake and build-essential (see "Installing Build Tools" section above)
- The `node-llama-cpp` package builds from source if prebuilt binaries aren't available
- This is normal and expected on some systems - the build process may take several minutes

**Model not found**: Ensure the GGUF model file is in the `models/` directory and the path in `.env` is correct.

**Out of memory**: The model requires ~1GB RAM. Close other applications or use a smaller quantization.

**No documents found**: Add `.txt` or `.md` files to the `knowledge-base/` directory.

**Slow performance**: CPU inference is slower than GPU. Consider using a smaller model or quantization.

**Module not found errors**: 
- Make sure to run `npm run build` after making changes
- The build process copies JavaScript files to the `dist/` folder
- If issues persist, delete `dist/` and `node_modules/`, then run `npm install` and `npm run build` again

**Port already in use**: 
- Change the `PORT` environment variable or stop the process using port 3000
- Check running processes: `lsof -i :3000` (macOS/Linux) or `netstat -ano | findstr :3000` (Windows)

**SSE connection issues**: 
- Ensure CORS is properly configured if accessing from a browser
- Check that the client supports Server-Sent Events
- For POST requests with SSE, use `fetch` API or a library that supports POST with streaming responses

## License

ISC

## Development

### Building the Project

```bash
# Build TypeScript to JavaScript
npm run build

# Build output goes to dist/
```

### Running in Development Mode

```bash
# Start with hot reload (watches for file changes)
npm run start:dev

# Start in debug mode
npm run start:debug
```

### Testing the API

After starting the server, you can test the endpoints:

```bash
# Check health
curl http://localhost:3000/health

# Warm up the system
curl -X POST http://localhost:3000/warmup

# Test chat stream
curl -N -X POST http://localhost:3000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?"}'
```

## Architecture Notes

- **Lazy Initialization**: The RAG system initializes on first use or when `/warmup` is called, allowing fast server startup
- **Streaming Responses**: Chat responses are streamed token-by-token using Server-Sent Events for real-time user experience
- **Module System**: TypeScript NestJS modules import JavaScript ES modules using dynamic imports
- **Singleton Pattern**: RAG service maintains singleton instances of LLM, vector store, and chat handler

## References

- [NestJS Documentation](https://docs.nestjs.com/)
- [LangChain.js FaissStore Documentation](https://docs.langchain.com/oss/javascript/integrations/vectorstores/faiss)
- [node-llama-cpp Documentation](https://node-llama-cpp.withcat.ai/)
- [RakutenAI-2.0-mini-instruct-gguf Model](https://huggingface.co/mmnga/RakutenAI-2.0-mini-instruct-gguf)
- [Server-Sent Events (SSE) Specification](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

