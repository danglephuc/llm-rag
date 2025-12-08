# LLM RAG Project

A Node.js application that implements Retrieval-Augmented Generation (RAG) using:
- **RakutenAI-2.0-mini-instruct** (GGUF format) for CPU-based LLM inference via `node-llama-cpp`
- **Faiss** vector store for efficient similarity search via LangChain
- **@xenova/transformers** for local embeddings

## Features

- Pure Node.js implementation (no Python required)
- CPU-based inference for both LLM and embeddings
- Interactive CLI chat interface
- Automatic document loading and chunking
- Persistent Faiss index for faster subsequent loads
- Configurable via environment variables

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

Start the application:
```bash
npm start
```

The system will:
1. Load the GGUF model (first time may take a moment)
2. Initialize the embedding model
3. Load or create the Faiss vector store
4. If no index exists, it will process documents from `knowledge-base/`

Once ready, you can start chatting:
```
You: What is the capital of France?
AI: [Response based on knowledge base and model]
```

Type `exit` or `quit` to stop the application.

## Project Structure

```
llm-rag/
├── src/
│   ├── index.js              # Main CLI entry point
│   ├── llm/
│   │   ├── modelLoader.js    # Model loading with node-llama-cpp
│   │   └── llamaAdapter.js   # LangChain adapter for node-llama-cpp
│   ├── rag/
│   │   ├── vectorStore.js    # Faiss vector store management
│   │   ├── embeddings.js     # Custom embeddings with @xenova/transformers
│   │   └── documentLoader.js # Document loading and chunking
│   └── chat/
│       └── chatHandler.js    # RAG pipeline orchestration
├── knowledge-base/           # Place your documents here (.txt, .md)
├── faiss-store/              # Generated Faiss index (auto-created)
├── models/                   # Place GGUF model file here
└── package.json
```

## Configuration

All configuration is done via environment variables (see `.env.example`):

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

## License

ISC

## References

- [LangChain.js FaissStore Documentation](https://docs.langchain.com/oss/javascript/integrations/vectorstores/faiss)
- [node-llama-cpp Documentation](https://node-llama-cpp.withcat.ai/)
- [RakutenAI-2.0-mini-instruct-gguf Model](https://huggingface.co/mmnga/RakutenAI-2.0-mini-instruct-gguf)

