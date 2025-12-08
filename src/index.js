import dotenv from 'dotenv';
import readline from 'readline';
import { loadModel } from './llm/modelLoader.js';
import { ChatLlamaCpp } from './llm/llamaAdapter.js';
import { XenovaEmbeddings } from './rag/embeddings.js';
import { initializeVectorStore, saveVectorStore, addDocuments } from './rag/vectorStore.js';
import { loadAndProcessDocuments } from './rag/documentLoader.js';
import { ChatHandler } from './chat/chatHandler.js';
import { cleanup } from './llm/modelLoader.js';

// Load environment variables
dotenv.config();

// Configuration
const config = {
  modelPath: process.env.MODEL_PATH || './models/rakutenai-2.0-mini-instruct-q4_k_m.gguf',
  embeddingModel: process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2',
  knowledgeBaseDir: process.env.KNOWLEDGE_BASE_DIR || './knowledge-base',
  faissIndexPath: process.env.FAISS_INDEX_PATH || './faiss-store',
  chunkSize: parseInt(process.env.CHUNK_SIZE || '1000'),
  chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || '200'),
  topK: parseInt(process.env.TOP_K || '3'),
  maxTokens: parseInt(process.env.MAX_TOKENS || '512'),
  temperature: parseFloat(process.env.TEMPERATURE || '0.7'),
};

/**
 * Initialize the RAG system
 */
async function initializeRAG() {
  console.log('Initializing RAG system...\n');

  // Step 1: Load the LLM model
  console.log('Step 1: Loading LLM model...');
  await loadModel(config.modelPath);
  const llm = new ChatLlamaCpp({
    maxTokens: config.maxTokens,
    temperature: config.temperature,
  });
  console.log('✓ LLM model loaded\n');

  // Step 2: Initialize embeddings
  console.log('Step 2: Loading embedding model...');
  const embeddings = new XenovaEmbeddings({
    modelName: config.embeddingModel,
  });
  console.log('✓ Embedding model loaded\n');

  // Step 3: Initialize or load vector store
  console.log('Step 3: Initializing vector store...');
  const fs = await import('fs');
  const path = await import('path');
  const indexPath = config.faissIndexPath;
  
  // FaissStore saves to a directory, check if the directory exists and contains index files
  const resolvedIndexPath = path.isAbsolute(indexPath) 
    ? indexPath 
    : path.resolve(process.cwd(), indexPath);
  
  // Ensure directory exists
  if (!fs.existsSync(resolvedIndexPath)) {
    fs.mkdirSync(resolvedIndexPath, { recursive: true });
  }
  
  const indexFile = path.join(resolvedIndexPath, 'faiss.index');
  const indexExists = fs.existsSync(indexFile);
  
  let vectorStore;
  
  if (indexExists) {
    // Load existing vector store
    const { FaissStore } = await import('@langchain/community/vectorstores/faiss');
    vectorStore = await FaissStore.load(resolvedIndexPath, embeddings);
    console.log('✓ Vector store loaded from existing index\n');
  } else {
    // Create new vector store from documents
    console.log('Vector store is new, loading documents from knowledge base...');
    const documents = await loadAndProcessDocuments(
      config.knowledgeBaseDir,
      config.chunkSize,
      config.chunkOverlap
    );
    
    if (documents.length > 0) {
      const { FaissStore } = await import('@langchain/community/vectorstores/faiss');
      // fromDocuments signature: fromDocuments(docs, embeddings, dbConfig)
      vectorStore = await FaissStore.fromDocuments(documents, embeddings);
      await saveVectorStore(vectorStore, resolvedIndexPath);
      console.log(`✓ Created vector store with ${documents.length} document chunks\n`);
    } else {
      // Create empty vector store if no documents found
      vectorStore = await initializeVectorStore(embeddings, resolvedIndexPath);
      console.log('⚠ No documents found in knowledge base directory, created empty vector store\n');
    }
  }

  return { llm, vectorStore };
}

/**
 * Create and run the CLI interface
 */
async function runCLI() {
  try {
    // Initialize RAG system
    const { llm, vectorStore } = await initializeRAG();
    const chatHandler = new ChatHandler(vectorStore, llm);

    // Create readline interface
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    console.log('='.repeat(60));
    console.log('RAG Chat System Ready!');
    console.log('Type your questions (or "exit" to quit)');
    console.log('='.repeat(60));
    console.log('');

    // Prompt for user input
    const askQuestion = () => {
      rl.question('You: ', async (query) => {
        if (query.toLowerCase() === 'exit' || query.toLowerCase() === 'quit') {
          console.log('\nGoodbye!');
          rl.close();
          cleanup();
          process.exit(0);
        }

        if (query.trim() === '') {
          askQuestion();
          return;
        }

        try {
          process.stdout.write('\nAI: ');
          const response = await chatHandler.generateResponse(query, config.topK);
          console.log(response);
          console.log('');
        } catch (error) {
          console.error('Error:', error.message);
          console.log('');
        }

        askQuestion();
      });
    };

    askQuestion();
  } catch (error) {
    console.error('Fatal error:', error);
    cleanup();
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\nShutting down...');
  cleanup();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n\nShutting down...');
  cleanup();
  process.exit(0);
});

// Start the application
runCLI().catch((error) => {
  console.error('Failed to start application:', error);
  cleanup();
  process.exit(1);
});

