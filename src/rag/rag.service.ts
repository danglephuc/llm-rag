import { Injectable, OnModuleDestroy } from '@nestjs/common';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';

dotenv.config();

interface RAGConfig {
  modelPath: string;
  embeddingModel: string;
  knowledgeBaseDir: string;
  faissIndexPath: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  maxTokens: number;
  temperature: number;
}

@Injectable()
export class RagService implements OnModuleDestroy {
  private llm: any = null;
  private vectorStore: any = null;
  private chatHandler: any = null;
  private isInitialized = false;
  private initializationPromise: Promise<void> | null = null;
  private config: RAGConfig;

  constructor() {
    this.config = {
      modelPath:
        process.env.MODEL_PATH ||
        './models/rakutenai-2.0-mini-instruct-q4_k_m.gguf',
      embeddingModel:
        process.env.EMBEDDING_MODEL || 'Xenova/all-MiniLM-L6-v2',
      knowledgeBaseDir:
        process.env.KNOWLEDGE_BASE_DIR || './knowledge-base',
      faissIndexPath: process.env.FAISS_INDEX_PATH || './faiss-store',
      chunkSize: parseInt(process.env.CHUNK_SIZE || '1000'),
      chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || '200'),
      topK: parseInt(process.env.TOP_K || '3'),
      maxTokens: parseInt(process.env.MAX_TOKENS || '512'),
      temperature: parseFloat(process.env.TEMPERATURE || '0.7'),
    };
  }

  /**
   * Initialize the RAG system
   */
  async initialize(): Promise<void> {
    // If already initialized, return immediately
    if (this.isInitialized) {
      return;
    }

    // If initialization is in progress, wait for it
    if (this.initializationPromise) {
      return this.initializationPromise;
    }

    // Start initialization
    this.initializationPromise = this._doInitialize();
    await this.initializationPromise;
  }

  private async _doInitialize(): Promise<void> {
    try {
      // Dynamically import ES modules
      const { loadModel, cleanup: cleanupModel } = await import(
        '../llm/modelLoader.js'
      );
      const { ChatLlamaCpp } = await import('../llm/llamaAdapter.js');
      const { XenovaEmbeddings } = await import('./embeddings.js');
      const {
        initializeVectorStore,
        saveVectorStore,
      } = await import('./vectorStore.js');
      const { loadAndProcessDocuments } = await import('./documentLoader.js');
      const { ChatHandler } = await import('../chat/chatHandler.js');

      // Store cleanup function for later use
      this.cleanupModel = cleanupModel;

      console.log('Initializing RAG system...\n');

      // Step 1: Load the LLM model
      console.log('Step 1: Loading LLM model...');
      await loadModel(this.config.modelPath);
      this.llm = new ChatLlamaCpp({
        maxTokens: this.config.maxTokens,
        temperature: this.config.temperature,
      });
      console.log('✓ LLM model loaded\n');

      // Step 2: Initialize embeddings
      console.log('Step 2: Loading embedding model...');
      const embeddings = new XenovaEmbeddings({
        modelName: this.config.embeddingModel,
      });
      console.log('✓ Embedding model loaded\n');

      // Step 3: Initialize or load vector store
      console.log('Step 3: Initializing vector store...');
      const indexPath = this.config.faissIndexPath;

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

      if (indexExists) {
        // Load existing vector store
        const { FaissStore } = await import(
          '@langchain/community/vectorstores/faiss'
        );
        this.vectorStore = await FaissStore.load(
          resolvedIndexPath,
          embeddings,
        );
        console.log('✓ Vector store loaded from existing index\n');
      } else {
        // Create new vector store from documents
        console.log(
          'Vector store is new, loading documents from knowledge base...',
        );
        const documents = await loadAndProcessDocuments(
          this.config.knowledgeBaseDir,
          this.config.chunkSize,
          this.config.chunkOverlap,
        );

        if (documents.length > 0) {
          const { FaissStore } = await import(
            '@langchain/community/vectorstores/faiss'
          );
          this.vectorStore =
            await FaissStore.fromDocuments(documents, embeddings);
          await saveVectorStore(this.vectorStore, resolvedIndexPath);
          console.log(
            `✓ Created vector store with ${documents.length} document chunks\n`,
          );
        } else {
          // Create empty vector store if no documents found
          this.vectorStore = await initializeVectorStore(
            embeddings,
            resolvedIndexPath,
          );
          console.log(
            '⚠ No documents found in knowledge base directory, created empty vector store\n',
          );
        }
      }

      // Step 4: Create chat handler
      this.chatHandler = new ChatHandler(this.vectorStore, this.llm);
      this.isInitialized = true;
      console.log('✓ RAG system initialized successfully\n');
    } catch (error) {
      console.error('Error initializing RAG system:', error);
      this.initializationPromise = null;
      throw error;
    }
  }

  /**
   * Get the chat handler instance
   * Will initialize the system if not already initialized
   */
  async getChatHandler(): Promise<any> {
    if (!this.isInitialized) {
      await this.initialize();
    }
    if (!this.chatHandler) {
      throw new Error('Chat handler not available');
    }
    return this.chatHandler;
  }

  /**
   * Check if the RAG system is initialized
   */
  isReady(): boolean {
    return this.isInitialized && this.llm !== null && this.vectorStore !== null;
  }

  /**
   * Get the default topK value
   */
  getTopK(): number {
    return this.config.topK;
  }

  private cleanupModel: (() => void) | null = null;

  /**
   * Cleanup resources on module destroy
   */
  onModuleDestroy() {
    if (this.cleanupModel) {
      this.cleanupModel();
    }
  }
}
