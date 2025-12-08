import { FaissStore } from '@langchain/community/vectorstores/faiss';
import path from 'path';
import { existsSync, mkdirSync } from 'fs';

/**
 * Initialize or load a Faiss vector store
 * @param {Object} embeddings - LangChain embeddings instance
 * @param {string} indexPath - Path to save/load the Faiss index
 * @returns {Promise<FaissStore>} - FaissStore instance
 */
export async function initializeVectorStore(embeddings, indexPath) {
  const resolvedPath = path.isAbsolute(indexPath) 
    ? indexPath 
    : path.resolve(process.cwd(), indexPath);

  // Create directory if it doesn't exist
  const dir = path.dirname(resolvedPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  // Check if index already exists
  if (existsSync(resolvedPath)) {
    try {
      console.log(`Loading existing Faiss index from: ${resolvedPath}`);
      const vectorStore = await FaissStore.load(resolvedPath, embeddings);
      console.log('Faiss index loaded successfully');
      return vectorStore;
    } catch (error) {
      console.warn('Error loading existing index, creating new one:', error.message);
    }
  }

  // Create new empty vector store
  console.log('Creating new Faiss vector store');
  const vectorStore = new FaissStore(embeddings, {});
  return vectorStore;
}

/**
 * Save the vector store to disk
 * @param {FaissStore} vectorStore - FaissStore instance to save
 * @param {string} indexPath - Path to save the index
 * @returns {Promise<void>}
 */
export async function saveVectorStore(vectorStore, indexPath) {
  const resolvedPath = path.isAbsolute(indexPath) 
    ? indexPath 
    : path.resolve(process.cwd(), indexPath);

  const dir = path.dirname(resolvedPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  try {
    console.log(`Saving Faiss index to: ${resolvedPath}`);
    await vectorStore.save(resolvedPath);
    console.log('Faiss index saved successfully');
  } catch (error) {
    console.error('Error saving Faiss index:', error);
    throw error;
  }
}

/**
 * Add documents to the vector store
 * @param {FaissStore} vectorStore - FaissStore instance
 * @param {Array} documents - Array of LangChain Document objects
 * @returns {Promise<string[]>} - Array of document IDs
 */
export async function addDocuments(vectorStore, documents) {
  try {
    const ids = await vectorStore.addDocuments(documents);
    console.log(`Added ${documents.length} documents to vector store`);
    return ids;
  } catch (error) {
    console.error('Error adding documents:', error);
    throw error;
  }
}

/**
 * Search for similar documents
 * @param {FaissStore} vectorStore - FaissStore instance
 * @param {string} query - Query string
 * @param {number} k - Number of documents to retrieve
 * @returns {Promise<Array>} - Array of similar documents
 */
export async function similaritySearch(vectorStore, query, k = 3) {
  try {
    const results = await vectorStore.similaritySearch(query, k);
    return results;
  } catch (error) {
    console.error('Error performing similarity search:', error);
    throw error;
  }
}

