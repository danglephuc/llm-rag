import { Document } from '@langchain/core/documents';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { readdir, readFile } from 'fs/promises';
import path from 'path';

/**
 * Load text files from a directory
 * @param {string} dirPath - Path to the directory containing text files
 * @param {string[]} extensions - File extensions to load (default: ['.txt', '.md'])
 * @returns {Promise<Document[]>} - Array of Document objects
 */
export async function loadDocumentsFromDirectory(dirPath, extensions = ['.txt', '.md']) {
  const resolvedPath = path.isAbsolute(dirPath) 
    ? dirPath 
    : path.resolve(process.cwd(), dirPath);

  const documents = [];
  
  try {
    const files = await readdir(resolvedPath);
    
    for (const file of files) {
      const filePath = path.join(resolvedPath, file);
      const ext = path.extname(file).toLowerCase();
      
      if (extensions.includes(ext)) {
        try {
          const content = await readFile(filePath, 'utf-8');
          documents.push(new Document({
            pageContent: content,
            metadata: {
              source: filePath,
              filename: file,
            },
          }));
        } catch (error) {
          console.warn(`Error reading file ${filePath}:`, error.message);
        }
      }
    }
    
    console.log(`Loaded ${documents.length} documents from ${resolvedPath}`);
    return documents;
  } catch (error) {
    console.error(`Error loading documents from ${resolvedPath}:`, error);
    throw error;
  }
}

/**
 * Split documents into chunks
 * @param {Document[]} documents - Array of Document objects
 * @param {number} chunkSize - Size of each chunk (default: 1000)
 * @param {number} chunkOverlap - Overlap between chunks (default: 200)
 * @returns {Promise<Document[]>} - Array of chunked Document objects
 */
export async function splitDocuments(documents, chunkSize = 1000, chunkOverlap = 200) {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
    separators: ['\n\n', '\n', ' ', ''],
  });

  try {
    const chunks = await textSplitter.splitDocuments(documents);
    console.log(`Split ${documents.length} documents into ${chunks.length} chunks`);
    return chunks;
  } catch (error) {
    console.error('Error splitting documents:', error);
    throw error;
  }
}

/**
 * Load and process documents from directory
 * @param {string} dirPath - Path to the directory
 * @param {number} chunkSize - Chunk size
 * @param {number} chunkOverlap - Chunk overlap
 * @returns {Promise<Document[]>} - Array of processed Document chunks
 */
export async function loadAndProcessDocuments(dirPath, chunkSize = 1000, chunkOverlap = 200) {
  const documents = await loadDocumentsFromDirectory(dirPath);
  const chunks = await splitDocuments(documents, chunkSize, chunkOverlap);
  return chunks;
}

