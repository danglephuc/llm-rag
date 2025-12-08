import { Embeddings } from '@langchain/core/embeddings';
import { pipeline } from '@xenova/transformers';

/**
 * Custom embeddings class using @xenova/transformers
 * Extends LangChain's Embeddings base class for compatibility
 */
export class XenovaEmbeddings extends Embeddings {
  modelName;
  pipeline;
  
  constructor(fields = {}) {
    super(fields);
    this.modelName = fields.modelName || 'Xenova/all-MiniLM-L6-v2';
    this.pipeline = null;
  }

  /**
   * Initialize the embedding pipeline
   * @returns {Promise<void>}
   */
  async _initializePipeline() {
    if (!this.pipeline) {
      console.log(`Loading embedding model: ${this.modelName}`);
      this.pipeline = await pipeline('feature-extraction', this.modelName, {
        quantized: true,
      });
      console.log('Embedding model loaded successfully');
    }
  }

  /**
   * Embed a single query text
   * @param {string} text - Text to embed
   * @returns {Promise<number[]>} - Embedding vector
   */
  async embedQuery(text) {
    await this._initializePipeline();
    
    try {
      const output = await this.pipeline(text, {
        pooling: 'mean',
        normalize: true,
      });
      
      // Convert tensor to array
      return Array.from(output.data);
    } catch (error) {
      console.error('Error embedding query:', error);
      throw error;
    }
  }

  /**
   * Embed multiple documents
   * @param {string[]} texts - Array of texts to embed
   * @returns {Promise<number[][]>} - Array of embedding vectors
   */
  async embedDocuments(texts) {
    await this._initializePipeline();
    
    try {
      const embeddings = [];
      
      // Process texts in batches to avoid memory issues
      const batchSize = 10;
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        const batchPromises = batch.map(async (text) => {
          const output = await this.pipeline(text, {
            pooling: 'mean',
            normalize: true,
          });
          return Array.from(output.data);
        });
        
        const batchEmbeddings = await Promise.all(batchPromises);
        embeddings.push(...batchEmbeddings);
      }
      
      return embeddings;
    } catch (error) {
      console.error('Error embedding documents:', error);
      throw error;
    }
  }
}

