import { similaritySearch } from '../rag/vectorStore.js';
import { ChatLlamaCpp } from '../llm/llamaAdapter.js';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

/**
 * Handle chat interactions with RAG
 */
export class ChatHandler {
  constructor(vectorStore, llm) {
    this.vectorStore = vectorStore;
    this.llm = llm;
  }

  /**
   * Generate a response using RAG
   * @param {string} query - User query
   * @param {number} topK - Number of documents to retrieve
   * @returns {Promise<string>} - Generated response
   */
  async generateResponse(query, topK = 3) {
    try {
      // Step 1: Retrieve relevant documents from vector store
      console.log(`Retrieving top ${topK} relevant documents for query: "${query}"`);
      const relevantDocs = await similaritySearch(this.vectorStore, query, topK);
      
      // Step 2: Construct context from retrieved documents
      const context = relevantDocs
        .map((doc, index) => `[Document ${index + 1}]\n${doc.pageContent}`)
        .join('\n\n');

      // Step 3: Construct prompt with context and query
      const systemPrompt = `You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, use your knowledge to provide a helpful response.

Context:
${context}

Answer the user's question based on the context provided above.`;

      const messages = [
        new SystemMessage(systemPrompt),
        new HumanMessage(query),
      ];

      // Step 4: Generate response using LLM
      console.log('Generating response...');
      const result = await this.llm.invoke(messages);
      
      // Step 5: Extract and return the response text
      const response = result.content || result.text || '';
      return response;
    } catch (error) {
      console.error('Error generating response:', error);
      throw error;
    }
  }
}

