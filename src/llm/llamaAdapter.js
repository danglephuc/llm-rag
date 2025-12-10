import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { getModel, getContext } from './modelLoader.js';
import { LlamaChatSession } from 'node-llama-cpp';

/**
 * LangChain adapter for node-llama-cpp
 * Wraps node-llama-cpp's LlamaChatSession to work with LangChain's ChatModel interface
 */
export class ChatLlamaCpp extends BaseChatModel {
  constructor(fields = {}) {
    super(fields);
    this.model = getModel();
    this.context = getContext();
    this.maxTokens = fields.maxTokens || 512;
    this.temperature = fields.temperature || 0.7;
    
    if (!this.model || !this.context) {
      throw new Error('Model and context must be loaded before creating ChatLlamaCpp instance');
    }

    this.chatSession = new LlamaChatSession({
      contextSequence: this.context.getSequence(),
    });
  }

  _llmType() {
    return 'llama-cpp';
  }

  /**
   * Convert LangChain messages to a prompt string
   * @param {Array} messages - Array of LangChain message objects
   * @returns {string} - Formatted prompt string
   */
  _formatMessages(messages) {
    let prompt = '';
    
    for (const message of messages) {
      if (message instanceof SystemMessage) {
        prompt += `System: ${message.content}\n\n`;
      } else if (message instanceof HumanMessage) {
        prompt += `User: ${message.content}\n\n`;
      } else if (message instanceof AIMessage) {
        prompt += `Assistant: ${message.content}\n\n`;
      }
    }
    
    prompt += 'Assistant:';
    return prompt;
  }

  /**
   * Generate a response from the model
   * @param {Array} messages - Array of LangChain message objects
   * @param {Object} options - Generation options
   * @returns {Promise<Object>} - LangChain generation result
   */
  async _generate(messages, options = {}) {
    try {
      const prompt = this._formatMessages(messages);
      const maxTokens = options.maxTokens || this.maxTokens;
      const temperature = options.temperature !== undefined ? options.temperature : this.temperature;

      // Generate response using LlamaChatSession
      const response = await this.chatSession.prompt(prompt, {
        maxTokens,
        temperature,
        topP: 0.9,
        topK: 40,
      });

      // Create AIMessage from response
      const aiMessage = new AIMessage(response);

      return {
        generations: [
          {
            text: response,
            message: aiMessage,
          },
        ],
      };
    } catch (error) {
      console.error('Error generating response:', error);
      throw error;
    }
  }

  /**
   * Stream responses from the model (optional implementation)
   * @param {Array} messages - Array of LangChain message objects
   * @param {Object} options - Generation options
   * @returns {AsyncGenerator} - Stream of generation chunks
   */
  async *_streamResponseChunks(messages, options = {}) {
    const prompt = this._formatMessages(messages);
    const maxTokens = options.maxTokens || this.maxTokens;
    const temperature = options.temperature !== undefined ? options.temperature : this.temperature;

    // Use a queue-based approach for streaming
    const chunkQueue = [];
    let isComplete = false;
    let streamingError = null;

    // Start the streaming in the background
    const streamingPromise = this.chatSession
      .completePrompt(prompt, {
        maxTokens,
        temperature,
        topP: 0.9,
        topK: 40,
        onTextChunk: (chunk) => {
          // Push chunks to the queue as they arrive
          chunkQueue.push(chunk);
        },
      })
      .then(() => {
        isComplete = true;
      })
      .catch((err) => {
        streamingError = err;
        isComplete = true;
      });

    // Yield chunks from the queue as they arrive
    while (!isComplete || chunkQueue.length > 0) {
      if (chunkQueue.length > 0) {
        const chunk = chunkQueue.shift();
        yield new ChatGenerationChunk({
          text: chunk,
          message: new AIMessage(chunk),
        });
      } else if (!isComplete) {
        // Wait a bit before checking again
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
    }

    // Wait for the streaming to complete and handle any errors
    try {
      await streamingPromise;
    } catch (err) {
      // Error already captured in streamingError
    }

    if (streamingError) {
      throw new Error(`Streaming error: ${streamingError.message}`);
    }
  }
}

