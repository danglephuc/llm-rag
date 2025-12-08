import { getLlama } from 'node-llama-cpp';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let llama = null;
let model = null;
let context = null;

/**
 * Initialize and load the GGUF model
 * @param {string} modelPath - Path to the GGUF model file
 * @returns {Promise<{model: LlamaModel, context: LlamaContext}>}
 */
export async function loadModel(modelPath) {
  try {
    if (!llama) {
      llama = await getLlama();
    }

    if (!model) {
      const resolvedPath = path.isAbsolute(modelPath) 
        ? modelPath 
        : path.resolve(process.cwd(), modelPath);
      
      console.log(`Loading model from: ${resolvedPath}`);
      
      // Use llama.loadModel() instead of new LlamaModel()
      model = await llama.loadModel({
        modelPath: resolvedPath,
      });

      console.log('Model loaded successfully');
    }

    if (!context) {
      context = await model.createContext({
        contextSize: 4096,
      });

      console.log('Context created successfully');
    }

    return { model, context };
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

/**
 * Get the current model instance
 * @returns {LlamaModel|null}
 */
export function getModel() {
  return model;
}

/**
 * Get the current context instance
 * @returns {LlamaContext|null}
 */
export function getContext() {
  return context;
}

/**
 * Clean up model resources
 */
export function cleanup() {
  if (context) {
    context.dispose();
    context = null;
  }
  if (model) {
    model.dispose();
    model = null;
  }
  llama = null;
}

