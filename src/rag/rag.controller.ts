import { Controller, Post, HttpStatus, Res } from '@nestjs/common';
import type { Response } from 'express';
import { RagService } from './rag.service';

@Controller('warmup')
export class RagController {
  constructor(private readonly ragService: RagService) {}

  @Post()
  async warmup(@Res() res: Response) {
    try {
      // Check if already initialized
      if (this.ragService.isReady()) {
        return res.status(HttpStatus.OK).json({
          status: 'success',
          message: 'RAG system is already initialized',
          timestamp: new Date().toISOString(),
        });
      }

      // Initialize the RAG system
      await this.ragService.initialize();

      return res.status(HttpStatus.OK).json({
        status: 'success',
        message: 'RAG system initialized successfully',
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      return res.status(HttpStatus.INTERNAL_SERVER_ERROR).json({
        status: 'error',
        message:
          error instanceof Error ? error.message : 'Failed to initialize RAG system',
        timestamp: new Date().toISOString(),
      });
    }
  }
}
