import { Controller, Get, HttpStatus, Res } from '@nestjs/common';
import type { Response } from 'express';
import { RagService } from '../rag/rag.service';

@Controller('health')
export class HealthController {
  constructor(private readonly ragService: RagService) {}

  @Get()
  checkHealth(@Res() res: Response) {
    const isReady = this.ragService.isReady();

    if (isReady) {
      return res.status(HttpStatus.OK).json({
        status: 'healthy',
        rag: {
          initialized: true,
          ready: true,
        },
        timestamp: new Date().toISOString(),
      });
    } else {
      return res.status(HttpStatus.SERVICE_UNAVAILABLE).json({
        status: 'unhealthy',
        rag: {
          initialized: false,
          ready: false,
        },
        timestamp: new Date().toISOString(),
      });
    }
  }
}
