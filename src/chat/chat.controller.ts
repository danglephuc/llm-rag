import { Controller, Post, Body, Sse, MessageEvent } from '@nestjs/common';
import { Observable } from 'rxjs';
import { RagService } from '../rag/rag.service';

interface ChatRequestDto {
  query: string;
  topK?: number;
}

@Controller('chat')
export class ChatController {
  constructor(private readonly ragService: RagService) {}

  @Post('stream')
  @Sse('stream')
  async streamChat(@Body() body: ChatRequestDto): Promise<Observable<MessageEvent>> {
    const { query, topK } = body;

    if (!query || typeof query !== 'string' || query.trim() === '') {
      return new Observable((observer) => {
        observer.next({
          data: JSON.stringify({ type: 'error', message: 'Query is required and must be a non-empty string' }),
        });
        observer.complete();
      });
    }

    return new Observable((observer) => {
      (async () => {
        try {
          const chatHandler = await this.ragService.getChatHandler();
          const k = topK || this.ragService.getTopK();

          // Stream the response
          for await (const chunk of chatHandler.streamResponse(query, k)) {
            observer.next({
              data: JSON.stringify({ type: 'chunk', content: chunk }),
            });
          }

          // Send completion event
          observer.next({
            data: JSON.stringify({ type: 'done', message: 'Stream completed' }),
          });
          observer.complete();
        } catch (error) {
          observer.next({
            data: JSON.stringify({
              type: 'error',
              message: error instanceof Error ? error.message : 'Unknown error occurred',
            }),
          });
          observer.complete();
        }
      })();
    });
  }
}
