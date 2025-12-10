import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Enable CORS for SSE connections
  app.enableCors({
    origin: true,
    credentials: true,
  });

  // Optional: Set global prefix for API routes
  // app.setGlobalPrefix('api');

  const port = process.env.PORT ?? 3000;
  await app.listen(port);
  console.log(`Application is running on: http://localhost:${port}`);
  console.log(`Health check: http://localhost:${port}/health`);
  console.log(`Warm-up endpoint: http://localhost:${port}/warmup`);
  console.log(`Chat stream endpoint: http://localhost:${port}/chat/stream`);
}
bootstrap();
