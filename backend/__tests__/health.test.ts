import request from 'supertest';
import express from 'express';
import cors from 'cors';
import { populationRouter } from '../src/routes/population';
import { predictionsRouter } from '../src/routes/predictions';

// Minimal app wiring mirrors src/index.ts without starting a server
function createApp() {
  const app = express();
  app.use(cors());
  app.use(express.json());
  app.get('/api/health', (_req, res) =>
    res.json({ ok: true, service: 'backend' })
  );
  app.use('/api/population', populationRouter);
  app.use('/api/predictions', predictionsRouter);
  return app;
}

describe('health endpoint', () => {
  const app = createApp();
  it('returns ok true', async () => {
    const res = await request(app).get('/api/health');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('ok', true);
  });
});

