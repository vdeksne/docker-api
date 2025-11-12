import Redis from 'ioredis';
import { getConfig } from './config.js';

const cfg = getConfig();

let redis: Redis;

if (cfg.REDIS_URL) {
  redis = new Redis(cfg.REDIS_URL, {
    maxRetriesPerRequest: 3,
    enableReadyCheck: true,
  });
} else {
  redis = new Redis({
    host: cfg.REDIS_HOST || 'localhost',
    port: cfg.REDIS_PORT ? Number(cfg.REDIS_PORT) : 6379,
    password: cfg.REDIS_PASSWORD,
    maxRetriesPerRequest: 3,
    enableReadyCheck: true,
  });
}

export { redis };





