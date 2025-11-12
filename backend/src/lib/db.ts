import { Pool } from 'pg';
import { getConfig } from './config.js';

const cfg = getConfig();

export const pool = new Pool({
  connectionString: cfg.DATABASE_URL,
  host: cfg.PGHOST,
  port: cfg.PGPORT ? Number(cfg.PGPORT) : undefined,
  database: cfg.PGDATABASE,
  user: cfg.PGUSER,
  password: cfg.PGPASSWORD,
  ssl: process.env.PGSSL === 'true' ? { rejectUnauthorized: false } : undefined,
});

export async function withClient<T>(fn: (client: any) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try {
    return await fn(client);
  } finally {
    client.release();
  }
}





