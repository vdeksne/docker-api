import { withClient } from './db.js';

export async function runMigrations(): Promise<void> {
  await withClient(async (client) => {
    await client.query(`
      CREATE TABLE IF NOT EXISTS countries (
        id SERIAL PRIMARY KEY,
        iso3_code VARCHAR(3) UNIQUE NOT NULL,
        name TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS indicators (
        id SERIAL PRIMARY KEY,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS observations (
        id BIGSERIAL PRIMARY KEY,
        country_id INTEGER NOT NULL REFERENCES countries(id) ON DELETE CASCADE,
        indicator_id INTEGER NOT NULL REFERENCES indicators(id) ON DELETE CASCADE,
        year INTEGER NOT NULL,
        value DOUBLE PRECISION,
        UNIQUE(country_id, indicator_id, year)
      );
      CREATE INDEX IF NOT EXISTS obs_country_indicator_year_idx ON observations(country_id, indicator_id, year);
    `);
  });
}





