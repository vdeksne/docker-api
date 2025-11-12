export type AppConfig = {
  PORT: string;
  FRONTEND_ORIGIN: string;
  DATABASE_URL?: string;
  PGHOST?: string;
  PGPORT?: string;
  PGDATABASE?: string;
  PGUSER?: string;
  PGPASSWORD?: string;
  REDIS_URL?: string;
  REDIS_HOST?: string;
  REDIS_PORT?: string;
  REDIS_PASSWORD?: string;
};

const isProduction = process.env.NODE_ENV === 'production';

const defaultDbHost = isProduction ? 'db' : 'localhost';
const defaultRedisHost = isProduction ? 'cache' : 'localhost';

const defaultDatabaseUrl =
  process.env.DATABASE_URL ||
  (isProduction ? 'postgres://postgres:postgres@db:5432/worldbank' : undefined);

export function getConfig(): AppConfig {
  return {
    PORT: process.env.PORT || '4000',
    FRONTEND_ORIGIN: process.env.FRONTEND_ORIGIN || '*',
    DATABASE_URL: defaultDatabaseUrl,
    PGHOST: process.env.PGHOST || defaultDbHost,
    PGPORT: process.env.PGPORT || '5432',
    PGDATABASE: process.env.PGDATABASE || (isProduction ? 'worldbank' : undefined),
    PGUSER: process.env.PGUSER || (isProduction ? 'postgres' : undefined),
    PGPASSWORD: process.env.PGPASSWORD || (isProduction ? 'postgres' : undefined),
    REDIS_URL:
      process.env.REDIS_URL ||
      (isProduction ? 'redis://default:redispass@cache:6379' : undefined),
    REDIS_HOST: process.env.REDIS_HOST || defaultRedisHost,
    REDIS_PORT: process.env.REDIS_PORT || '6379',
    REDIS_PASSWORD: process.env.REDIS_PASSWORD || (isProduction ? 'redispass' : undefined),
  };
}





