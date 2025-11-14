# docker-api

## World Bank Population App

React frontend + Express (Node.js, TypeScript) backend with Postgres database and Redis caching (compatible with AWS ElastiCache). Dockerized for local development.

## About the App

This application allows you to compare World Bank data across different countries and indicators. You can visualize historical trends and get AI-powered predictions for future values.

### AI-Powered Predictions

The app includes an AI-powered prediction system using Graph Neural Networks (GNN) that:

- **Automatically trains** on historical World Bank data (1990-2023)
- **Prioritizes Baltic countries** (Latvia, Lithuania, Estonia) for enhanced accuracy
- **Predicts future values** for any World Bank indicator
- **Uses 10+ years of historical data** for Baltic countries to improve prediction accuracy
- **Trains in the background** so the API is always available

The ML model is specifically optimized for Baltic countries (LVA, EST, LTU) with:

- Extended historical data collection (10 years vs 5 for other countries)
- Weighted recent years for better trend analysis
- Enhanced fallback mechanisms for missing data
- Automatic training on startup with 26 countries and 34 years of data

### Features

- Compare multiple countries and indicators simultaneously
- Visualize data with interactive charts
- Generate AI predictions for future indicator values
- Support for all World Bank indicators (population, GDP, life expectancy, etc.)
- Automatic data caching and persistence

## Services

- Frontend: React + Vite (served via Nginx in production container)
- Backend: Express + TypeScript
- Database: Postgres (stores countries, indicators, observations)
- Cache: Redis (can point to AWS ElastiCache in production)
- ML Service: FastAPI + PyTorch Geometric (GNN model for predictions)

## World Bank API

Data source: `https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL`

Backend endpoint wraps this with caching and persistence:

- GET `/api/population?country=USA&from=2000&to=2020`

## Run locally (Docker)

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend: http://localhost:4000/api/health
- ML Service: http://localhost:5001/health

**Note:** The ML service will automatically train the model in the background on first startup. This may take 5-10 minutes to collect training data and train the model. The API remains available during training.

Default local env (from compose):

- Postgres: `postgres://postgres:postgres@localhost:5432/worldbank`
- Redis: `redis://default:redispass@localhost:6379`

## Configure AWS ElastiCache (Redis)

In production, set one of:

- `REDIS_URL=redis://:<password>@<primary-endpoint>:6379`
- Or `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`

## Configure Postgres (e.g., RDS)

Set one of:

- `DATABASE_URL=postgres://<user>:<pass>@<host>:<port>/<db>`
- Or `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`

## Production deployment

- Frontend: deployed on Vercel at [https://frontend-nine-theta-37.vercel.app/](https://frontend-nine-theta-37.vercel.app/).
- Backend: Docker container on a DigitalOcean droplet exposing `http://139.59.138.164:8081`.
- Database: Neon serverless Postgres (project `docker-api`).
- Redis: single-node container on the droplet (`redis://default:redispass@localhost:6379`).

For Docker/local environments the backend falls back to the internal Postgres service (`postgres://postgres:postgres@db:5432/worldbank`) and Redis service (`redis://default:redispass@cache:6379`). Set `DATABASE_URL`, `REDIS_URL`, or `USE_NEON=true` explicitly when you want the API to talk to Neon/RediLabs instead.

### Rebuild & redeploy backend

```bash
ssh root@139.59.138.164
cd /root/docker-api
git pull
docker build -f Dockerfile.backend -t wb-backend:latest .
docker rm -f docker-api || true
docker run -d --name docker-api \
  -p 8081:4000 \
  -e NODE_ENV=production \
  -e PORT=4000 \
  -e USE_NEON=true \
  -e DATABASE_URL="postgresql://neondb_owner:npg_x6Sk8tyCZaRW@ep-silent-union-a4vy2o2n-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require" \
  -e REDIS_URL="redis://default:redispass@redis-12760.crce175.eu-north-1-1.ec2.cloud.redislabs.com:12760" \
  -e PYTHON_EXECUTABLE=python3 \
  -e ML_SCRIPT_PATH=/app/ml/run_prediction.py \
  -e ML_CACHE_TTL_MS=300000 \
  -e FRONTEND_ORIGIN="https://frontend-nine-theta-37.vercel.app" \
  wb-backend:latest
```

Health check: `curl http://139.59.138.164:8081/api/health`
docker ps
curl http://127.0.0.1:8081/api/health
curl https://139-59-138-164.sslip.io/api/health

### Frontend environment

- Copy `frontend/env.production.example` to `frontend/.env.production` (or configure in Vercel).
- Set `VITE_BACKEND_URL` to the droplet URL and trigger a new Vercel deploy.

## Development without Docker

Backend:

```bash
cd backend
npm install
# cp .env.example .env
npm run dev
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Notes

- On backend start, simple migrations create tables if they do not exist.
- Redis cache TTL is 1 hour per query key: `pop:<country>:<from>:<to>`.
- Frontend expects `VITE_BACKEND_URL`; default is `http://localhost:4000`.
- ML model automatically trains on startup if `AUTO_TRAIN=true` or model doesn't exist.
- Trained models are persisted in Docker volume `ml_models`.
- To manually trigger training: `POST http://localhost:5001/api/train`
