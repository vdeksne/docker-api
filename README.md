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
