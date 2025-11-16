# docker-api

## World Bank Population App

React frontend + Express (Node.js, TypeScript) backend with Postgres database and Redis caching (compatible with AWS ElastiCache). Dockerized for local development.

## About the App

The UI shows World Bank data for any supported country/indicator pair and lets you compare trends over time. The backend also exposes a prediction endpoint that uses a graph model trained on historical data, with some extra weight on Baltic countries (LVA, EST, LTU) because they are the primary focus of this project.

### Features

- Compare multiple countries/indicators side by side
- Chart historical data pulled from the World Bank API
- Request population or generic indicator predictions
- Cache responses in Redis so repeated calls stay fast

## Services

- Frontend: React + Vite (served by Nginx in production)
- Backend: Express + TypeScript
- Database: Postgres (observations, indicators, metadata)
- Cache: Redis (local container for dev; Redis Labs/Elasticache in prod)
- ML service: FastAPI + PyTorch Geometric (prediction jobs)

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

## Production deployment (current setup)

- Frontend: Vercel `https://frontend-nine-theta-37.vercel.app/`
- Backend: Docker container on DigitalOcean droplet (port 8081 -> 4000)
- Database: Neon serverless Postgres
- Redis: Redis Labs instance (`redis://default:…@redis-12760…:12760`)

When running with `docker compose` locally, the backend defaults to the internal Postgres (`postgres://postgres:postgres@db:5432/worldbank`) and Redis (`redis://default:redispass@cache:6379`). Set `DATABASE_URL`, `REDIS_URL`, or `USE_NEON=true` to point at the hosted services instead. The backend discovers the model API via `ML_SERVICE_URL` (defaults to `http://ml-service:5000` when using Compose).

### Build & run the ML service (DigitalOcean)

```bash
ssh root@139.59.138.164
cd /root/docker-api/ml-service
git pull
docker build -t wb-ml-service:latest .
docker rm -f ml-service || true
docker run -d --name ml-service \
  --restart unless-stopped \
  -p 5001:5000 \
  -e REDIS_URL="redis://default:beZtcrGSW1xiwb7XgD3A6AiZLj70pmAU@redis-12760.crce175.eu-north-1-1.ec2.cloud.redislabs.com:12760" \
  -e DEVICE=cpu \
  -e AUTO_TRAIN=true \
  wb-ml-service:latest
```

Health checks:

```bash
docker ps
docker logs -f ml-service
curl http://127.0.0.1:5001/health
```

To keep the service private, create a Docker network (for example `docker network create wb-prod`) and start both containers with `--network wb-prod`. Then you can drop `-p 5001:5000` and the backend can call `http://ml-service:5000`.

### Rebuild & redeploy backend

```bash
ssh root@139.59.138.164
cd /root/docker-api
git pull
docker build -f Dockerfile.backend -t wb-backend:latest .
docker rm -f docker-api || true
docker run -d --name docker-api \
  --network wb-prod \
  -p 8081:4000 \
  -e NODE_ENV=production \
  -e PORT=4000 \
  -e USE_NEON=true \
  -e DATABASE_URL="postgresql://neondb_owner:npg_x6Sk8tyCZaRW@ep-silent-union-a4vy2o2n-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require" \
  -e REDIS_URL="redis://default:beZtcrGSW1xiwb7XgD3A6AiZLj70pmAU@redis-12760.crce175.eu-north-1-1.ec2.cloud.redislabs.com:12760" \
  -e PYTHON_EXECUTABLE=python3 \
  -e ML_SCRIPT_PATH=/app/ml/run_prediction.py \
  -e ML_SERVICE_URL="http://ml-service:5000" \
  -e ML_CACHE_TTL_MS=300000 \
  -e FRONTEND_ORIGIN="https://frontend-nine-theta-37.vercel.app" \
  wb-backend:latest
```

Useful checks once it’s running:

- `docker ps`
- `curl http://127.0.0.1:8081/api/health`
- `curl https://139-59-138-164.sslip.io/api/health` (fronted by Nginx/Let’s Encrypt)

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
