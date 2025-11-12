FROM node:20-alpine AS base
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:20-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev || npm install --omit=dev
COPY --from=base /app/dist ./dist
EXPOSE 4000
CMD ["node", "dist/index.js"]



