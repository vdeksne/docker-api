import "dotenv/config";
import express, { Request, Response } from "express";
import cors from "cors";
import { getConfig } from "./lib/config.js";
import { runMigrations } from "./lib/migrate.js";
import { populationRouter } from "./routes/population.js";
import { predictionsRouter } from "./routes/predictions.js";

const app = express();
app.use(cors());
app.use(express.json());

app.get("/api/health", (_req: Request, res: Response) => {
  res.json({ status: "ok" });
});

app.use("/api/population", populationRouter);
app.use("/api/predictions", predictionsRouter);

const config = getConfig();
const port = Number(config.PORT) || 4000;

runMigrations()
  .then(() => {
    app.listen(port, "0.0.0.0", () => {
      // eslint-disable-next-line no-console
      console.log(`Backend listening on port ${port}`);
    });
  })
  .catch((err) => {
    // eslint-disable-next-line no-console
    console.error("Migration failed", err);
    process.exit(1);
  });
