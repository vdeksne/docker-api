import { Router, Request, Response } from "express";
import { getWorldBankData } from "../services/population.js";

export const populationRouter = Router();

populationRouter.get("/", async (req: Request, res: Response) => {
  try {
    const country = String(req.query.country || "").toUpperCase();
    if (!country) {
      res
        .status(400)
        .json({ error: "country query param required (e.g., USA)" });
      return;
    }
    const indicator = String(
      req.query.indicator || "SP.POP.TOTL"
    ).toUpperCase();
    const from = req.query.from ? Number(req.query.from) : undefined;
    const to = req.query.to ? Number(req.query.to) : undefined;

    console.log(
      `[Route] Received request: indicator=${indicator}, country=${country}, from=${from}, to=${to}`
    );

    const { rows: dataRows, indicatorName } = await getWorldBankData(
      indicator,
      country,
      from,
      to
    );

    // Log for debugging
    console.log(
      `[Route] Fetched ${
        dataRows.length
      } rows for ${indicator}/${country}, indicatorName: ${
        indicatorName || "MISSING"
      }, first value: ${dataRows[0]?.value}`
    );

    // Ensure indicator and indicatorName are always included
    const finalIndicatorName = indicatorName || indicator;

    // Transform to match frontend expectation: only year and value
    const rows = dataRows.map((row) => ({
      year: row.year,
      value: row.value,
    }));

    // Always include indicator and indicatorName in response
    res.json({
      country,
      indicator: indicator, // Always use the requested indicator
      indicatorName: finalIndicatorName,
      from,
      to,
      rows,
    });
  } catch (e: any) {
    res
      .status(500)
      .json({ error: "Failed to fetch data", details: e?.message });
  }
});
