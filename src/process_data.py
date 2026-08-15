"""
Process raw FRED/BLS data into annual series for the simulation model.
Outputs: data/processed/annual_data.csv
"""

import csv
from datetime import datetime
from pathlib import Path
import statistics

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    """Read a FRED CSV and return list of (date, value) tuples."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = datetime.strptime(row["observation_date"], "%Y-%m-%d")
            val_col = [k for k in row.keys() if k != "observation_date"][0]
            val = float(row[val_col]) if row[val_col] not in (".", "") else None
            rows.append((date, val))
    return rows


def annualize(rows, method="mean"):
    """Aggregate monthly/quarterly rows into annual values."""
    by_year = {}
    for date, val in rows:
        if val is None:
            continue
        by_year.setdefault(date.year, []).append(val)
    out = {}
    for year, vals in by_year.items():
        if method == "mean":
            out[year] = statistics.mean(vals)
        elif method == "sum":
            out[year] = sum(vals)
        elif method == "end":
            out[year] = vals[-1]
    return out


def main():
    manemp = annualize(read_csv(RAW_DIR / "MANEMP.csv"), method="end")
    payems = annualize(read_csv(RAW_DIR / "PAYEMS.csv"), method="end")
    indpro = annualize(read_csv(RAW_DIR / "INDPRO.csv"), method="mean")
    awhman = annualize(read_csv(RAW_DIR / "AWHMAN.csv"), method="mean")
    mfg_wage = annualize(read_csv(RAW_DIR / "CES3000000003.csv"), method="mean")
    pvt_wage = annualize(read_csv(RAW_DIR / "CES0500000003.csv"), method="mean")
    outms = annualize(read_csv(RAW_DIR / "OUTMS.csv"), method="mean")

    # Robot stock data: IFR executive summary gives a few anchors; we use a
    # simplified interpolation based on published figures.
    # Sources:
    #   - Acemoglu & Restrepo (2017) NBER 23285: US stock increased fourfold
    #     between 1993 and 2007, and reached ~one robot per thousand workers.
    #   - IFR World Robotics 2020: 2019 US installations 33,339; 2018 peak 40,373.
    #   - We construct a plausible stock series from 1990 to 2020 using these anchors.
    # Because IFR's full country-year panel is proprietary, this is a transparent
    # synthetic baseline; the model accepts the user swapping in the official panel.
    robot_stock = {}
    # Anchors (thousands of units)
    anchors = {
        1990: 35.0,    # rough pre-boom level
        1993: 40.0,    # Acemoglu & Restrepo baseline
        2007: 160.0,   # fourfold increase
        2018: 265.0,   # supports ~33k installations/year with depreciation
        2019: 293.0,   # IFR 2020 executive summary: ~293k operational stock
        2020: 300.0,   # slight growth despite pandemic
    }
    years = sorted(anchors.keys())
    for i in range(len(years) - 1):
        y0, y1 = years[i], years[i + 1]
        v0, v1 = anchors[y0], anchors[y1]
        for y in range(y0, y1 + 1):
            robot_stock[y] = v0 + (v1 - v0) * (y - y0) / (y1 - y0)

    output_path = PROCESSED_DIR / "annual_data.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year",
            "manemp_thousands",
            "payems_thousands",
            "mfg_share_pct",
            "indpro_index",
            "outms_index",
            "awhman_hours",
            "mfg_wage_usd",
            "pvt_wage_usd",
            "mfg_wage_premium_pct",
            "robot_stock_thousands",
            "robots_per_thousand_workers",
        ])
        for year in range(1990, 2021):
            mfg = manemp.get(year)
            tot = payems.get(year)
            mfg_share = mfg / tot * 100 if mfg and tot else None
            ind = indpro.get(year)
            out = outms.get(year)
            awh = awhman.get(year)
            mw = mfg_wage.get(year)
            pw = pvt_wage.get(year)
            premium = (mw / pw - 1) * 100 if mw and pw else None
            rs = robot_stock.get(year)
            robots_per_k = rs / (mfg / 1000) if rs and mfg else None
            writer.writerow([
                year,
                mfg,
                tot,
                round(mfg_share, 2) if mfg_share else None,
                round(ind, 2) if ind else None,
                round(out, 2) if out else None,
                round(awh, 2) if awh else None,
                round(mw, 2) if mw else None,
                round(pw, 2) if pw else None,
                round(premium, 2) if premium else None,
                round(rs, 2) if rs else None,
                round(robots_per_k, 3) if robots_per_k else None,
            ])

    print(f"Annual data written to {output_path}")


if __name__ == "__main__":
    main()
