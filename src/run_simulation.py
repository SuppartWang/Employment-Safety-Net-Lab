"""
Run a minimal batch of simulations for the three policy scenarios.
Outputs CSV files and prints a summary table.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from model import ManufacturingModel


SCENARIOS = ["free_market", "moderate", "aggressive"]
SEEDS = [42, 123, 999]


def run_experiment():
    records = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            model = ManufacturingModel(scenario=scenario, seed=seed)
            model_df, _ = model.run()
            final = model_df.iloc[-1]
            records.append({
                "scenario": scenario,
                "seed": seed,
                "final_year": int(final["year"]),
                "total_employment": int(final["total_employment"]),
                "unemployment_rate": final["unemployment_rate"],
                "total_automation": final["total_automation"],
                "avg_wage": final["avg_wage"],
                "avg_profit": final["avg_profit"],
                "gini": final["gini"],
                "budget": final["budget"],
                "deficit_rounds": int(final["deficit_rounds"]),
            })

    df = pd.DataFrame(records)
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scenario_comparison.csv", index=False)

    print("=== Scenario comparison (3 seeds per scenario) ===")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_dir / 'scenario_comparison.csv'}")

    return df


if __name__ == "__main__":
    run_experiment()
