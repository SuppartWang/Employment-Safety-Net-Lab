"""
Generate visualizations from the simulation results and raw macro data.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import ManufacturingModel


FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_annual():
    data = []
    with open(Path(__file__).parent.parent / "data" / "processed" / "annual_data.csv", "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) if v != "" and k != "year" else int(v) if k == "year" else None for k, v in row.items()})
    return data


def plot_macro_baseline():
    data = load_annual()
    years = [d["year"] for d in data]
    mfg_share = [d["mfg_share_pct"] for d in data]
    robots_per_k = [d["robots_per_thousand_workers"] for d in data]
    premium = [d["mfg_wage_premium_pct"] for d in data]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(years, mfg_share, "b-o", label="Manufacturing employment share (%)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Manufacturing employment share (%)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(years, robots_per_k, "r--s", label="Robots per 1,000 workers")
    ax2.set_ylabel("Robots per 1,000 workers", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    fig.suptitle("US manufacturing employment share vs robot density (1990-2020)")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_macro_baseline.png", dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR / 'fig1_macro_baseline.png'}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(years, premium, "g-o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Manufacturing wage premium over private sector (%)")
    ax.set_title("Manufacturing wage premium (2006-2020, BLS CES data)")
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_wage_premium.png", dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR / 'fig2_wage_premium.png'}")


def plot_simulation_trajectories():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for scenario in ["free_market", "moderate", "aggressive"]:
        model = ManufacturingModel(scenario=scenario, seed=42)
        df, _ = model.run()
        axes[0, 0].plot(df["year"], df["total_employment"] / 1e6, label=scenario)
        axes[0, 1].plot(df["year"], df["total_automation"], label=scenario)
        axes[1, 0].plot(df["year"], df["unemployment_rate"] * 100, label=scenario)
        axes[1, 1].plot(df["year"], df["budget"] / 1e6, label=scenario)

    axes[0, 0].set_title("Total employment (millions)")
    axes[0, 0].set_ylabel("Millions")
    axes[0, 1].set_title("Average automation rate")
    axes[0, 1].set_ylabel("Automation share")
    axes[1, 0].set_title("Unemployment rate (%)")
    axes[1, 0].set_ylabel("Percent")
    axes[1, 1].set_title("Government budget (millions USD)")
    axes[1, 1].set_ylabel("Millions")

    for ax in axes.flat:
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Simulation trajectories under three policy scenarios (seed=42)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_simulation_trajectories.png", dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR / 'fig3_simulation_trajectories.png'}")


def plot_agent_breakdown():
    model = ManufacturingModel(scenario="moderate", seed=42)
    _, agent_df = model.run()
    agent_df = agent_df.reset_index()
    agent_df = agent_df[agent_df["size_class"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for size in ["small", "medium", "large"]:
        sub = agent_df[agent_df["size_class"] == size]
        mean_emp = sub.groupby("Step")["employees"].mean() / 1e6
        mean_auto = sub.groupby("Step")["automation"].mean()
        years = 1990 + np.arange(len(mean_emp))
        axes[0].plot(years, mean_emp, label=size)
        axes[1].plot(years, mean_auto, label=size)

    axes[0].set_title("Employment by firm size")
    axes[0].set_ylabel("Millions")
    axes[1].set_title("Automation by firm size")
    axes[1].set_ylabel("Automation share")

    for ax in axes:
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_agent_breakdown.png", dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR / 'fig4_agent_breakdown.png'}")


if __name__ == "__main__":
    plot_macro_baseline()
    plot_simulation_trajectories()
    plot_agent_breakdown()
