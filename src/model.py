"""
Mesa-based multi-agent simulation of US manufacturing automation shock (1990-2020).

This version uses a demand-constrained model:
- Each firm has a fixed market share.
- Industry output is driven by the FRED INDPRO manufacturing index.
- Firms must produce enough effective output to meet their demand share.
- Automation augments labor productivity, allowing firms to substitute capital for labor.
- Large firms optimize over a 3-year horizon; small/medium firms are myopic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_annual_data() -> Dict[int, Dict[str, float]]:
    """Load annual macro data processed by process_data.py."""
    data = {}
    with open(DATA_DIR / "annual_data.csv", "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            record = {}
            for key, val in row.items():
                if key == "year":
                    continue
                if val == "":
                    record[key] = None
                else:
                    record[key] = float(val)
            data[year] = record
    return data


class FirmAgent(Agent):
    """A manufacturing firm of size small, medium, or large."""

    MARKET_SHARES = {"small": 0.03, "medium": 0.10, "large": 0.30}

    def __init__(
        self,
        unique_id: int,
        model: ManufacturingModel,
        size_class: str,
        initial_employees: int,
        initial_wage: float,
        initial_automation: float,
        productivity: float,
        horizon: int = 1,
    ):
        super().__init__(unique_id, model)
        self.size_class = size_class
        self.employees = initial_employees
        self.wage = initial_wage
        self.automation = initial_automation
        self.productivity = productivity
        self.horizon = horizon
        self.market_share = self.MARKET_SHARES[size_class]
        self.profit = 0.0
        self.revenue = 0.0

    def required_output(self, demand: float) -> float:
        """Demand share the firm must satisfy."""
        return self.market_share * demand * self.model.demand_scale

    def actual_output(self, employees: float, automation: float) -> float:
        """Effective production capacity."""
        return employees * self.productivity * (1 + automation * self.productivity)

    def compute_revenue(self, demand: float, employees: float, automation: float) -> float:
        """Revenue is demand-constrained; slight overproduction is not rewarded."""
        output = self.actual_output(employees, automation)
        required = self.required_output(demand)
        sellable = min(output, required * 1.05)  # 5% slack before revenue cap
        return sellable * self.model.price_level

    def compute_profit(
        self,
        demand: float,
        robot_cost: float,
        tax_rate: float,
        employees: float,
        wage: float,
        automation: float,
    ) -> float:
        revenue = self.compute_revenue(demand, employees, automation)
        # Government policies affect firm costs
        subsidy_rate = self.model.gov.s if self.size_class != "large" else 0.0
        credit_rate = self.model.gov.r
        effective_wage = wage * (1 - subsidy_rate)
        effective_robot_cost = robot_cost * (1 - credit_rate)
        labor_cost = employees * effective_wage
        automation_cost = automation * effective_robot_cost * employees
        tax = revenue * tax_rate
        return revenue - labor_cost - automation_cost - tax

    def candidate_actions(self) -> List[Tuple[float, float, float]]:
        """Enumerate discrete candidate actions for (automation_delta, hiring_ratio, wage_delta)."""
        automation_choices = np.linspace(-0.05, 0.25, 7)
        hiring_choices = np.linspace(-0.15, 0.10, 8)
        wage_delta_choices = np.linspace(-0.03, 0.05, 5)
        actions = []
        for da in automation_choices:
            for dh_ratio in hiring_choices:
                for dw in wage_delta_choices:
                    actions.append((da, float(dh_ratio), dw))
        return actions

    def step(self):
        """Choose action that maximizes discounted expected profit."""
        current_year = self.model.year
        current_demand = self.model.annual_data[current_year]["indpro_index"]
        robot_cost = self.model.robot_cost
        tax_rate = self.model.tax_rate

        best_action = None
        best_score = -1e18

        for da, dh_ratio, dw in self.candidate_actions():
            new_automation = np.clip(self.automation + da, 0.0, 1.0)
            new_employees = int(np.clip(self.employees * (1 + dh_ratio), 1000, None))
            new_wage = self.wage * (1 + dw)

            score = 0.0
            for h in range(self.horizon):
                year = min(current_year + h, 2020)
                demand = self.model.annual_data[year]["indpro_index"]
                future_robot_cost = robot_cost * (1 - 0.03) ** h
                profit = self.compute_profit(
                    demand, future_robot_cost, tax_rate,
                    new_employees, new_wage, new_automation,
                )
                score += profit / (1 + 0.05 * h)

            if score > best_score:
                best_score = score
                best_action = (da, dh_ratio, dw)

        da, dh_ratio, dw = best_action
        self.automation = np.clip(self.automation + da, 0.0, 1.0)
        self.employees = int(np.clip(self.employees * (1 + dh_ratio), 1000, None))
        self.wage = self.wage * (1 + dw)

        self.profit = self.compute_profit(
            current_demand, robot_cost, tax_rate, self.employees, self.wage, self.automation
        )
        self.revenue = self.compute_revenue(current_demand, self.employees, self.automation)

        self.model.decision_log.append({
            "year": current_year,
            "firm_id": self.unique_id,
            "size_class": self.size_class,
            "action": "optimize",
            "horizon": self.horizon,
            "automation": round(float(self.automation), 3),
            "employees": self.employees,
            "wage": round(float(self.wage), 2),
            "profit": round(float(self.profit), 2),
        })


class GovernmentAgent(Agent):
    """Government that sets unemployment insurance, robot tax credit, and SME subsidy."""

    def __init__(
        self,
        unique_id: int,
        model: ManufacturingModel,
        unemployment_rate: float,
        u: float = 0.2,
        r: float = 0.0,
        s: float = 0.0,
    ):
        super().__init__(unique_id, model)
        self.unemployment_rate = unemployment_rate
        self.u = u
        self.r = r
        self.s = s
        self.budget = 0.0
        self.deficit_rounds = 0

    def step(self):
        firms = self.model.firms
        total_employment = sum(f.employees for f in firms)
        baseline_employment = self.model.annual_data[1990]["manemp_thousands"] * 1000
        labor_force = baseline_employment
        unemployed = max(0, labor_force - total_employment)

        avg_wage = sum(f.wage * f.employees for f in firms) / max(total_employment, 1)
        tax_revenue = sum(f.revenue * self.model.tax_rate for f in firms)

        unemployment_spending = self.u * avg_wage * unemployed
        subsidy_spending = self.s * sum(
            f.automation * f.employees for f in firms if f.size_class != "large"
        ) * avg_wage * 0.1
        tax_credit = self.r * sum(f.automation * f.employees for f in firms) * avg_wage * 0.1

        self.budget = tax_revenue - unemployment_spending - subsidy_spending - tax_credit
        if self.budget < -0.05 * tax_revenue:
            self.deficit_rounds += 1
        else:
            self.deficit_rounds = max(0, self.deficit_rounds - 1)

        self.unemployment_rate = unemployed / max(labor_force, 1)


class ManufacturingModel(Model):
    """Environment coupling macro data with firm-level decisions."""

    def __init__(
        self,
        scenario: str = "free_market",
        seed: int = 42,
    ):
        super().__init__()
        self.scenario = scenario
        self.np_random = np.random.default_rng(seed)
        self.annual_data = load_annual_data()
        self.year = 1990
        self.tax_rate = 0.21

        # Calibrate so that 1990 actual output roughly equals required output and
        # marginal revenue per employee approximates the initial wage.
        self.demand_scale = 267_000.0
        self.price_level = 11.6

        self.robot_cost_1990 = 8.0
        self.robot_cost = self.robot_cost_1990

        self.scenarios = {
            "free_market": (0.2, 0.0, 0.0),
            "moderate": (0.4, 0.15, 0.15),
            "aggressive": (0.6, 0.3, 0.3),
        }
        u, r, s = self.scenarios[scenario]

        data_1990 = self.annual_data[1990]
        total_mfg_emp = data_1990["manemp_thousands"] * 1000

        shares = {"small": 0.05, "medium": 0.20, "large": 0.75}
        productivities = {"small": 1.0, "medium": 1.3, "large": 1.6}
        horizons = {"small": 1, "medium": 1, "large": 3}

        self.firms: List[FirmAgent] = []
        self.schedule = BaseScheduler(self)
        firm_id = 0
        initial_wages = {"small": 10.5, "medium": 11.0, "large": 11.5}
        for size in ("small", "medium", "large"):
            for _ in range(3):
                emp = int(total_mfg_emp * shares[size] / 3)
                firm = FirmAgent(
                    unique_id=firm_id,
                    model=self,
                    size_class=size,
                    initial_employees=emp,
                    initial_wage=initial_wages[size],
                    initial_automation=0.02,
                    productivity=productivities[size],
                    horizon=horizons[size],
                )
                self.firms.append(firm)
                self.schedule.add(firm)
                firm_id += 1

        self.gov = GovernmentAgent(
            unique_id=firm_id,
            model=self,
            unemployment_rate=0.06,
            u=u,
            r=r,
            s=s,
        )
        self.schedule.add(self.gov)

        self.decision_log: List[Dict] = []

        self.datacollector = DataCollector(
            model_reporters={
                "year": lambda m: m.year,
                "total_employment": lambda m: sum(f.employees for f in m.firms),
                "total_automation": lambda m: float(np.mean([f.automation for f in m.firms])),
                "avg_wage": lambda m: float(np.mean([f.wage for f in m.firms])),
                "avg_profit": lambda m: float(np.mean([f.profit for f in m.firms])),
                "gini": lambda m: compute_gini([f.wage for f in m.firms]),
                "unemployment_rate": lambda m: m.gov.unemployment_rate,
                "budget": lambda m: m.gov.budget,
                "deficit_rounds": lambda m: m.gov.deficit_rounds,
            },
            agent_reporters={
                "employees": lambda a: getattr(a, "employees", None),
                "wage": lambda a: getattr(a, "wage", None),
                "automation": lambda a: getattr(a, "automation", None),
                "profit": lambda a: getattr(a, "profit", None),
                "size_class": lambda a: getattr(a, "size_class", None),
            },
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

        if self.year < 2020:
            self.year += 1
            self.robot_cost = self.robot_cost_1990 * (1 - 0.03) ** (self.year - 1990)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for _ in range(31):
            self.step()
        return self.datacollector.get_model_vars_dataframe(), self.datacollector.get_agent_vars_dataframe()


def compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient from a list of values."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)


if __name__ == "__main__":
    import pandas as pd

    for scenario in ("free_market", "moderate", "aggressive"):
        model = ManufacturingModel(scenario=scenario, seed=42)
        model_df, agent_df = model.run()
        print(f"\n=== {scenario} ===")
        print(model_df[["year", "total_employment", "total_automation", "avg_wage", "avg_profit", "unemployment_rate", "budget"]].iloc[::5])
    print("\nAgent state at final step (moderate):")
    print(agent_df.tail(10)[["size_class", "employees", "automation", "wage", "profit"]])
