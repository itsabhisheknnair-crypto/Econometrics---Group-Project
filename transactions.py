from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


TX_FILE = Path("transactions.csv")
TX_COLUMNS = ["date", "amount_inr", "rate", "eur_received"]


@dataclass
class Transaction:
    date: date
    amount_inr: float
    rate: float
    eur_received: float


def load_transactions() -> pd.DataFrame:
    """Load transaction history from CSV (if it exists)."""
    if TX_FILE.exists():
        df = pd.read_csv(TX_FILE, parse_dates=["date"])
        # Ensure expected columns exist
        for col in TX_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[TX_COLUMNS]
    return pd.DataFrame(columns=TX_COLUMNS)


def append_transaction(
    tx_date: date,
    amount_inr: float,
    rate: float,
) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Append a new transaction and compute savings vs. last logged rate.

    Savings is measured as additional EUR you receive now compared to
    using the previous logged rate for the same INR amount.
    """
    df = load_transactions()

    eur_now = amount_inr / rate if rate > 0 else 0.0
    savings_eur: Optional[float] = None

    if not df.empty:
        last_rate = float(df.iloc[-1]["rate"])
        eur_at_last = amount_inr / last_rate if last_rate > 0 else 0.0
        savings_eur = eur_now - eur_at_last

    new_row = {
        "date": tx_date,
        "amount_inr": amount_inr,
        "rate": rate,
        "eur_received": eur_now,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(TX_FILE, index=False)

    return df, savings_eur


