# src/trade_models.py
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime, timezone

Side = Literal["long", "short"]


class Order(BaseModel):
    id: str
    symbol: str
    side: Side
    type: Literal["market", "limit"]
    price: Optional[float] = None
    qty: float
    status: Literal["new", "filled", "canceled"] = "new"
    created_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class Config:
        extra = "ignore"  # ignore unknown fields for forward/backward compat


class Position(BaseModel):
    symbol: str
    side: Side
    qty: float
    entry_price: float
    stop_price: Optional[float] = None
    targets: List[float] = Field(default_factory=list)
    filled_targets: List[float] = Field(default_factory=list)
    opened_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_update_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class Config:
        extra = "ignore"


class AccountState(BaseModel):
    # Paper account cash used for sizing & PnL baseline.
    equity_usd: float = 10_000.0

    # ✅ New: persisted baseline for PnL tracking (set on first load if None)
    starting_equity_usd: Optional[float] = None

    open_positions: List[Position] = Field(default_factory=list)
    open_orders: List[Order] = Field(default_factory=list)
    cooldown_until: Dict[str, str] = Field(default_factory=dict)  # symbol -> ISO time
    last_run_iso: Optional[str] = None

    class Config:
        extra = "ignore"