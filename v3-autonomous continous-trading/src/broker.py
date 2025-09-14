# src/broker.py
from __future__ import annotations
import os, time
from typing import Optional
from .trade_models import Order, Position, AccountState

try:
    import ccxt
except Exception:
    ccxt = None  # let user install later

class Broker:
    def __init__(self, exchange: str = "binance", testnet: bool = True, live: bool = False):
        self.live = live
        self.testnet = testnet
        self.exchange_name = exchange
        self.client = None
        if live and ccxt is not None:
            klass = getattr(ccxt, exchange)
            self.client = klass({
                "apiKey": os.getenv("API_KEY", ""),
                "secret": os.getenv("API_SECRET", ""),
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            if exchange == "binance":
                self.client.set_sandbox_mode(testnet)

    # -------- paper trading helpers --------
    def paper_market_buy(self, state: AccountState, symbol: str, price: float, qty: float) -> Order:
        cost = price * qty
        if state.equity_usd < cost:
            raise ValueError("paper: insufficient equity")
        state.equity_usd -= cost
        order = Order(id=f"paper-{symbol}-{time.time():.0f}", symbol=symbol, side="long",
                      type="market", price=price, qty=qty, status="filled")
        state.open_orders.append(order)
        return order

    def paper_market_sell(self, state: AccountState, symbol: str, price: float, qty: float) -> Order:
        proceeds = price * qty
        state.equity_usd += proceeds
        order = Order(id=f"paper-{symbol}-{time.time():.0f}", symbol=symbol, side="short",
                      type="market", price=price, qty=qty, status="filled")
        state.open_orders.append(order)
        return order

    # -------- live (simplified) --------
    def live_market_buy(self, symbol: str, qty: float) -> Order:
        if self.client is None:
            raise RuntimeError("CCXT not initialized for live trading")
        o = self.client.create_market_buy_order(symbol, qty)
        return Order(id=str(o.get("id")), symbol=symbol, side="long", type="market",
                     price=float(o["average"] or o["price"] or 0), qty=float(o["filled"] or qty),
                     status="filled" if o.get("status") == "closed" else "new")

    def live_market_sell(self, symbol: str, qty: float) -> Order:
        if self.client is None:
            raise RuntimeError("CCXT not initialized for live trading")
        o = self.client.create_market_sell_order(symbol, qty)
        return Order(id=str(o.get("id")), symbol=symbol, side="short", type="market",
                     price=float(o["average"] or o["price"] or 0), qty=float(o["filled"] or qty),
                     status="filled" if o.get("status") == "closed" else "new")

    # unified
    def market_buy(self, state: AccountState, symbol: str, price: float, qty: float) -> Order:
        if self.live:
            return self.live_market_buy(symbol, qty)
        return self.paper_market_buy(state, symbol, price, qty)

    def market_sell(self, state: AccountState, symbol: str, price: float, qty: float) -> Order:
        if self.live:
            return self.live_market_sell(symbol, qty)
        return self.paper_market_sell(state, symbol, price, qty)