from .schema import IndicatorPacket, LLMDecision, RankedIdea

# Default guards (tweak via CLI later if you want)
MIN_ADV_USD = 2_000_000     # skip illiquid stuff
MAX_ATR_PCT = 0.25          # skip if ATR% > 25% of price

def heuristic_score(ind: IndicatorPacket, dec: LLMDecision) -> float:
    # Hard guards first
    if ind.adv_usd_20 is not None and ind.adv_usd_20 < MIN_ADV_USD:
        return -1e9  # effectively hides it
    if ind.atr_pct is not None and ind.atr_pct > MAX_ATR_PCT:
        return -1e9

    s = 0.0
    # LLM conviction
    s += 60.0 * max(0.0, min(1.0, dec.conviction))
    # Trend
    if ind.above_sma200: s += 10.0
    s += 10.0 * max(0.0, min(1.0, (ind.sma50_slope + 0.001)))
    # Momentum sweet spot
    if 35 <= ind.rsi <= 65: s += 5.0
    # Participation
    if ind.volume_trend > 1.0: s += 3.0
    # Relative volume bonus (more selective)
    if ind.rvol_20 >= 1.5: s += 5.0
    elif ind.rvol_20 >= 1.2: s += 2.0
    # Strength
    if ind.adx > 20: s += 5.0
    # Penalty for no-action
    if dec.action == "wait": s -= 10.0
    return s

def rank_ideas(pairs):
    ranked = []
    for ind, dec in pairs:
        ranked.append(RankedIdea(symbol=ind.symbol, score=heuristic_score(ind, dec), decision=dec))
    # filter out hidden (guarded)
    ranked = [r for r in ranked if r.score > -1e8]
    return sorted(ranked, key=lambda x: x.score, reverse=True)