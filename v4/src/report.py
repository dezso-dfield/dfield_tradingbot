from .utils import save_json

def write_signals_json(path: str, ranked):
    payload = []
    for r in ranked:
        payload.append({
            "symbol": r.symbol,
            "score": round(r.score, 2),
            "decision": r.decision.model_dump()
        })
    save_json(path, payload)