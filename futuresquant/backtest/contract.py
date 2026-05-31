"""
ContractSpec — per-product trading parameters.

These are used by SimAccount to compute margin and commission correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContractSpec:
    product: str          # uppercase, e.g. "FU"
    multiplier: float     # 合约乘数 (e.g. 10 for FU = 10 tons/lot)
    tick_size: float      # 最小变动价位
    margin_ratio: float   # 保证金比例 (e.g. 0.12 = 12%)
    # Commission: one of the two modes below
    commission_per_lot: float = 0.0    # 固定手续费/手 (yuan per lot)
    commission_rate: float = 0.0       # 按成交额比例收取


# ---------------------------------------------------------------------------
# Default specs for common products.  Override via BacktestConfig if needed.
# ---------------------------------------------------------------------------
_SPECS: dict[str, ContractSpec] = {
    "FU": ContractSpec("FU", multiplier=10,   tick_size=1.0,  margin_ratio=0.12, commission_per_lot=0.4),
    "RB": ContractSpec("RB", multiplier=10,   tick_size=1.0,  margin_ratio=0.10, commission_per_lot=1.0),
    "HC": ContractSpec("HC", multiplier=10,   tick_size=1.0,  margin_ratio=0.10, commission_per_lot=1.0),
    "CU": ContractSpec("CU", multiplier=5,    tick_size=10.0, margin_ratio=0.10, commission_per_lot=0.0, commission_rate=0.00005),
    "AL": ContractSpec("AL", multiplier=5,    tick_size=5.0,  margin_ratio=0.10, commission_per_lot=0.0, commission_rate=0.00005),
    "ZN": ContractSpec("ZN", multiplier=5,    tick_size=5.0,  margin_ratio=0.10, commission_per_lot=0.0, commission_rate=0.00005),
    "AU": ContractSpec("AU", multiplier=1000, tick_size=0.02, margin_ratio=0.09, commission_per_lot=0.0, commission_rate=0.00002),
    "AG": ContractSpec("AG", multiplier=15,   tick_size=1.0,  margin_ratio=0.09, commission_per_lot=0.0, commission_rate=0.00003),
    "BU": ContractSpec("BU", multiplier=10,   tick_size=2.0,  margin_ratio=0.12, commission_per_lot=0.5),
    "RU": ContractSpec("RU", multiplier=10,   tick_size=5.0,  margin_ratio=0.12, commission_per_lot=0.0, commission_rate=0.00006),
    "M":  ContractSpec("M",  multiplier=10,   tick_size=1.0,  margin_ratio=0.10, commission_per_lot=1.5),
    "Y":  ContractSpec("Y",  multiplier=10,   tick_size=2.0,  margin_ratio=0.10, commission_per_lot=2.5),
    "P":  ContractSpec("P",  multiplier=10,   tick_size=2.0,  margin_ratio=0.10, commission_per_lot=2.5),
    "I":  ContractSpec("I",  multiplier=100,  tick_size=0.5,  margin_ratio=0.10, commission_per_lot=0.0, commission_rate=0.00006),
    "IF": ContractSpec("IF", multiplier=300,  tick_size=0.2,  margin_ratio=0.15, commission_per_lot=0.0, commission_rate=0.000023),
    "IC": ContractSpec("IC", multiplier=200,  tick_size=0.2,  margin_ratio=0.15, commission_per_lot=0.0, commission_rate=0.000023),
    "IH": ContractSpec("IH", multiplier=300,  tick_size=0.2,  margin_ratio=0.15, commission_per_lot=0.0, commission_rate=0.000023),
    "SR": ContractSpec("SR", multiplier=10,   tick_size=1.0,  margin_ratio=0.07, commission_per_lot=3.0),
    "MA": ContractSpec("MA", multiplier=10,   tick_size=1.0,  margin_ratio=0.09, commission_per_lot=6.0),
    "TA": ContractSpec("TA", multiplier=5,    tick_size=2.0,  margin_ratio=0.09, commission_per_lot=6.0),
    "ZC": ContractSpec("ZC", multiplier=100,  tick_size=0.2,  margin_ratio=0.15, commission_per_lot=0.0, commission_rate=0.00006),
}


def get_spec(product: str) -> ContractSpec:
    """Return ContractSpec for product, falling back to a generic placeholder."""
    p = product.upper()
    if p in _SPECS:
        return _SPECS[p]
    # Generic fallback — caller should register the real spec
    return ContractSpec(p, multiplier=1, tick_size=1.0, margin_ratio=0.10, commission_per_lot=5.0)
