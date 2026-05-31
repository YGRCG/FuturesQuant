"""
Execution layer unit tests — RiskManager and Settings.
No tqsdk credentials or live connection required.
"""

import pytest
from futuresquant.execution.risk import RiskManager
from futuresquant.config.settings import Settings


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

def test_settings_defaults():
    s = Settings()
    assert s.tq_mode == "sim"
    assert s.risk_max_position_lots > 0
    assert 0 < s.risk_max_drawdown_pct < 1
    assert s.risk_daily_loss_limit > 0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("TQ_MODE", "live")
    monkeypatch.setenv("RISK_MAX_POSITION_LOTS", "5")
    s = Settings()
    assert s.tq_mode == "live"
    assert s.risk_max_position_lots == 5


# ------------------------------------------------------------------
# RiskManager — open checks
# ------------------------------------------------------------------

@pytest.fixture
def risk():
    return RiskManager(
        max_position_lots=5,
        max_drawdown_pct=0.10,
        daily_loss_limit=10_000,
        max_order_notional=500_000,
    )


def test_open_approved_within_limits(risk):
    ok, reason = risk.check_open("SHFE.fu2509", volume=2,
                                  price=3000, multiplier=10, current_position_lots=0)
    assert ok
    assert reason == ""


def test_open_rejected_position_limit(risk):
    ok, reason = risk.check_open("SHFE.fu2509", volume=4,
                                  price=3000, multiplier=10, current_position_lots=3)
    assert not ok
    assert "Position limit" in reason


def test_open_rejected_notional_limit(risk):
    # 2 lots × 50000 price × 10 multiplier = 1_000_000 > 500_000
    ok, reason = risk.check_open("SHFE.fu2509", volume=2,
                                  price=50_000, multiplier=10, current_position_lots=0)
    assert not ok
    assert "Notional" in reason


def test_close_approved(risk):
    ok, reason = risk.check_close(volume=2, current_position_lots=3)
    assert ok


def test_close_rejected_insufficient_position(risk):
    ok, reason = risk.check_close(volume=5, current_position_lots=2)
    assert not ok
    assert "Insufficient" in reason


# ------------------------------------------------------------------
# RiskManager — drawdown circuit breaker
# ------------------------------------------------------------------

def test_drawdown_circuit_breaker(risk):
    risk.reset_daily(initial_capital=1_000_000)
    risk.update(equity=1_000_000, initial_capital=1_000_000)  # peak set
    risk.update(equity=880_000, initial_capital=1_000_000)    # -12% > 10% threshold

    assert risk.is_halted
    assert "drawdown" in risk.halt_reason.lower()

    ok, reason = risk.check_open("SHFE.fu2509", volume=1,
                                  price=3000, multiplier=10, current_position_lots=0)
    assert not ok
    assert "halted" in reason.lower()


def test_daily_loss_circuit_breaker(risk):
    risk.reset_daily(initial_capital=1_000_000)
    # Simulate a -15,000 yuan loss (> daily_loss_limit=10,000)
    risk.update(equity=985_000, initial_capital=1_000_000)

    assert risk.is_halted
    assert "daily loss" in risk.halt_reason.lower()


def test_reset_clears_halt(risk):
    risk.reset_daily(initial_capital=1_000_000)
    risk.update(equity=800_000, initial_capital=1_000_000)  # trip breaker
    assert risk.is_halted

    risk.reset_daily(initial_capital=1_000_000)             # new day
    assert not risk.is_halted


# ------------------------------------------------------------------
# RiskManager — peak equity tracking
# ------------------------------------------------------------------

def test_peak_equity_updates_correctly(risk):
    risk.reset_daily(initial_capital=1_000_000)
    risk.update(equity=1_050_000, initial_capital=1_000_000)
    risk.update(equity=1_030_000, initial_capital=1_000_000)
    # 1_050_000 is peak; drawdown = (1_050_000 - 1_030_000) / 1_050_000 ≈ 1.9% < 10%
    assert not risk.is_halted


def test_no_false_halt_on_recovery(risk):
    risk.reset_daily(initial_capital=1_000_000)
    risk.update(equity=1_000_000, initial_capital=1_000_000)
    risk.update(equity=1_100_000, initial_capital=1_000_000)  # new peak
    risk.update(equity=1_050_000, initial_capital=1_000_000)  # small pullback ~4.5%
    assert not risk.is_halted
