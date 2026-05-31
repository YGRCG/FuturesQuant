"""
Global settings loaded from environment / .env file.

Usage
-----
    from futuresquant.config.settings import settings
    print(settings.tq_username)
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # 天勤行情账号
    # ------------------------------------------------------------------
    tq_username: str = Field(default="", description="天勤用户名")
    tq_password: str = Field(default="", description="天勤密码")

    # ------------------------------------------------------------------
    # CTP 期货账户
    # ------------------------------------------------------------------
    tq_broker_id: str = Field(default="9999", description="期货公司代码（9999=快期模拟）")
    tq_account_id: str = Field(default="", description="期货账户号")
    tq_account_password: str = Field(default="", description="期货账户密码")

    # ------------------------------------------------------------------
    # 运行模式
    # ------------------------------------------------------------------
    tq_mode: Literal["sim", "live"] = Field(
        default="sim",
        description="sim=TqSim仿真  live=TqAccount实盘",
    )

    # ------------------------------------------------------------------
    # 风控参数
    # ------------------------------------------------------------------
    risk_max_position_lots: int = Field(default=10, ge=1)
    risk_max_drawdown_pct: float = Field(default=0.05, gt=0, le=1)
    risk_daily_loss_limit: float = Field(default=5_000.0, gt=0)
    risk_max_order_notional: float = Field(default=500_000.0, gt=0)


# 单例
settings = Settings()
