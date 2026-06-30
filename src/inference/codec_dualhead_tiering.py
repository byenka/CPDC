from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd


ConfidenceStrategy = Literal["min_teacher", "avg"]
UncertaintyStrategy = Literal["max_teacher", "avg"]


@dataclass(frozen=True)
class DualHeadTieringConfig:
    q_min_A: float = 0.80
    q_min_B: float = 0.50

    agreement_A_min: float = 0.80
    agreement_B_min: float = 0.60

    c_min_A: float = 0.85
    c_min_B: float = 0.70

    entropy_A_max: float = 0.60
    entropy_B_max: float = 0.95

    margin_A_min: float = 0.35
    margin_B_min: float = 0.10

    confidence_strategy: ConfidenceStrategy = "min_teacher"
    uncertainty_strategy: UncertaintyStrategy = "max_teacher"


def _get_cfg(base_cfg: Dict[str, Any]) -> DualHeadTieringConfig:
    cfg = base_cfg.get("codec_dualhead", {}) or {}
    tier = cfg.get("tiering", {}) or {}

    def f(key: str, default: float) -> float:
        try:
            v = float(tier.get(key, default))
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return float(default)

    confidence_strategy = str(tier.get("confidence_strategy", "min_teacher"))
    if confidence_strategy not in ("min_teacher", "avg"):
        confidence_strategy = "min_teacher"

    uncertainty_strategy = str(tier.get("uncertainty_strategy", "max_teacher"))
    if uncertainty_strategy not in ("max_teacher", "avg"):
        uncertainty_strategy = "max_teacher"

    return DualHeadTieringConfig(
        q_min_A=f("q_min_A", DualHeadTieringConfig.q_min_A),
        q_min_B=f("q_min_B", DualHeadTieringConfig.q_min_B),
        agreement_A_min=f("agreement_A_min", DualHeadTieringConfig.agreement_A_min),
        agreement_B_min=f("agreement_B_min", DualHeadTieringConfig.agreement_B_min),
        c_min_A=f("c_min_A", DualHeadTieringConfig.c_min_A),
        c_min_B=f("c_min_B", DualHeadTieringConfig.c_min_B),
        entropy_A_max=f("entropy_A_max", DualHeadTieringConfig.entropy_A_max),
        entropy_B_max=f("entropy_B_max", DualHeadTieringConfig.entropy_B_max),
        margin_A_min=f("margin_A_min", DualHeadTieringConfig.margin_A_min),
        margin_B_min=f("margin_B_min", DualHeadTieringConfig.margin_B_min),
        confidence_strategy=confidence_strategy,  # type: ignore[arg-type]
        uncertainty_strategy=uncertainty_strategy,  # type: ignore[arg-type]
    )


def assign_tiers_dualhead(df: pd.DataFrame, base_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Assign A/B/C tiers using quality + agreement + confidence + uncertainty.

    Expected columns (recommended):
            - Q_score_continuous (optional; if missing/NaN -> treated as 0.0, conservative)
            - agreement (optional; if missing/NaN -> treated as 0.0, conservative)
            - C_score_D, C_score_R, C_score (avg)
            - entropy_D, entropy_R, entropy (avg)
            - margin_D, margin_R, margin (avg)

    Output:
      - adds/overwrites column `tier`
    """

    cfg = _get_cfg(base_cfg)

    out = df.copy()

    def col(name: str, default: float) -> np.ndarray:
        if name not in out.columns:
            return np.full((len(out),), default, dtype=np.float64)
        v = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=np.float64)
        v = np.where(np.isfinite(v), v, default)
        return v

    # Default missing values to strict (0.0) so data without signals is not promoted to A/B.
    q = col("Q_score_continuous", 0.0)
    agreement = col("agreement", 0.0)

    c_D = col("C_score_D", np.nan)
    c_R = col("C_score_R", np.nan)
    c_avg = col("C_score", np.nan) if "C_score" in out.columns else col("C_score_avg", np.nan)

    e_D = col("entropy_D", np.nan)
    e_R = col("entropy_R", np.nan)
    e_avg = col("entropy", np.nan) if "entropy" in out.columns else col("entropy_avg", np.nan)

    m_D = col("margin_D", np.nan)
    m_R = col("margin_R", np.nan)
    m_avg = col("margin", np.nan) if "margin" in out.columns else col("margin_avg", np.nan)

    # Confidence & uncertainty aggregations
    if cfg.confidence_strategy == "min_teacher":
        c_for = np.nanmin(np.stack([c_D, c_R], axis=0), axis=0)
    else:
        c_for = c_avg

    if cfg.uncertainty_strategy == "max_teacher":
        e_for = np.nanmax(np.stack([e_D, e_R], axis=0), axis=0)
        m_for = np.nanmin(np.stack([m_D, m_R], axis=0), axis=0)
    else:
        e_for = e_avg
        m_for = m_avg

    # Tier rules
    a_ok = (
        (q >= cfg.q_min_A)
        & (agreement >= cfg.agreement_A_min)
        & (c_for >= cfg.c_min_A)
        & (e_for <= cfg.entropy_A_max)
        & (m_for >= cfg.margin_A_min)
    )

    b_ok = (
        (q >= cfg.q_min_B)
        & (agreement >= cfg.agreement_B_min)
        & (c_for >= cfg.c_min_B)
        & (e_for <= cfg.entropy_B_max)
        & (m_for >= cfg.margin_B_min)
    )

    tier = np.full((len(out),), "C", dtype=object)
    tier = np.where(b_ok, "B", tier)
    tier = np.where(a_ok, "A", tier)

    out["tier"] = tier
    return out
