"""Validate the LLM judge against human scores."""
import numpy as np
from scipy.stats import pearsonr


def agreement(human_scores, judge_scores):
    h = np.array(human_scores, dtype=float)
    j = np.array(judge_scores, dtype=float)
    corr, _ = pearsonr(h, j)
    mae = float(np.mean(np.abs(h - j)))
    within_1 = float(np.mean(np.abs(h - j) <= 1) * 100)
    return {
        "n": len(h),
        "pearson_r": float(corr),
        "mean_absolute_error": mae,
        "percent_within_1_point": within_1,
    }