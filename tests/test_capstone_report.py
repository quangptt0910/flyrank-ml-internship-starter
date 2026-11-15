"""The capstone report must cover the rubric sections and quote numbers that match the receipts."""
import json
import re
from pathlib import Path

REPORT = Path("work/capstone_report.md")
OUT = Path("work/outputs")


def _text():
    return REPORT.read_text(encoding="utf-8")


def _load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _shown(x, text, ndigits=2):
    """True if x appears in text at the given precision (either rounding direction at the half)."""
    cands = {f"{x:.{ndigits}f}", f"{round(x + 1e-9, ndigits):.{ndigits}f}", f"{round(x - 1e-9, ndigits):.{ndigits}f}"}
    return any(c in text for c in cands)


def test_report_has_all_rubric_sections():
    text = _text()
    for heading in ["## 1. Problem framing", "## 2. Data safety", "## 3. Baseline", "## 4. Model / analysis",
                    "## 5. Evaluation", "## 6. Interpretation", "## 7. Recommendation", "## 8. Reproducibility"]:
        assert heading in text, heading
    assert "<your lane>" not in text and "- **Author:**\n" not in text


def test_twenty_split_numbers_match_receipt():
    text = _text()
    g = _load("robustness_summary.json")["grouped"]
    for model in ["HistGradientBoosting", "HistGradientBoosting (strict features)", "Week 4 rule"]:
        for metric in ["P@10", "P@50", "ROC-AUC"]:
            assert _shown(g[model][metric]["mean"], text), (model, metric)
    for metric in ["P@50", "ROC-AUC"]:
        assert _shown(g["Logistic Regression"][metric]["mean"], text), ("LR", metric)


def test_split_counts_and_base_rate_match_receipt():
    text = _text()
    r = _load("robustness_summary.json")
    assert f"{r['strict_model_beats_rule_splits']['P@50']} / 20" in text
    assert f"{r['model_beats_rule_splits']['P@50']} / 20" in text
    assert _shown(r["test_base_rate"]["mean"], text) and _shown(r["test_base_rate"]["min"], text)
    assert _shown(r["random_split_auc"]["mean"], text)


def test_error_analysis_numbers_match_receipt():
    text = _text()
    ea = _load("error_analysis.json")
    assert f"{ea['top100_wrong_picks']} wrong picks" in text
    assert f"{ea['share_declining_tiny_prev30']:.1%}" in text
    assert _shown(ea["auc_without_tiny"], text, 3)
    aucs = [c["AUC"] for c in ea["per_client"] if c["AUC"] is not None]
    assert _shown(min(aucs), text) and _shown(max(aucs), text)
    top = max(ea["per_client"], key=lambda c: c["top-20 slots in global queue"])
    assert f"takes {top['top-20 slots in global queue']} of the 20" in text


def test_queue_numbers_match_receipt():
    text = _text()
    assert _shown(_load("action_playbook_summary.json")["held_out_test_precision_at_50"], text)
    assert "16 of the top 20" in text


def test_report_is_public_safe():
    text = _text()
    assert not re.search(r"(?:client|content)_[0-9a-f]{10,}", text), "hashed IDs must not appear"
    for banned in ["predicts Google", "will recover", "causes decline"]:
        assert banned not in text
