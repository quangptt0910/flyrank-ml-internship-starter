from pathlib import Path
import re
import pytest

DOCS_INDEX_PATH = Path("docs/index.html")

def test_docs_index_exists():
    assert DOCS_INDEX_PATH.exists(), f"docs/index.html does not exist at {DOCS_INDEX_PATH}"

def test_docs_index_has_all_canonical_sections():
    """Verify that all 9 canonical research paper sections are present."""
    html_content = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    
    canonical_sections = [
        "Abstract",
        "Introduction",
        "Data",
        "Methodology",
        "Results",
        "Limitations",
        "Ranked Recommendations",
        "Reproducibility",
        "Acknowledgments"
    ]
    for section in canonical_sections:
        pattern = re.compile(rf"<h[1-3][^>]*>.*{section}.*</h[1-3]>", re.IGNORECASE)
        assert pattern.search(html_content), f"Missing canonical section header for: {section}"

def test_docs_index_has_mandatory_data_credit():
    """Verify that data attribution to flyrank.ai is explicitly present and hyperlinked."""
    html_content = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    assert "Built on the FlyRank ML Internship dataset" in html_content
    assert "https://flyrank.ai" in html_content

def test_docs_index_images_are_relative_and_exist():
    """Verify that all embedded images use relative paths and exist on disk."""
    html_content = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    img_srcs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html_content)
    assert len(img_srcs) >= 3, f"Expected at least 3 embedded figures, found {len(img_srcs)}"
    
    for src in img_srcs:
        assert not src.startswith("http"), f"Image src should be relative, found external/absolute URL: {src}"
        assert not src.startswith("/"), f"Image src should be relative (no leading slash for GitHub Pages): {src}"
        img_path = Path("docs") / src
        assert img_path.exists(), f"Embedded image does not exist: {img_path}"

def test_public_safety_no_client_names_or_leakage():
    """Verify that no private client identifiers or forbidden terms appear."""
    html_content = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    # Must not contain unhashed private client references
    assert "client_0" not in html_content.lower()
    assert "predicted google's algorithm" not in html_content.lower()


def _receipt(name):
    import json
    return json.loads((Path("work/outputs") / name).read_text(encoding="utf-8"))


def test_paper_headline_numbers_match_receipts():
    """Every headline number on the page must come from a receipt in work/outputs, not be typed by hand."""
    html = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    rob = _receipt("robustness_summary.json")["grouped"]
    hgb, rule = rob["HistGradientBoosting"], rob["Week 4 rule"]
    expected = [
        f"{hgb['P@10']['mean']:.2f} ± {hgb['P@10']['std']:.2f}",
        f"{hgb['P@50']['mean']:.2f} ± {hgb['P@50']['std']:.2f}",
        f"{rule['P@10']['mean']:.2f} ± {rule['P@10']['std']:.2f}",
        f"{rule['P@50']['mean']:.2f} ± {rule['P@50']['std']:.2f}",
        f"{hgb['ROC-AUC']['mean']:.2f} ± {hgb['ROC-AUC']['std']:.2f}",
    ]
    single = {m["Model"]: m for m in _receipt("model_comparison.json")["models"]}
    for k in (10, 20, 50, 100):
        n = round(single["Week 4 Rule Baseline (on test)"][f"P@{k}"] * k)
        expected.append(f"({n}/{k})")
    for needle in expected:
        assert needle in html, f"Paper is missing receipt-backed value: {needle}"


def test_paper_has_no_known_wrong_or_overclaimed_text():
    html = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    for bad in ["Release: March 2026", "15,335", "0.40 * stale", "Infinite ROI",
                "standard commercial heuristic", "No-Go", "proves"]:
        assert bad not in html, f"Paper still contains: {bad}"


def test_ml12_paper_ties_findings_to_flyrank_case():
    html = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    abstract = html[html.index('class="abstract-box"'):html.index("</div>", html.index('class="abstract-box"'))]
    assert "FlyRank" in abstract, "Abstract must tie the work to FlyRank's content problem"
    assert "The FlyRank case" in html, "Introduction must contain the FlyRank case study"
    assert "What this means for FlyRank" in html, "Recommendations must map back to FlyRank's workflow"
    assert "does <strong>not</strong> compare against them" in html, "Must state production flags were not compared"


def test_paper_strict_feature_numbers_match_receipts():
    """The overlapping-window sensitivity check (ML-05) must be reported with receipt-backed numbers."""
    html = DOCS_INDEX_PATH.read_text(encoding="utf-8")
    s = _receipt("robustness_summary.json")["grouped"]["HistGradientBoosting (strict features)"]
    for m in ["P@10", "P@20", "P@50", "P@100", "ROC-AUC"]:
        needle = f"{s[m]['mean']:.2f} ± {s[m]['std']:.2f}"
        assert needle in html, f"Paper missing strict-feature {m}: {needle}"
    assert "4.4 Part of the gain comes from overlapping windows" in html
    assert "w03_feature_leakage_check.ipynb" in html
