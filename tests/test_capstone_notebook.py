import json
from pathlib import Path
import pytest
import nbformat

CAPSTONE_NOTEBOOK_PATH = Path("work/notebooks/capstone.ipynb")

def test_capstone_notebook_exists():
    assert CAPSTONE_NOTEBOOK_PATH.exists(), f"Notebook not found at {CAPSTONE_NOTEBOOK_PATH}"

def test_capstone_has_all_required_sections():
    """Verify all canonical sections from the assignment and skeleton are present."""
    with open(CAPSTONE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    markdown_sources = [cell.source for cell in nb.cells if cell.cell_type == "markdown"]
    combined_markdown = "\n".join(markdown_sources)
    
    required_sections = [
        "## 1. Question",
        "## 2. Data",
        "## 3. Methodology",
        "## 4. Results (vs baseline)",
        "## 5. Limitations",
        "## 6. Ranked recommendations",
        "## 7. Artifacts the paper embeds",
        "## Self-check"
    ]
    
    for section in required_sections:
        assert section in combined_markdown, f"Missing required section: {section}"

def test_capstone_cells_are_implemented():
    """Verify that code cells are not empty and have been filled with logic."""
    with open(CAPSTONE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 7, f"Expected at least 7 code cells, found {len(code_cells)}"
    
    empty_cells = [i for i, cell in enumerate(code_cells) if not cell.source.strip()]
    assert not empty_cells, f"Found empty code cells at indices: {empty_cells}"

def test_capstone_is_fully_executed():
    """Verify that every code cell has been executed without errors."""
    with open(CAPSTONE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    for i, cell in enumerate(code_cells):
        assert cell.execution_count is not None, f"Cell {i} has not been executed (execution_count is null)"
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", f"Cell {i} failed with error: {output.get('ename')}: {output.get('evalue')}"


def _closing_markdown():
    with open(CAPSTONE_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "markdown"]


def test_ml12_demo_outline_and_cuts_close_the_notebook():
    """ML-12: demo outline is the closing section, with the two shareable cuts right under it."""
    md = _closing_markdown()
    assert md[-2].startswith("## 8. Demo outline"), "Demo outline must be the second-to-last markdown cell"
    assert md[-1].startswith("## 9. Shareable cuts"), "Shareable cuts must sit right under the outline"
    for beat in ["Question", "Method", "One chart", "One honest result", "One recommendation"]:
        assert f"**{beat}**" in md[-2], f"Demo outline missing beat: {beat}"
    assert "### Social post" in md[-1] and "### Employer summary" in md[-1]


def test_ml12_cuts_numbers_match_receipts():
    """Numbers quoted in the shareable cuts must match the 20-split receipt."""
    rob = json.loads(Path("work/outputs/robustness_summary.json").read_text(encoding="utf-8"))
    g = rob["grouped"]
    pct = lambda x: f"{int(x * 100 + 0.5)}%"  # round half up, as written in the paper (0.885 -> 89%)
    model_p10 = pct(g['HistGradientBoosting']['P@10']['mean'])
    rule_p10 = pct(g['Week 4 rule']['P@10']['mean'])
    ratio = f"{g['HistGradientBoosting']['P@10']['mean'] / g['Week 4 rule']['P@10']['mean']:.1f}×"
    n = rob["n_grouped_splits"]
    cuts = _closing_markdown()[-1]
    strict_p10 = f"{g['HistGradientBoosting (strict features)']['P@10']['mean']:.0%}"
    for needle in [model_p10, rule_p10, ratio, strict_p10, f"all {n}", f"{n} different client splits"]:
        assert needle in cuts, f"Shareable cuts missing receipt-backed value: {needle}"
    url = Path("submission/paper_url.txt").read_text(encoding="utf-8").strip()
    assert url in cuts, "Social post must link the deployed paper"
