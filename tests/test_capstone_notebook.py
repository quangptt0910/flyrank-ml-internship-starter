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
