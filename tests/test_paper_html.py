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
