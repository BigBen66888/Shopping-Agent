"""Package only source code for upload to Kaggle (no local data or models)."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


root = Path(__file__).resolve().parents[1]
output = root / "offline_embedding" / "kaggle_bundle.zip"
sources = [
    root / "shopping_agent" / "__init__.py",
    root / "shopping_agent" / "models.py",
    root / "shopping_agent" / "dataset.py",
    root / "shopping_agent" / "index_contract.py",
    *sorted((root / "offline_embedding").glob("*.py")),
]
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for path in sources:
        archive.write(path, Path("Shopping_agent") / path.relative_to(root))
print(output)
