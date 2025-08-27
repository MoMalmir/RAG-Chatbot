import fnmatch
import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def device_info() -> Tuple[str, bool]:
    return DEVICE, DEVICE == "cuda"

def stable_id(path: Path) -> str:
    """Short, stable id from absolute path (good for grouping citations)."""
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]

def iter_input_files(
    paths: Sequence[str],
    recursive: bool = True,
    include: str | None = "*.pdf,*.docx,*.md,*.txt,*.html,*.htm",
    exclude: str | None = None,
) -> List[Path]:
    """
    Expand files/directories into a unique, sorted list of files filtered by include/exclude globs.
    """
    include_pats = [p.strip() for p in (include.split(",") if include else ["*"])]
    exclude_pats = [p.strip() for p in (exclude.split(",") if exclude else [])]
    out: list[Path] = []

    for p in paths:
        root = Path(p)
        it = (root.rglob("*") if (root.is_dir() and recursive) else
              (root.glob("*") if root.is_dir() else [root]))
        for f in it:
            if not f.is_file():
                continue
            name = f.name
            if not any(fnmatch.fnmatch(name, pat) for pat in include_pats):
                continue
            if any(fnmatch.fnmatch(name, pat) for pat in exclude_pats):
                continue
            out.append(f)

    # unique + sorted by path string
    uniq = sorted({str(p): p for p in out}.values(), key=lambda x: str(x))
    return uniq
