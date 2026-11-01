"""Download a YOLO-format dataset from Kaggle.

Kaggle credentials are required.  Create a token at
https://www.kaggle.com/settings -> "Create New Token", or (as on this machine)
use an existing ``~/.kaggle/access_token``.  Either place ``kaggle.json`` in
``~/.kaggle/`` or export ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.

The Kaggle CLI is not installed by default because it is only needed when you
want to fine-tune on a custom dataset::

    .venv/bin/python -m pip install kaggle
    .venv/bin/python download_dataset.py owner/dataset-slug --dest datasets

Many Kaggle uploads keep a ``path:`` value from the author's machine.  If the
dataset was unzipped somewhere else, this script writes a
``<name>.resolved.yaml`` next to the original with the corrected absolute path.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def kaggle_command() -> list[str]:
    """Locate the Kaggle CLI, including one installed inside the active venv.

    ``shutil.which`` only searches ``PATH``.  When the script is started as
    ``.venv/bin/python download_dataset.py``, the venv's ``bin`` directory is
    not necessarily on ``PATH``, so look next to the running interpreter too.
    """
    found = shutil.which("kaggle")
    if found:
        return [found]

    candidate = Path(sys.executable).parent / "kaggle"
    if candidate.exists():
        return [str(candidate)]

    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "The Kaggle CLI is not installed. Install it with:\n"
            "  .venv/bin/python -m pip install kaggle\n"
            "then add ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY."
        ) from None
    return [sys.executable, "-m", "kaggle"]


def download_kaggle_dataset(slug: str, destination: str = "datasets", force: bool = False) -> Path:
    """Download and unzip a Kaggle dataset, returning the destination folder."""
    if not _SLUG.fullmatch(slug):
        raise ValueError("Kaggle dataset must look like 'owner/dataset-slug'")
    cli = kaggle_command()
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    command = [*cli, "datasets", "download", "-d", slug, "-p", str(target), "--unzip"]
    if force:
        command.append("--force")
    print("Running:", " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)
    return target


def find_data_yaml(root: Path) -> Path | None:
    """Find the first YOLO ``data.yaml`` under a downloaded dataset."""
    for pattern in ("data.yaml", "*.yaml", "*.yml"):
        matches = sorted(root.rglob(pattern))
        for match in matches:
            text = match.read_text(encoding="utf-8", errors="ignore")
            if "train:" in text and "names:" in text:
                return match
    return None


def _load_yaml(path: Path) -> dict | None:
    try:
        import yaml
    except ImportError:
        return None
    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return spec if isinstance(spec, dict) else None


def _relative_entries(spec: dict) -> list[str]:
    """Return the relative ``train``/``val`` entries of a dataset config."""
    entries: list[str] = []
    for key in ("train", "val"):
        value = spec.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            entries.append(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            first = value[0]
            if not Path(first).is_absolute():
                entries.append(first)
    return entries


def _looks_like_root(candidate: Path, entries: list[str]) -> bool:
    """Return True when ``candidate`` really contains the train/val folders."""
    if not entries:
        return False
    for entry in entries:
        target = candidate / entry
        try:
            if target.is_file():
                return True
            if target.is_dir() and any(target.iterdir()):
                return True
        except OSError:
            continue
    return False


def find_dataset_root(data_yaml: Path) -> Path | None:
    """Locate the folder that the ``train``/``val`` entries are relative to.

    Kaggle archives are frequently uploaded with a ``path:`` value from the
    author's own machine (for example ``../datasets/brain-tumor``).  This walks
    the downloaded tree and returns the folder that actually holds the data.
    """
    spec = _load_yaml(data_yaml)
    if spec is None:
        return None
    entries = _relative_entries(spec)
    if not entries:
        return data_yaml.parent.resolve()

    declared = spec.get("path")
    candidates: list[Path] = []
    if isinstance(declared, str) and declared.strip():
        raw = Path(declared.strip()).expanduser()
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(data_yaml.parent / raw)
            # Ultralytics resolves relative paths against ~/datasets.
            candidates.append(Path.home() / "datasets" / raw)
    candidates.append(data_yaml.parent)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if _looks_like_root(resolved, entries):
            return resolved
        nested = resolved / data_yaml.stem
        if nested.is_dir() and _looks_like_root(nested, entries):
            return nested.resolve()

    # Breadth-first search below the downloaded folder (depth-limited).
    queue: list[tuple[Path, int]] = [(data_yaml.parent, 0)]
    seen: set[Path] = set()
    while queue:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if _looks_like_root(current, entries):
            return current.resolve()
        if depth >= 3:
            continue
        try:
            children = sorted(child for child in current.iterdir() if child.is_dir())
        except OSError:
            continue
        queue.extend((child, depth + 1) for child in children)
    return None


def prepare_data_yaml(data_yaml: str | Path) -> Path:
    """Return a dataset config with ``path:`` pointing at the real folder.

    When the original file is already correct it is returned untouched.  When
    it is not, a sibling ``<name>.resolved.yaml`` is written so the download is
    not modified and the fix is visible.
    """
    path = Path(data_yaml).expanduser()
    if not path.is_file():
        return path
    spec = _load_yaml(path)
    if spec is None:
        return path
    entries = _relative_entries(spec)

    declared = spec.get("path")
    if isinstance(declared, str) and declared.strip():
        raw = Path(declared.strip()).expanduser()
        # A relative ``path`` is resolved by Ultralytics against its global
        # datasets directory (usually ~/datasets), not against this file, so
        # only an absolute path can be trusted as-is.
        if raw.is_absolute() and entries and _looks_like_root(raw, entries):
            return path

    root = find_dataset_root(path)
    if root is None:
        return path
    spec["path"] = str(root)
    suffix = path.suffix or ".yaml"
    output = path.with_name(f"{path.stem}.resolved{suffix}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML ships with ultralytics
        raise RuntimeError("PyYAML is required to rewrite the dataset config") from exc
    output.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a YOLO dataset from Kaggle")
    parser.add_argument("slug", help="dataset slug, for example owner/object-detection")
    parser.add_argument("--dest", default="datasets", help="destination directory")
    parser.add_argument("--force", action="store_true", help="overwrite an existing download")
    args = parser.parse_args()

    try:
        root = download_kaggle_dataset(args.slug, args.dest, args.force)
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc

    data_yaml = find_data_yaml(root)
    print(f"Downloaded to: {root}")
    if data_yaml:
        usable = prepare_data_yaml(data_yaml)
        if usable != data_yaml:
            print(f"Fixed relative 'path:' in the dataset config -> {usable}")
        print(f"Dataset config: {usable}")
        print(f"Train with: .venv/bin/python train.py --data {usable}")
    else:
        print(
            "No data.yaml was found. Point --data at the dataset's YOLO config, "
            "or create one from data/objects.example.yaml."
        )


if __name__ == "__main__":
    main()
