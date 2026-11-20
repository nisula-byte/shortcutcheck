"""Step 1.3: register the downloaded Dada zip.

Records the zip's SHA-256 in DATA.md, extracts it once into ``data/raw/``, and makes the
extracted files read-only. This is the only code that ever writes into ``data/raw``; every
other writer goes through :mod:`shortcutcheck.data.safe_io`, which refuses that folder.
"""

from __future__ import annotations

import hashlib
import os
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path

DATA_MD_HEADER = (
    "# DATA.md\n\n"
    "Raw data provenance (roadmap step 1.3). Dada is cited by DOI 10.21227/0frv-zb46 "
    "and is never re-hosted or committed.\n\n"
    "| Registered (UTC) | File | Bytes | SHA-256 | Extracted to |\n"
    "| --- | --- | --- | --- | --- |\n"
)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def make_read_only(folder: Path) -> int:
    """Remove write permission from every file and folder under ``folder``."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in filenames:
            path = Path(dirpath) / name
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            count += 1
        for name in dirnames:
            path = Path(dirpath) / name
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
    return count


def register_zip(zip_path: Path, extract_to: Path, data_md: Path) -> str:
    """Hash, extract once, lock, and log. Returns the SHA-256."""
    if extract_to.exists() and any(extract_to.iterdir()):
        raise FileExistsError(
            f"{extract_to} already holds data. Raw data is registered once; "
            "to replace it, delete the folder by hand and record why in DATA.md."
        )
    digest = sha256_of(zip_path)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            target = (extract_to / member).resolve()
            if extract_to.resolve() not in target.parents and target != extract_to.resolve():
                raise ValueError(f"Unsafe path in zip: {member}")
        archive.extractall(extract_to)
    make_read_only(extract_to)
    if not data_md.exists():
        data_md.write_text(DATA_MD_HEADER, encoding="utf-8")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    row = (
        f"| {stamp} | {zip_path.name} | {zip_path.stat().st_size} | `{digest}` | "
        f"{extract_to.as_posix()} |\n"
    )
    with data_md.open("a", encoding="utf-8") as handle:
        handle.write(row)
    return digest
