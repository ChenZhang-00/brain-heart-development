"""Repository-relative path helpers shared by every analysis stage."""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def project_root(value=None, env=None):
    """Use an explicit root, an environment override, or this repository."""
    value = value or (os.environ.get(env) if env else None)
    return Path(value).expanduser().resolve() if value else REPOSITORY


def resolve_path(value, root):
    """Resolve a user-supplied path from the repository root, not the shell cwd."""
    path = Path(str(value).strip().strip("\"'")).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def relative_path(path, root=REPOSITORY):
    """Show a repository path relative to its root; keep external paths absolute."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
