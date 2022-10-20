from pathlib import Path


def format_path(path: Path, **kwargs: str) -> Path:
    return path.parent / path.name.format(**kwargs)
