from pathlib import Path


def read_local_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Input path is not a file: {path}")
    return file_path.read_text(encoding="utf-8")


def write_local_file(path: str, content: str) -> None:
    file_path = Path(path)
    if not str(path).strip():
        raise ValueError("Output path must not be empty")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists() and file_path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {path}")
    file_path.write_text(content, encoding="utf-8")
