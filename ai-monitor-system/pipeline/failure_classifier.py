def classify_failure(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "input_not_found"
    if isinstance(error, IsADirectoryError):
        return "invalid_path"
    if isinstance(error, PermissionError):
        return "permission_denied"
    return "runtime_error"
