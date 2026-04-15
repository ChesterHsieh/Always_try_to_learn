def classify_failure(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "input_not_found"
    return "runtime_error"
