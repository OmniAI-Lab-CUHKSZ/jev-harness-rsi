"""Access a public field by its dotted path."""


def path_get(value, path, default=None):
    for part in path.split(".") if path else []:
        try:
            value = value[int(part)] if isinstance(value, (list, tuple)) else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return default
    return value
