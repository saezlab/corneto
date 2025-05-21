import logging


def enable_logging(level="info", stream=None):
    """Enable logging output for the mypackage package.

    Args:
        level (str or int, optional): Logging level (e.g., "info", "debug", "warning", logging.INFO, etc.).
            String (case-insensitive) or int accepted. Defaults to "info".
        stream (file-like, optional): Stream for logging output. Defaults to sys.stderr.

    Example:
        import corneto as cn
        cn.enable_logging("debug")
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level_name = level.strip().upper()
        level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(__package__ or "corneto")  # fallback for script use

    # Avoid adding multiple handlers if already present
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def disable_logging():
    """Disable all logging output from the mypackage package.
    Removes all handlers and sets level to WARNING.
    """
    logger = logging.getLogger(__package__ or "mypackage")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)


# Alias for discoverability
set_verbosity = enable_logging
