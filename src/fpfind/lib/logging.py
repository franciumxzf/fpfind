import logging
import sys

class LoggingCustomFormatter(logging.Formatter):
    """Supports injection of custom name overrides during logging.

    Copied from <https://stackoverflow.com/a/71228329>.
    """
    def format(self, record):
        if hasattr(record, "_funcname"):
            record.funcName = record._funcname
        if hasattr(record, "_filename"):
            record.filename = record._filename
        if hasattr(record, "_lineno"):
            record.lineno= record._lineno
        return super().format(record)

def get_logger(name, level=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            LoggingCustomFormatter(
                fmt="{asctime}\t{levelname:<7s}\t{funcName}:{lineno}\t| {message}",
                datefmt="%Y%m%d_%H%M%S",
                style="{",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(label2level(level))
    return logger

def verbosity2level(verbosity):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    verbosity = min(verbosity, len(levels)-1)
    return levels[verbosity]

def label2level(label):
    LOG_LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    return LOG_LEVELS.get(label, logging.WARNING)
