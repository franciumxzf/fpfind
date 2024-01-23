import logging
import sys

class LoggingOverrideFormatter(logging.Formatter):
    """Supports injection of overrides during logging.

    The following attributes are injectable: '_funcname', '_filename',
    '_lineno', 'details'. The first three are for higher-level stack trace
    information, adapted from [1].

    Note that multiline logging is generally discouraged. For legacy and
    convenience reasons, multiline logging is enabled by passing
    'human_readable=True' to the Formatter, then furnishing each logger call
    with the 'details' argument, either as a list or a dict. This relies on
    the presence of the '| ' delimiter to separate debugging information from
    text. This delimiter can be redefined.

    Examples:

        # Usual logging setup
        >>> logger = logging.getLogger(__name__)
        >>> handler = logging.StreamHandler(stream=sys.stderr)
        >>> handler.setFormatter(LoggingOverrideFormatter())
        >>> logger.addHandler(handler)

        # Injection of stack trace information
        >>> caller = inspect.getframeinfo(inspect.stack()[1][0])
        >>> extras = {
        ...     "_funcname": f"[{f.__name__}]",
        ...     "_filename": os.path.basename(caller.filename),
        ...     "_lineno": caller.lineno,
        ... }
        >>> logger.warning("callme", stacklevel=2, extra=extras)

        # Append details (machine-parseable format)
        >>> handler.setFormatter(
        ...     LoggingOverrideFormatter(
        ...         fmt="{asctime}\t{levelname:<7s}\t| {message}", style="{",
        ...     )
        ... )
        >>> logger.warning(
        ...     "call!", extras={"details": {
        ...         "value1": 2,
        ...     }
        ... )
        2024-01-23 07:59:24,743 WARNING | call! {"value1": 2}

        # Append details (human-readable format)
        >>> handler.setFormatter(
        ...     LoggingOverrideFormatter(
        ...         fmt="{asctime}\t{levelname:<7s}\t| {message}", style="{",
        ...         human_readable=True, delimiter="| ",
        ...     )
        ... )
        >>> logger.warning(
        ...     "call!", extras={"details": [
        ...         "value1 is 2ns",
        ...         "value2 has been read",
        ...     ]
        ... )
        2024-01-23 07:59:24,743 WARNING | call!
                                        |   value1 is 2ns
                                        |   value2 has been read

    References:
        [1]: <https://stackoverflow.com/a/71228329>
    """
    def __init__(self, *args, human_readable=False, delimiter="| ", **kwargs):
        # For multiline logging
        self.human_readable = human_readable
        self.delim = delimiter
        super().__init__(*args, **kwargs)

    def format(self, record):
        # Override attributes with debugging-relevant information
        if hasattr(record, "_funcname"):
            record.funcName = record._funcname
        if hasattr(record, "_filename"):
            record.filename = record._filename
        if hasattr(record, "_lineno"):
            record.lineno= record._lineno
        message = super().format(record)

        # Append additional debugging info
        details = getattr(record, "details", None)
        if details is None:
            pass  # ignore
        elif not self.human_readable:
            message = f"{message}\t{details}"  # append details to the back
        else:
            # Enable multiline logging for humans to read
            if isinstance(details, dict):  # parse into array
                details = [f"{k}: {v}" for k, v in details.items()]
            if isinstance(details, (list, tuple)) and len(details) > 0:
                pre, _, text = message.partition(self.delim)
                pad = " " * (len(text) - len(text.lstrip(" ")))
                # preserve tabs, alternatively use 'str.expandtabs'
                _pre = "".join([c if c.isspace() else " " for c in pre])
                text = text.lstrip(" ")

                # Collect all messages before concatenating
                messages = [message]
                if text == "":  # replace first line if empty
                    text, details = details[0], details[1:]
                    messages = [f"{pre}{self.delim}{pad}  {text}"]
                messages.extend(
                    [f"{_pre}{self.delim}{pad}  {text}" for text in details]
                )
                message = "\n".join(messages)

        return message

def get_logger(name, level=None, human_readable=False):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            LoggingOverrideFormatter(
                fmt="{asctime}\t{levelname:<7s}\t{funcName}:{lineno}\t| {message}",
                datefmt="%Y%m%d_%H%M%S",
                style="{",
                human_readable=human_readable,
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
