import logging

from . import articular_surfaces, comak_osim_update, nsm_fitting, osim_utils, utils

__version__ = "0.0.1"

# Export the configure_logging function for easy access
__all__ = [
    "configure_logging",
    "articular_surfaces",
    "comak_osim_update",
    "nsm_fitting",
    "utils",
]


def configure_logging(level="INFO", format_string=None):
    """
    Configure logging for the entire nsosim package.

    Args:
        level (str): Logging level. Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        format_string (str, optional): Custom format string. If None, uses a sensible default.

    Examples:
        # Basic usage - show info and above
        import nsosim
        nsosim.configure_logging('INFO')

        # Show all debug information
        nsosim.configure_logging('DEBUG')

        # Only show warnings and errors
        nsosim.configure_logging('WARNING')

        # Custom format
        nsosim.configure_logging('DEBUG', '%(name)s - %(levelname)s - %(message)s')
    """
    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level.upper() not in level_map:
        raise ValueError(f"Invalid log level: {level}. Must be one of {list(level_map.keys())}")

    log_level = level_map[level.upper()]

    # Default format if none provided
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure the root logger for the nsosim package
    logger = logging.getLogger("nsosim")
    logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    logger.info(f"nsosim logging configured to level: {level.upper()}")
