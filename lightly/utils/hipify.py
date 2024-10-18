import copy
import warnings
from typing import Optional, Type, Union


class bcolors:
    """ANSI escape sequences for colored terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_as_warning(message: str, warning_class: Type[Warning] = UserWarning) -> None:
    """Prints a warning message with custom formatting.

    Temporarily overrides the default warning format to apply custom styling, then
    restores the original formatting after the warning is printed.

    Args:
        message:
            The warning message to print.
        warning_class:
            The type of warning to raise.

    """
    old_format = copy.copy(warnings.formatwarning)
    warnings.formatwarning = _custom_formatwarning
    warnings.warn(message, warning_class)
    warnings.formatwarning = old_format


def _custom_formatwarning(
    message: Union[str, Warning],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:
    """Custom format for warning messages.

    Only the warning message is printed, with additional styling applied.

    Args:
        message:
            The warning message or warning object.
        category:
            The warning class.
        filename:
            The file where the warning originated.
        lineno:
            The line number where the warning occurred.
        line:
            The line of code that triggered the warning (if available).

    Returns:
        str: The formatted warning message.

    """
    return f"{bcolors.WARNING}{message}{bcolors.WARNING}\n"
