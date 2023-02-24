import copy
import warnings
from typing import Type


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return f"{bcolors.WARNING}{msg}{bcolors.WARNING}\n"



def print_as_warning(message: str, warning_class: Type[Warning] = UserWarning):
    old_format = copy.copy(warnings.formatwarning)

    warnings.formatwarning = _custom_formatwarning
    warnings.warn(message, warning_class)

    warnings.formatwarning = old_format