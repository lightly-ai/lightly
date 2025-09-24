import random

_CHARACTER_SET = "abcdef0123456789"


def generate_id(length: int = 24) -> str:
    return "".join([random.choice(_CHARACTER_SET) for i in range(length)])
