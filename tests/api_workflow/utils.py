import random

# [a-f0-9]
_CHARACTER_SET = [str(i) for i in range(10)] + [chr(i + 97) for i in range(6)]


def generate_id(length: int = 24) -> str:
    return "".join([random.choice(_CHARACTER_SET) for i in range(length)])
