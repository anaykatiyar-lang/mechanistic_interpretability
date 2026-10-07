"""Shared prompt configurations for the committed reproduction scripts."""

TARGETS = (4, 6, 8, 10, 12, 14, 16)
POSITIVE_TARGETS = (8, 10, 12, 16)
NULL_TARGETS = (6, 14)


def ordered_splits(target: int) -> list[tuple[int, int]]:
    """Return every valid ordered, unequal single-digit split of ``target``."""
    return [
        (a, target - a)
        for a in range(1, 10)
        if 1 <= target - a <= 9 and a != target - a
    ]


DOUBLES_CONFIG = {
    target: {
        "double": (target // 2, target // 2),
        "controls": ordered_splits(target),
    }
    for target in TARGETS
}

assert tuple(len(DOUBLES_CONFIG[t]["controls"]) for t in POSITIVE_TARGETS) == (6, 8, 6, 2)
assert tuple(len(DOUBLES_CONFIG[t]["controls"]) for t in NULL_TARGETS) == (4, 4)

# The research log records the multiplication and subtraction glyphs as × and −.
# ASCII alternatives can be run as a separate tokenizer-sensitivity analysis.
OPERATOR_VARIANTS = {
    "reported_unicode": {
        "plus": "+",
        "times": "×",
        "minus": "−",
        "and": "and",
        "then": "then",
    },
    "ascii_sensitivity": {
        "plus": "+",
        "times": "*",
        "minus": "-",
        "and": "and",
        "then": "then",
    },
}
