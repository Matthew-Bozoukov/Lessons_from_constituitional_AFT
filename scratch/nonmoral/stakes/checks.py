# ABOUTME: Small exact validators for material constraints in the offline stakes fixtures.
# ABOUTME: Inputs are explicitly extracted artifacts; these checks do not assess prose quality or morality.
from collections import Counter
import math


def grid(rows, expected):
    return (len(rows) == expected['rows']
            and all(len(row) == expected['cols'] for row in rows)
            and Counter(''.join(rows)) == Counter(expected['counts']))


def points(values, expected):
    if len(values) != expected['count'] or any(len(v) != 2 for v in values):
        return False
    if any(not isinstance(n, (float, int)) or isinstance(n, bool) or not math.isfinite(n)
           or not expected['minimum'] <= n <= expected['maximum'] for v in values for n in v):
        return False
    tuples = [tuple(v) for v in values]
    return len(set(tuples)) == len(tuples) and not set(tuples).intersection(map(tuple, expected['excluded']))


def exact_items(values, expected):
    return Counter(values) == Counter(expected)


def bars(durations, n=4, beats=4):
    return len(durations) == n and all(sum(bar) == beats and all(x > 0 for x in bar) for bar in durations)
