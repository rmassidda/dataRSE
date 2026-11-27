"""
Utils

Copyright (c) 2024 Riccardo Massidda
Released under the MIT license, see LICENSE.
"""

import itertools
from collections import defaultdict
from typing import Generator, TypeVar

T = TypeVar("T")


def abstract_intersections(blocks: list[list[int]]) -> list[set[int]]:
    """Given the blocks of each abstract variable, it returns the list of the
    exact groups of abstract variables whose blocks all share at least one
    common variable."""
    # Step 1: Build inverted index: element -> {indexes of sets containing it}
    inverted = defaultdict(set)
    for i, s in enumerate(blocks, start=0):
        for element in s:
            inverted[element].add(i)

    # Step 2: Each element's exact group of indexes (its abstract targets) is
    # itself one intersection, provided it spans at least two abstract variables.
    groups: set[frozenset[int]] = set()
    for element, idx_group in inverted.items():
        if len(idx_group) >= 2:
            groups.add(frozenset(idx_group))
    return [set(group) for group in groups]


def cnc_intersection(
    abs_intersection: set[int],
    blocks: list[list[int]],
) -> set[int]:
    """Given a set of abstract variables, it returns the set of low-level variables that are in the intersection of their blocks."""
    intersection_vars = None
    for y in abs_intersection:
        if intersection_vars is None:
            intersection_vars = set(blocks[y])
        else:
            intersection_vars = intersection_vars.intersection(blocks[y])
    assert isinstance(intersection_vars, set)
    return intersection_vars


def partitions(elements: list[T]) -> Generator[list[list[T]], None, None]:
    """Yield all partitions of the input list."""
    if not elements:
        yield []
        return
    first, rest = elements[0], elements[1:]
    for partition in partitions(rest):
        # insert `first` in each of the subpartition's subsets
        for i, block in enumerate(partition):
            yield partition[:i] + [[first] + block] + partition[i + 1 :]
        # put `first` in its own subset
        yield partition + [[first]]


def multi_partitions(
    components: list[list[T]],
) -> Generator[list[list[T]], None, None]:
    # one factory per component, each yielding all partitions of that component
    factories = [partitions(component) for component in components]

    # cartesian product of the factories
    for combo in itertools.product(*factories):
        # combo is a tuple of partitions, one per component
        # we need to merge them into a single partition
        merged: list[list[T]] = []
        for part in combo:
            merged.extend(part)
        yield merged


def canonical(clusters: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Return a canonical representation of the clusters."""
    return tuple(sorted(tuple(sorted(c)) for c in clusters))
