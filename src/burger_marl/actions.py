"""Standalone action contract for the burger collaboration environment.

This module intentionally has no dependency on the repository's original
Overcooked-AI soup environment. It follows the commercial game's stable core:
cardinal movement plus separate context-sensitive pick/drop and process
buttons. Throwing and dashing are reserved for a later physics extension.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Point = Tuple[int, int]
Motion = Tuple[int, int]


class Direction:
    NORTH: Motion = (0, -1)
    SOUTH: Motion = (0, 1)
    EAST: Motion = (1, 0)
    WEST: Motion = (-1, 0)
    ALL_DIRECTIONS: List[Motion] = [NORTH, SOUTH, EAST, WEST]
    INDEX_TO_DIRECTION = ALL_DIRECTIONS
    DIRECTION_TO_INDEX: Dict[Motion, int] = {
        direction: index
        for index, direction in enumerate(INDEX_TO_DIRECTION)
    }


class Action:
    STAY: Motion = (0, 0)
    PICK_DROP = "pick_drop"
    PROCESS = "process"

    INDEX_TO_ACTION = Direction.INDEX_TO_DIRECTION + [
        STAY,
        PICK_DROP,
        PROCESS,
    ]
    ALL_ACTIONS = INDEX_TO_ACTION
    ACTION_TO_INDEX = {
        action: index for index, action in enumerate(INDEX_TO_ACTION)
    }
    MOTION_ACTIONS = Direction.ALL_DIRECTIONS + [STAY]
    NUM_ACTIONS = len(ALL_ACTIONS)

    @staticmethod
    def move_in_direction(point: Point, direction: Motion) -> Point:
        if direction not in Action.MOTION_ACTIONS:
            raise ValueError("Not a motion action: {!r}".format(direction))
        x, y = point
        dx, dy = direction
        return x + dx, y + dy


__all__ = ["Action", "Direction"]
