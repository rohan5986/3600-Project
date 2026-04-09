"""
Leaf evaluation for search (expectiminimax / minimax).

Extend `evaluate_leaf` with richer components (mobility, carpet potential, rat EV, …).
"""

from __future__ import annotations

from typing import Protocol

from game.board import Board


class SupportsHeuristic(Protocol):
    """Callable shape for `ExpectiminimaxSearch(..., evaluate=...)`."""

    def __call__(self, board: Board, root_side_is_player_a: bool) -> float: ...


def evaluate_leaf(board: Board, root_side_is_player_a: bool) -> float:
    """
    Scalar utility from the root player's viewpoint (by canonical A/B scores).
    """
    pw = board.player_worker
    ow = board.opponent_worker
    a_pts = pw.get_points() if pw.is_player_a else ow.get_points()
    b_pts = ow.get_points() if pw.is_player_a else pw.get_points()
    if root_side_is_player_a:
        return float(a_pts - b_pts)
    return float(b_pts - a_pts)


def is_terminal_board(board: Board) -> bool:
    return board.is_game_over()


class Heuristic:
    """Namespace matching older `Heuristic.evaluate` usage in `agent.py`."""

    @staticmethod
    def evaluate(board: Board, root_side_is_player_a: bool) -> float:
        return evaluate_leaf(board, root_side_is_player_a)
