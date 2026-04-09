"""
Expectiminimax-style tree search over deterministic worker moves.

The environment also has stochastic rat observations; those are tracked separately via
`RatHMM`. This module focuses on adversarial alternation MAX (root player) /
MIN (opponent). A dedicated `ChanceNode` placeholder is reserved for later extensions
(e.g. averaging over stochastic transitions or search outcomes).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from game.board import Board
from game.move import Move

from .heuristics import Heuristic, SupportsHeuristic


class NodeKind(Enum):
    MAX = auto()
    MIN = auto()
    CHANCE = auto()


@dataclass
class ChanceOutcome:
    """Placeholder for future weighted branches (rat roll, search noise, etc.)."""

    label: str
    probability: float
    value: float


def expectiminimax_stub_chance_children(
    board: Board,
    build_children: Callable[[Board], list[tuple[str, float, Board]]],
) -> float:
    """
    Expectation at a CHANCE node: sum_k p_k * V(child_k).

    Currently unused by `search_best_move`; hook for future stochastic layers.
    """
    total = 0.0
    for _label, p, _child_board in build_children(board):
        total += p * Heuristic.evaluate(_child_board, True)
    return total


class ExpectiminimaxSearch:
    """
    Depth-limited MAX/MIN with alpha-beta pruning. After each simulated ply the board
    perspective is reversed so `player_worker` is always the side to move, matching
    tournament `gameplay.py`.

    CHANCE (expectation) nodes are not wired into the main recursion yet; extend
    `_minimax` when you add stochastic branches at interior nodes.
    """

    def __init__(
        self,
        max_depth: int = 2,
        evaluate: Optional[SupportsHeuristic] = None,
        max_depth_cap: int = 6,
    ):
        self.max_depth = max_depth
        self.max_depth_cap = max_depth_cap
        self.evaluate = evaluate or Heuristic.evaluate
        self._abort_time_left: Callable[[], float] | None = None
        self._abort_reserve: float = 0.0
        self._node_counter: int = 0

    def search_best_move(
        self,
        board: Board,
        time_left: Callable[[], float] | None = None,
        reserve_sec: float = 0.45,
    ) -> Optional[Move]:
        """
        With `time_left`, iteratively deepens from depth 1 until time falls below
        `reserve_sec` or `max_depth_cap` is reached; returns the best move from the
        last completed depth. With `time_left` None, runs one search at `max_depth`.

        While a depth is being searched, `time_left` is polled periodically inside
        minimax so one deep iteration cannot run unbounded.
        """
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return None

        if time_left is None:
            self._abort_time_left = None
            return self._search_at_depth(board, self.max_depth)

        self._abort_time_left = time_left
        self._abort_reserve = reserve_sec
        try:
            last_best: Move = moves[0]
            depth = 1
            while depth <= self.max_depth_cap:
                if time_left() <= reserve_sec:
                    break
                found = self._search_at_depth(board, depth)
                if found is not None:
                    last_best = found
                depth += 1

            return last_best
        finally:
            self._abort_time_left = None

    def _deadline_hit(self) -> bool:
        if self._abort_time_left is None:
            return False
        self._node_counter += 1
        if self._node_counter & 1023:
            return False
        return self._abort_time_left() <= self._abort_reserve

    def _search_at_depth(self, board: Board, ply_depth: int) -> Optional[Move]:
        """Root MAX over legal moves; tree extends `ply_depth` plies from the root."""
        self._node_counter = 0
        root_side_is_player_a = board.player_worker.is_player_a
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return None

        best: Optional[Move] = None
        best_val = float("-inf")

        for mv in moves:
            if (
                self._abort_time_left is not None
                and self._abort_time_left() <= self._abort_reserve
            ):
                break
            child = board.forecast_move(mv, check_ok=True)
            if child is None:
                continue
            child = self._board_after_ply(child)
            val = self._minimax(
                child,
                depth=ply_depth - 1,
                maximizing=False,
                alpha=float("-inf"),
                beta=float("inf"),
                root_side_is_player_a=root_side_is_player_a,
            )
            if val > best_val:
                best_val = val
                best = mv

        return best if best is not None else moves[0]

    def _board_after_ply(self, board: Board) -> Board:
        board.reverse_perspective()
        return board

    def _minimax(
        self,
        board: Board,
        depth: int,
        maximizing: bool,
        alpha: float,
        beta: float,
        root_side_is_player_a: bool,
    ) -> float:
        if self._deadline_hit():
            return self.evaluate(board, root_side_is_player_a)
        if depth <= 0 or board.is_game_over():
            return self.evaluate(board, root_side_is_player_a)

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return self.evaluate(board, root_side_is_player_a)

        if maximizing:
            value = float("-inf")
            for mv in moves:
                child = board.forecast_move(mv, check_ok=True)
                if child is None:
                    continue
                child = self._board_after_ply(child)
                value = max(
                    value,
                    self._minimax(
                        child,
                        depth - 1,
                        False,
                        alpha,
                        beta,
                        root_side_is_player_a,
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = float("inf")
        for mv in moves:
            child = board.forecast_move(mv, check_ok=True)
            if child is None:
                continue
            child = self._board_after_ply(child)
            value = min(
                value,
                self._minimax(
                    child,
                    depth - 1,
                    True,
                    alpha,
                    beta,
                    root_side_is_player_a,
                ),
            )
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
