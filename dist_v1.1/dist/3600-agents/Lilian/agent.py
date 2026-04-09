from __future__ import annotations

from collections.abc import Callable
from typing import Tuple

from game import board, enums, move

from .expectiminimax import ExpectiminimaxSearch
from .heuristics import Heuristic
from .rat_hmm import RatHMM


class PlayerAgent:
    """
    HMM rat belief + depth-limited expectiminimax over worker moves.

    Heuristic is a stub until you plug in features in `heuristics.py`.
    SEARCH moves are omitted from the adversarial tree (large branching); use
    `self._rat` in a future search policy.
    """

    def __init__(
        self,
        board: board.Board,
        transition_matrix=None,
        time_left: Callable | None = None,
    ):
        _ = board, time_left
        if transition_matrix is None:
            raise ValueError("transition_matrix is required for RatHMM")
        self._rat = RatHMM(transition_matrix)
        self._search = ExpectiminimaxSearch(
            max_depth=2,
            evaluate=Heuristic.evaluate,
            max_depth_cap=6,
        )

    def commentate(self) -> str:
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        noise, est_distance = sensor_data

        opp_loc, opp_found = board.opponent_search
        _p_loc, p_found = board.player_search
        if opp_found or p_found:
            self._rat.reset_to_spawn_prior()
            if p_found:
                # Opponent had a full turn after spawn; rat stepped once before their obs.
                self._rat.predict()
        elif opp_loc is not None:
            self._rat.eliminate_cell(opp_loc)

        self._rat.predict()
        self._rat.update(board, noise, int(est_distance))

        chosen = self._search.search_best_move(
            board,
            time_left=time_left,
            reserve_sec=0.45,
        )
        if chosen is not None:
            return chosen

        moves = board.get_valid_moves(exclude_search=True)
        if moves:
            return moves[0]
        return move.Move.plain(enums.Direction.UP)
