"""
Hidden Markov Model over rat cell (64 states) using the official transition matrix
and observation model (noise + noisy Manhattan distance).
"""

from __future__ import annotations

import numpy as np

from game.board import Board
from game.enums import BOARD_SIZE
from game.rat import DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS, NOISE_PROBS
from game.rat import manhattan_distance


def _to_numpy_t(transition_matrix) -> np.ndarray:
    if transition_matrix is None:
        raise ValueError("transition_matrix is required for RatHMM")
    T = np.asarray(transition_matrix, dtype=np.float64)
    if T.shape != (64, 64):
        raise ValueError(f"Expected 64x64 transition matrix, got {T.shape}")
    rs = T.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0.0, 1.0, rs)
    return T / rs


def _noise_likelihood(cell_type, noise_enum) -> float:
    probs = NOISE_PROBS.get(cell_type)
    if probs is None:
        probs = NOISE_PROBS[list(NOISE_PROBS.keys())[0]]
    return float(probs[int(noise_enum)])


def _distance_likelihood(
    worker_xy: tuple[int, int], rat_xy: tuple[int, int], reported_dist: int
) -> float:
    d = manhattan_distance(worker_xy, rat_xy)
    s = 0.0
    for p, off in zip(DISTANCE_ERROR_PROBS, DISTANCE_ERROR_OFFSETS):
        obs = d + off
        if obs < 0:
            obs = 0
        if obs == reported_dist:
            s += float(p)
    return s


class RatHMM:
    """
    Belief b[i] = P(rat at cell i | observations so far), i = y*8+x.

    Each turn the engine does: rat.move() then (noise, est_dist). This class
    assumes `predict` is called once per turn before `update` with that tuple.
    """

    def __init__(self, transition_matrix, eps: float = 1e-12):
        self.T = _to_numpy_t(transition_matrix)
        self.eps = eps
        self.belief: np.ndarray
        self._spawn_distribution: np.ndarray
        self._init_spawn_distribution()
        self.belief = self._spawn_distribution.copy()

    def _init_spawn_distribution(self):
        e0 = np.zeros(64, dtype=np.float64)
        e0[0] = 1.0
        t1000 = np.linalg.matrix_power(self.T, 1000)
        self._spawn_distribution = e0 @ t1000
        s = self._spawn_distribution.sum()
        if s > 0:
            self._spawn_distribution /= s

    def reset_to_spawn_prior(self):
        """Use after a new rat is spawned (1000 headstart steps, distribution marginal)."""
        self.belief = self._spawn_distribution.copy()

    @property
    def transition_matrix(self) -> np.ndarray:
        return self.T

    def predict(self):
        """One rat step: call once per turn before incorporating the new observation."""
        self.belief = self.belief @ self.T
        self._normalize()

    def update(self, board: Board, noise, estimated_distance: int):
        lik = np.zeros(64, dtype=np.float64)
        worker_xy = board.player_worker.get_location()
        for i in range(64):
            x = i % BOARD_SIZE
            y = i // BOARD_SIZE
            rat_xy = (x, y)
            cell = board.get_cell(rat_xy)
            lik[i] = _noise_likelihood(cell, noise) * _distance_likelihood(
                worker_xy, rat_xy, int(estimated_distance)
            )
        self.belief *= lik
        self._normalize()

    def _normalize(self):
        s = self.belief.sum()
        if s <= 0:
            self.belief[:] = 1.0 / 64.0
        else:
            self.belief /= s
        self.belief = np.maximum(self.belief, self.eps)
        self.belief /= self.belief.sum()

    def most_likely_cell(self) -> tuple[int, int]:
        i = int(self.belief.argmax())
        return (i % BOARD_SIZE, i // BOARD_SIZE)

    def eliminate_cell(self, loc: tuple[int, int] | None) -> None:
        """Hard evidence the rat was not at `loc` (e.g. opponent failed search there)."""
        if loc is None:
            return
        x, y = loc
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return
        i = y * BOARD_SIZE + x
        self.belief[i] = 0.0
        self._normalize()

    def should_reset_from_board_hints(self, board: Board) -> bool:
        """Successful search by either player spawns a new rat."""
        o_loc, o_ok = board.opponent_search
        p_loc, p_ok = board.player_search
        if o_ok is True:
            return True
        if p_ok is True:
            return True
        return False
