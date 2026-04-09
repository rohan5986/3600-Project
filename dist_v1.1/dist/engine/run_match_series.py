"""
Run many matches between two agents (e.g. Lilian vs Yolanda) and print aggregate stats.

From the directory that contains both `engine/` and `3600-agents/` (same as run_local_agents):

  python engine/run_match_series.py --games 50 Lilian Yolanda

Use --swap-half so each agent plays as player A for half of the games (fair seatings).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import sys
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from typing import Any

from gameplay import play_game

from game.board import Board
from game.enums import ResultArbiter, WinReason


@dataclass
class GameRecord:
    index: int
    player_a: str
    player_b: str
    winner: str
    win_reason: str
    score_a: int
    score_b: int
    turn_count: int
    seconds: float
    subject_was_player_a: bool
    subject_won: bool | None  # None if tie / ambiguous

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_scores(board: Board) -> tuple[int, int]:
    a_pts = b_pts = 0
    for w in (board.player_worker, board.opponent_worker):
        if w.is_player_a:
            a_pts = w.get_points()
        else:
            b_pts = w.get_points()
    return a_pts, b_pts


def _winner_name(board: Board) -> str:
    w = board.get_winner()
    if w is None:
        return "UNKNOWN"
    return w.name if hasattr(w, "name") else str(w)


def _reason_name(board: Board) -> str:
    r = board.get_win_reason()
    if r is None:
        return "UNKNOWN"
    if isinstance(r, WinReason):
        return r.name
    return str(r)


def _subject_won_tie(
    winner: str,
    subject_was_a: bool,
) -> bool | None:
    if winner == ResultArbiter.TIE.name:
        return None
    if winner == ResultArbiter.PLAYER_A.name:
        return subject_was_a
    if winner == ResultArbiter.PLAYER_B.name:
        return not subject_was_a
    return None


def run_series(
    play_directory: str,
    subject: str,
    player_a: str,
    player_b: str,
    games: int,
    swap_half: bool,
    quiet: bool,
    limit_resources: bool,
) -> tuple[list[GameRecord], Counter[str], Counter[str], dict[str, float]]:
    records: list[GameRecord] = []
    reason_ctr: Counter[str] = Counter()
    winner_ctr: Counter[str] = Counter()
    subject_wins = subject_losses = ties = 0
    t0_all = time.perf_counter()

    for i in range(games):
        use_swap = swap_half and (i >= games // 2)
        a_name = player_b if use_swap else player_a
        b_name = player_a if use_swap else player_b
        subject_was_a = subject == a_name

        buf_out = StringIO()
        buf_err = StringIO()
        t0 = time.perf_counter()
        if quiet:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                final_board, _rat_hist, _spawn_a, _spawn_b, _ma, _mb = play_game(
                    play_directory,
                    play_directory,
                    a_name,
                    b_name,
                    display_game=False,
                    delay=0.0,
                    clear_screen=False,
                    record=False,
                    limit_resources=limit_resources,
                )
        else:
            final_board, _rat_hist, _spawn_a, _spawn_b, _ma, _mb = play_game(
                play_directory,
                play_directory,
                a_name,
                b_name,
                display_game=False,
                delay=0.0,
                clear_screen=False,
                record=False,
                limit_resources=limit_resources,
            )
        elapsed = time.perf_counter() - t0

        winner = _winner_name(final_board)
        reason = _reason_name(final_board)
        sa, sb = _canonical_scores(final_board)
        sub_result = _subject_won_tie(winner, subject_was_a)

        reason_ctr[reason] += 1
        winner_ctr[winner] += 1
        if sub_result is True:
            subject_wins += 1
        elif sub_result is False:
            subject_losses += 1
        else:
            ties += 1

        records.append(
            GameRecord(
                index=i,
                player_a=a_name,
                player_b=b_name,
                winner=winner,
                win_reason=reason,
                score_a=sa,
                score_b=sb,
                turn_count=final_board.turn_count,
                seconds=elapsed,
                subject_was_player_a=subject_was_a,
                subject_won=sub_result,
            )
        )

    total_wall = time.perf_counter() - t0_all
    summary_extra = {
        "subject": subject,
        "games": games,
        "subject_wins": subject_wins,
        "subject_losses": subject_losses,
        "subject_ties_unresolved": ties,
        "wall_seconds": total_wall,
        "swap_half": swap_half,
    }
    return records, reason_ctr, winner_ctr, summary_extra


def main() -> None:
    p = argparse.ArgumentParser(description="Batch benchmark two agents (multiprocess game runner).")
    p.add_argument("player_a", help="Player A package name under 3600-agents (e.g. Lilian)")
    p.add_argument("player_b", help="Player B package name (e.g. Yolanda)")
    p.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of matches (default: 20)",
    )
    p.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Agent name to treat as 'your' bot for win%% stats (default: player_a)",
    )
    p.add_argument(
        "--swap-half",
        action="store_true",
        help="First half: player_a as A; second half: swap seats (B plays as A)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Do not suppress engine stdout/stderr per game",
    )
    p.add_argument(
        "--limit-resources",
        action="store_true",
        help="Use tournament-style limits (see gameplay.play_game)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON with per-game rows and summary (optional)",
    )
    args = p.parse_args()

    top_level = pathlib.Path(__file__).parent.parent.resolve()
    play_directory = os.path.join(top_level, "3600-agents")
    subject = args.subject or args.player_a

    records, reasons, winners, summary = run_series(
        play_directory=play_directory,
        subject=subject,
        player_a=args.player_a,
        player_b=args.player_b,
        games=args.games,
        swap_half=args.swap_half,
        quiet=not args.verbose,
        limit_resources=args.limit_resources,
    )

    n = args.games
    sw, sl, tu = summary["subject_wins"], summary["subject_losses"], summary["subject_ties_unresolved"]
    decided = sw + sl
    pct = (100.0 * sw / decided) if decided else 0.0

    print(f"Series: {args.player_a} (A first) vs {args.player_b} — {n} games")
    if args.swap_half:
        print(f"Seating: swap half-way ({n // 2} as listed, {n - n // 2} swapped)")
    print(f"Subject '{subject}' — wins: {sw}, losses: {sl}, ties/unresolved: {tu}")
    print(f"Subject win rate (decided games only): {pct:.1f}% ({sw}/{decided})")
    print(f"Outcome counts: {dict(winners)}")
    print(f"Win reasons: {dict(reasons)}")
    print(f"Total wall time: {summary['wall_seconds']:.1f}s ({summary['wall_seconds'] / max(n, 1):.2f}s / game)")

    if args.output:
        payload = {
            "summary": {**summary, "subject_win_rate_decided": pct},
            "winner_counts": dict(winners),
            "win_reason_counts": dict(reasons),
            "games": [r.to_json_dict() for r in records],
        }
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
