#!/usr/bin/env python3
"""Table de jeu Songo -- mode spectateur (premiere version) : deux agents
s'affrontent, on regarde. Meme moteur que le reste du projet
(songo_ai.songo.rules.SongoLegacyGame -- la variante de reference, choisie
ici pour sa trace de distribution/capture qui alimente l'animation ;
aucune regle dupliquee). Chaque coup s'anime case par case au lieu de
sauter directement au resultat final.

Exemples (depuis la racine du repo) :
    .venv/bin/python apps/table/src/main.py --player1 model:champion --player2 minimax:4
    .venv/bin/python apps/table/src/main.py --player1 minimax:1 --player2 minimax:6 --delay 0.3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pygame

import config as C
from animation import MoveAnimation
from board_view import BoardView
from controllers import make_controller

from songo_ai.songo.rules import PLAYER_ONE, SongoLegacyGame, local_action_to_pit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--player1", default="minimax:2", help="random | minimax:N | model:VERSION|champion")
    parser.add_argument("--player2", default="minimax:2")
    parser.add_argument("--delay", type=float, default=0.5, help="pause (s) entre la fin d'une animation et le coup suivant")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


_REASON_LABELS = {
    "player_1_store_over_35": "magasin > 35",
    "player_2_store_over_35": "magasin > 35",
    "stores_35_35": "35-35",
    "famine_after_move": "famine (territoire adverse asseche)",
}


def _winner_message(game: SongoLegacyGame, label1: str, label2: str) -> str:
    reason = game.history[-1].reason if game.history else ""
    reason_label = _REASON_LABELS.get(reason, reason)
    suffix = f" -- {reason_label}" if reason_label else ""

    if game.winner == 0:
        return f"Match nul{suffix}"
    if game.winner == PLAYER_ONE:
        return f"Victoire Joueur 1 ({label1}){suffix}"
    return f"Victoire Joueur 2 ({label2}){suffix}"


def run() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    agent1, label1 = make_controller(args.player1)
    agent2, label2 = make_controller(args.player2)

    pygame.init()
    screen = pygame.display.set_mode((C.WINDOW_WIDTH, C.WINDOW_HEIGHT))
    pygame.display.set_caption("SongoFish -- table (spectateur)")
    clock = pygame.time.Clock()
    view = BoardView(screen)

    game = SongoLegacyGame()
    animation: Optional[MoveAnimation] = None
    wait_elapsed = 0.0
    finished_message = ""

    def reset() -> None:
        nonlocal game, animation, wait_elapsed, finished_message
        game = SongoLegacyGame()
        animation = None
        wait_elapsed = 0.0
        finished_message = ""

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset()

        # L'animation du coup final (celui qui termine la partie) doit
        # toujours pouvoir se derouler jusqu'au bout : on ne conditionne PAS
        # cette mise a jour a `not game.finished`, sinon elle se figeait
        # instantanement des sa creation pendant que le message de victoire
        # s'affichait deja (bug remonte par l'utilisateur : le magasin
        # semblait "s'arreter" avant d'avoir vraiment atteint son compte final).
        if animation is not None and not animation.finished:
            animation.update(dt)
        elif not game.finished:
            wait_elapsed += dt
            if wait_elapsed >= args.delay:
                wait_elapsed = 0.0
                legal = game.legal_local_actions()
                if not legal:
                    game.normalize_terminal()
                else:
                    current_agent = agent1 if game.turn == PLAYER_ONE else agent2
                    local_action = current_agent(game, rng)
                    if local_action not in legal:
                        local_action = legal[0]

                    board_before = list(game.board)
                    source_pit = local_action_to_pit(game.turn, local_action)
                    game.play_local(local_action)

                    animation = MoveAnimation.start(
                        board_before=board_before,
                        source_pit=source_pit,
                        sow_trace=game.last_sow_trace,
                        capture_trace=game.last_capture_trace,
                        final_board=list(game.board),
                    )
                    if game.finished:
                        finished_message = _winner_message(game, label1, label2)

        display_board = animation.board if animation is not None else list(game.board)
        source_pit = animation.source_pit if animation is not None else None
        receiving_pit = animation.receiving_pit if animation is not None else None
        capturing = animation is not None and animation.phase == "capturing"
        animation_done = animation is None or animation.finished

        view.draw_board(display_board, f"J1: {label1}", f"J2: {label2}", source_pit, receiving_pit, capturing)
        view.draw_header(
            "SongoFish -- table (spectateur)",
            f"J1: {label1}   vs   J2: {label2}   (R = recommencer)",
        )
        if game.finished and animation_done:
            view.draw_status(finished_message, C.COLOR_STATUS_GOOD)
        elif not game.finished:
            turn_label = label1 if game.turn == PLAYER_ONE else label2
            view.draw_status(f"Au tour de : Joueur {game.turn} ({turn_label})", C.COLOR_ACCENT)
        # sinon : partie terminee mais animation du dernier coup encore en
        # cours -> pas de message tant que le magasin n'a pas fini de compter.

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
