"""Rendu du plateau (section "premiere version" : formes geometriques,
piles de pions en points, pas de texture).

Sens de semis -- sens des aiguilles d'une montre, comme au Songo reel.
Verifie contre l'exemple du livre (Mbarga Owona, p.20) : jouer S3 (5
graines) distribue vers S2, S1, S0 (index decroissant dans la rangee sud)
puis N0, N1 (index croissant dans la rangee nord). Nos index internes
0..13 croissants (cf. songo_ai.songo.rules) suivent exactement cet ordre,
donc :
  - rangee du bas (pits 0-6, joueur 1) : affichee de DROITE a GAUCHE
    (pit 0 = case "1" du joueur, la plus a droite ; pit 6 = case la plus
    a gauche, adjacente a la rangee du haut).
  - rangee du haut (pits 7-13, joueur 2) : affichee de GAUCHE a DROITE
    (pit 7 = case "1" du joueur, la plus a gauche, juste apres le pit 6 ;
    pit 13 = case la plus a droite).
Le circuit visuel est donc bien circulaire et dans le sens horaire :
droite->gauche en bas, gauche->droite en haut."""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import pygame

from songo_ai.songo.rules import P1_STORE, P2_STORE

import config as C


def pit_position(index: int) -> Tuple[int, int]:
    if 0 <= index <= 6:
        # rangee du bas : pit 0 a droite, pit 6 a gauche (semis droite->gauche)
        x = C.BOARD_LEFT + (6 - index) * C.PIT_SPACING_X
        return x, C.ROW_Y_BOTTOM
    if 7 <= index <= 13:
        # rangee du haut : pit 7 a gauche, pit 13 a droite (semis gauche->droite)
        x = C.BOARD_LEFT + (index - 7) * C.PIT_SPACING_X
        return x, C.ROW_Y_TOP
    raise ValueError(f"pas un pit jouable: {index}")


def store_rect(store_index: int) -> pygame.Rect:
    last_pit_right_edge = C.BOARD_LEFT + 6 * C.PIT_SPACING_X + C.PIT_RADIUS
    if store_index == P1_STORE:
        x = last_pit_right_edge + C.STORE_GAP
    elif store_index == P2_STORE:
        first_pit_left_edge = C.BOARD_LEFT - C.PIT_RADIUS
        x = first_pit_left_edge - C.STORE_GAP - C.STORE_WIDTH
    else:
        raise ValueError(f"pas un magasin: {store_index}")
    return pygame.Rect(x, C.STORE_Y, C.STORE_WIDTH, C.STORE_HEIGHT)


class BoardView:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font_pit = pygame.font.SysFont(C.FONT_NAME, C.FONT_SIZE_PIT_COUNT, bold=True)
        self.font_label = pygame.font.SysFont(C.FONT_NAME, C.FONT_SIZE_LABEL)
        self.font_header = pygame.font.SysFont(C.FONT_NAME, C.FONT_SIZE_HEADER, bold=True)
        self.font_status = pygame.font.SysFont(C.FONT_NAME, C.FONT_SIZE_STATUS, bold=True)
        self._seed_jitter_cache: Dict[Tuple[int, int], list] = {}

    def _seed_offsets(self, index: int, count: int) -> list:
        key = (index, min(count, C.MAX_SEED_DOTS))
        if key not in self._seed_jitter_cache:
            rng = random.Random(index * 97 + count)
            offsets = []
            radius = C.PIT_RADIUS - 14
            for _ in range(min(count, C.MAX_SEED_DOTS)):
                angle = rng.uniform(0, 6.283)
                dist = rng.uniform(0, radius)
                point = dist * pygame.math.Vector2(1, 0).rotate_rad(angle)
                offsets.append((point.x, point.y))
            self._seed_jitter_cache[key] = offsets
        return self._seed_jitter_cache[key]

    def draw_pit(self, index: int, count: int, is_source: bool = False, receiving_color=None) -> None:
        x, y = pit_position(index)

        if receiving_color is not None:
            border_color, width = receiving_color, 5
        elif is_source:
            border_color, width = C.COLOR_GOLD_SOURCE, 4
        else:
            border_color, width = C.COLOR_PIT_BORDER, 2

        pygame.draw.circle(self.screen, C.COLOR_PIT, (x, y), C.PIT_RADIUS)
        pygame.draw.circle(self.screen, border_color, (x, y), C.PIT_RADIUS, width=width)

        for dx, dy in self._seed_offsets(index, count):
            pygame.draw.circle(self.screen, C.COLOR_SEED, (int(x + dx), int(y + dy)), C.SEED_DOT_RADIUS)

        label = self.font_pit.render(str(count), True, C.COLOR_TEXT_PRIMARY)
        self.screen.blit(label, label.get_rect(center=(x, y + C.PIT_RADIUS + 16)))

    def draw_store(self, store_index: int, count: int, owner_label: str, receiving: bool = False) -> None:
        rect = store_rect(store_index)
        border_color = C.COLOR_STORE_RECEIVING if receiving else C.COLOR_STORE_BORDER
        pygame.draw.rect(self.screen, C.COLOR_STORE, rect, border_radius=14)
        pygame.draw.rect(self.screen, border_color, rect, width=4 if receiving else 2, border_radius=14)

        label = self.font_label.render(owner_label, True, C.COLOR_TEXT_SECONDARY)
        self.screen.blit(label, label.get_rect(center=(rect.centerx, rect.top + 24)))

        count_label = self.font_header.render(str(count), True, C.COLOR_TEXT_PRIMARY)
        self.screen.blit(count_label, count_label.get_rect(center=(rect.centerx, rect.centery)))

    def draw_row_label(self, text: str, y: int, color: Tuple[int, int, int]) -> None:
        label = self.font_label.render(text, True, color)
        self.screen.blit(label, label.get_rect(center=(C.WINDOW_WIDTH // 2, y)))

    def draw_board(
        self,
        board: list,
        p1_label: str,
        p2_label: str,
        source_pit: Optional[int] = None,
        receiving_pit: Optional[int] = None,
        capturing: bool = False,
    ) -> None:
        self.screen.fill(C.COLOR_BACKGROUND)
        receiving_color = C.COLOR_GOLD_CAPTURE if capturing else C.COLOR_GOLD_RECEIVING

        for index in range(14):
            self.draw_pit(
                index,
                board[index],
                is_source=(index == source_pit),
                receiving_color=receiving_color if index == receiving_pit else None,
            )

        store_receiving = receiving_pit in (P1_STORE, P2_STORE)
        # Le magasin est trop etroit pour un nom de controleur complet
        # (deja vu : ca deborde) -- il garde un label court fixe. Le nom
        # complet passe par p1_label/p2_label est reserve a l'etiquette de
        # rangee ci-dessous, qui a toute la largeur de la fenetre.
        self.draw_store(P1_STORE, board[P1_STORE], "J1", receiving=(store_receiving and receiving_pit == P1_STORE))
        self.draw_store(P2_STORE, board[P2_STORE], "J2", receiving=(store_receiving and receiving_pit == P2_STORE))

        # Rappel explicite de qui joue quelle rangee, dans l'espace vide
        # entre les deux rangees (juste sous celle du haut / juste au-dessus
        # de celle du bas) -- l'ancien affichage ne le montrait qu'en petit
        # texte dans les magasins, pas assez visible pour suivre une partie.
        self.draw_row_label(f"^ {p2_label} (rangee du haut)", C.ROW_Y_TOP + 85, C.COLOR_ACCENT_2)
        self.draw_row_label(f"v {p1_label} (rangee du bas)", C.ROW_Y_BOTTOM - 60, C.COLOR_ACCENT)

    def draw_header(self, text: str, sub_text: str = "") -> None:
        header = self.font_header.render(text, True, C.COLOR_TEXT_PRIMARY)
        self.screen.blit(header, (20, 20))
        if sub_text:
            sub = self.font_label.render(sub_text, True, C.COLOR_TEXT_SECONDARY)
            self.screen.blit(sub, (20, 48))

    def draw_status(self, text: str, color: Tuple[int, int, int] = C.COLOR_TEXT_PRIMARY) -> None:
        status = self.font_status.render(text, True, color)
        self.screen.blit(status, status.get_rect(center=(C.WINDOW_WIDTH // 2, C.WINDOW_HEIGHT - 30)))
