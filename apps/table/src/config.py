"""Constantes de mise en page et de couleurs (plateau simple et
fonctionnel, section "premiere version" -- pas de texture bois pour
l'instant, cf. apps/table/README.md)."""

from __future__ import annotations

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 560

PIT_RADIUS = 42
PIT_SPACING_X = 128
ROW_Y_TOP = 180
ROW_Y_BOTTOM = 380
BOARD_LEFT = 222  # centre du premier pit (index 0 bas-gauche / index 13 haut-gauche)

STORE_MARGIN = 30  # marge exterieure fenetre
STORE_GAP = 40  # espace entre le dernier pit et le magasin
STORE_WIDTH = 110
STORE_HEIGHT = 340
STORE_Y = 130

SEED_DOT_RADIUS = 4
MAX_SEED_DOTS = 18  # au-dela, on affiche juste le nombre

COLOR_BACKGROUND = (247, 245, 240)
COLOR_PIT = (255, 255, 255)
COLOR_PIT_BORDER = (70, 68, 64)
COLOR_GOLD_SOURCE = (196, 155, 30)  # case jouee (bordure doree, reste affichee toute l'animation)
COLOR_GOLD_RECEIVING = (230, 190, 60)  # case qui recoit la graine animee en ce moment
COLOR_GOLD_CAPTURE = (214, 90, 40)  # case en cours de capture (teinte distincte, chaude)
COLOR_STORE = (235, 231, 222)
COLOR_STORE_BORDER = (70, 68, 64)
COLOR_STORE_RECEIVING = (230, 190, 60)
COLOR_SEED = (60, 45, 30)
COLOR_TEXT_PRIMARY = (11, 11, 11)
COLOR_TEXT_SECONDARY = (82, 81, 78)
COLOR_ACCENT = (42, 120, 214)
COLOR_ACCENT_2 = (27, 175, 122)
COLOR_STATUS_GOOD = (12, 163, 12)

FONT_NAME = None  # police systeme par defaut
FONT_SIZE_PIT_COUNT = 20
FONT_SIZE_LABEL = 16
FONT_SIZE_HEADER = 22
FONT_SIZE_STATUS = 26
