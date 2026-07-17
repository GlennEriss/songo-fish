"""Moteur SongoFish (etape 8 du plan directeur) : reseau entraine
(ordonnancement + evaluation de feuille) + recherche Alpha-Beta bornee
(temps/noeuds/profondeur). Reutilise entierement
songo_ai.search.negamax -- aucune logique de recherche dupliquee
(principe section 12.1)."""

from songo_ai.hybrid.network_eval import make_network_evaluate, make_network_priority
from songo_ai.hybrid.songofish import SongoFishConfig, make_songofish_agent

__all__ = [
    "make_network_evaluate",
    "make_network_priority",
    "SongoFishConfig",
    "make_songofish_agent",
]
