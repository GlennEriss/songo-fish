"""Professeur Alpha-Beta profond, annotation adaptative, cache (etape 4 du plan directeur)."""

from .cache import AnnotationCache, cache_key
from .deep_teacher import ENRICHI, PREMIUM, STANDARD, Annotation, DeepTeacher, TeacherConfig

__all__ = [
    "AnnotationCache",
    "cache_key",
    "ENRICHI",
    "PREMIUM",
    "STANDARD",
    "Annotation",
    "DeepTeacher",
    "TeacherConfig",
]
