from .core import AttackContext, AttackStrategy
from .jmpa import MpapfrAttackStrategy


class DoNothingAttackStrategy(AttackStrategy):
    pass


__all__ = [
    "AttackContext",
    "AttackStrategy",
    "DoNothingAttackStrategy",
    "MpapfrAttackStrategy",
]
