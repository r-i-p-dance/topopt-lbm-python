from topopt.src.core.brinkman_mixin import BrinkmanMixin
from lbm.src.cases.pressure import PressurePoiseuille

class PressureBrinkman(BrinkmanMixin, PressurePoiseuille):
    """Brinkman collision on a pressure-driven channel."""
    pass