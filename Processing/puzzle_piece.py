"""Classes to interface and process puzzle pieces."""

import numpy as np
from loguru import logger


class PuzzlePiece:
    """Individual Puzzle Piece"""

    def __init__(self, image: np.ndarray):
        self.image = image




class Puzzle:
    """Entire Puzzle"""
