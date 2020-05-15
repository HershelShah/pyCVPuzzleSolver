"""Preprocessing functions for Jigsaw Puzzle Solver Pipeline."""
import os
import numpy as np
import cv2 as cv
from loguru import logger
from pdf2image import convert_from_path
from skimage.segmentation import slic

KERNEL_SHARPENING = np.array([[-4, -4, -4],
                              [-4, 36, -4],
                              [-4, -4, -4]])


def convert_pdf_to_image(source: str, dpi: int, output_folder: str) -> None:
    """Convert multipage image to a directory of images.

    Parameters:
    -----------
        source: str
            PDF File of scanned pieces.
        dpi: int
            Image resolution.
        output_folder: str
            Folder of scanned puzzle pieces as images.
    """
    logger.info('Starting conversion')
    pages = convert_from_path(source, dpi)
    number = 0
    for page in pages:
        filename = os.path.join(output_folder, ''.join([str(number), '.jpg']))
        page.save(filename)
        logger.info(f'Processed {number} of {len(pages)}')
        number += 1
    logger.info('Finished conversion')


def get_mask_puzzle_pieces_background(background: np.ndarray,
                                      image: np.ndarray) -> list:
    """Function gets puzzle image from image.

    Parameters:
    -----------
        background: np.ndarray
            Background image used for segmentation of the puzzle pieces.
        image: np.ndarray
            Image of the puzzle pieces.

    Returns:
    --------
        pieces_mask: list
            Mask of all possible pieces. Lighting may create extra pieces
            which need to be filtered out at a later stage.

    """
    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_mask = cv.absdiff(background, image)
    image_mask = cv.medianBlur(image_mask, 11)
    ret, image_mask = cv.threshold(image_mask, 20, 255, cv.THRESH_BINARY)
    image_mask = cv.morphologyEx(image_mask,
                                 cv.MORPH_OPEN,
                                 (5, 5),
                                 iterations=3)
    image_mask = cv.filter2D(image_mask, -1, KERNEL_SHARPENING)
    puzzle_pieces, hiearchy = cv.findContours(image_mask,
                                              cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_NONE)
    pieces_mask = []
    for piece in range(len(puzzle_pieces)):
        blank = np.zeros((image_mask.shape[0], image_mask.shape[1]))
        single_piece = cv.drawContours(blank, puzzle_pieces, piece, 255,
                                       cv.FILLED)
        single_piece = cv.cvtColor(single_piece.astype(np.uint8),
                                   cv.COLOR_GRAY2BGR)
        pieces_mask.append(single_piece)
    return pieces_mask


def get_piece_corner_side(puzzle: np.ndarray) -> list:
    """
    """
    puzzle_grey = cv.cvtColor(puzzle, cv.COLOR_BGR2GRAY).astype(np.uint8)
    puzzle_piece_outline, hiearchy = cv.findContours(puzzle_grey,
                                                     cv.RETR_CCOMP,
                                                     cv.CHAIN_APPROX_SIMPLE)
    assert(len(puzzle_piece_outline) == 1)
    corner_epsilon = 0.1 * cv.arcLength(puzzle_piece_outline[0], True)
    side_epsilon = 0.001 * cv.arcLength(puzzle_piece_outline[0], True)
    corner_approx = cv.approxPolyDP(puzzle_piece_outline[0], corner_epsilon, True)
    side_approx = cv.approxPolyDP(puzzle_piece_outline[0], side_epsilon, True)
    return corner_approx, side_approx


def get_edge_features(puzzle: np.ndarry, corner_approx: np.ndarray, side_approx: np.ndarray):
    """
    """

