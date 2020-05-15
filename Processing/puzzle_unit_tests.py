"""Puzzle Unit Tests"""
import os
import unittest

import numpy as np
import cv2 as cv
import preprocessing


@unittest.skip
class PDFConversionTester(unittest.TestCase):
    """PDF Conversion Tester"""
    INPUT = './Datasets/Background.pdf'
    DPI = 200
    OUTPUT = './Datasets/Background/'

    def test_pdf_convert(self):
        os.makedirs(self.OUTPUT, exist_ok=True)
        preprocessing.convert_pdf_to_image(self.INPUT, self.DPI, self.OUTPUT)


# @unittest.skip
class PuzzlePieceExtractTester(unittest.TestCase):
    """Testing extraction of puzzle pieces."""
    OUTPUT = './Datasets/Dinosaur_Large_48_Piece_Dataset/'
    BACKGROUND = os.path.join(OUTPUT, '0.jpg')
    TEST = os.path.join(OUTPUT, '1.jpg')

    def test_get_mask(self):
        background = cv.imread(self.BACKGROUND)
        test = cv.imread(self.TEST)
        result = preprocessing.get_mask_puzzle_pieces_background(background,
                                                                 test)
        for r in result:
            cv.imshow('Ouput', cv.resize(r, (480, 480)))
            cv.waitKey(0)

    def test_get_edges_and_corners(self):
        background = cv.imread(self.BACKGROUND)
        test = cv.imread(self.TEST)
        result = preprocessing.get_mask_puzzle_pieces_background(background,
                                                                 test)
        for r in result:
            r_corner, r_side = preprocessing.get_piece_corner_side(r)
            blank = np.zeros((test.shape[0], test.shape[1], 3))
            output = cv.drawContours(blank, [r_corner], -1, (255, 255, 0), 3, cv.FILLED)
            output = cv.drawContours(blank, [r_side], -1, (0, 255, 255), 3, cv.FILLED)
            cv.imshow('Ouput', cv.resize(output, (480, 480)))
            cv.waitKey(0)


if __name__ == '__main__':
    unittest.main()
