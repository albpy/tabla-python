import cv2 as cv
import numpy
from constants import *

class utils:
    def read(path):
        img = cv.imread(path)
        return img
    
    def show(img):
        cv.imshow('image', img)
        cv.waitKey(5000)
        cv.destroyAllWindows()

    def color_2_gray(img):
        if img.shape[-1] == 3: 
            grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            return grey
        else:
            return img
    def invert_img(img):
        inverted_img = cv.bitwise_not(img)
        return inverted_img
    
    def sperate_fgd(img):
        binInv = cv.adaptiveThreshold(img, 255, 
                                      cv.ADAPTIVE_THRESH_MEAN_C, 
                                      cv.THRESH_BINARY, 
                                      constants.ADAPTIVE_THRESHOLD_BLOCK_SIZE, 
                                      constants.ADAPTIVE_THRESHOLD_CONSTANT)
        return binInv
 
