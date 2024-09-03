# import numpy as np
# import cv2 as cv
import sys
from utils import *
from img import *

print(f"current recursion limit is : {sys.getrecursionlimit()}") # mostly 3000
sys.setrecursionlimit(6000)
print(f"New recursion limit is : {sys.getrecursionlimit()}")

path = "raw_invoices/Invoice.jpg"
raw_img = utils.read(path)

utils.show(raw_img)

if __name__ == "__main__":
    print("hi")
    img = Image(raw_img)
    binInv = img.process_img()
    utils.show(binInv)
    mask = Mask(binInv)

    Vlines, Hlines = mask.extract()
    print("---------------")
    print(Vlines)
    print("---------------")
    print(Hlines)

    mask.draw_line(img.original_img, Hlines, Vlines)
    # mask.crop_img(img.original_img, Hlines, Vlines)
    words = Word.read_image(raw_img)
    # Word.show_words(raw_img.copy(), words)

    Vintersection = Mask.find_intersection(Vlines)
    Hintersection = Mask.find_intersection(Hlines)

    print(Vintersection)
    print(Hintersection)

    
    Word().show_table_words(Vintersection, raw_img, words)
    Word().show_table_words(Hintersection, raw_img, words)

    Word().to_excel()




