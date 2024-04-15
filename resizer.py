import PIL
from PIL import Image
import os, sys
import tqdm
path = "./_out/"
dirs = os.listdir( path )
def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            img = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            img = img.resize((160,80 ), Image.ANTIALIAS)
            img.save(f + '.png') 
resize()