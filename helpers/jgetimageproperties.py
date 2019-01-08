#!/usr/bin/jython
import sys
sys.path.append('/path/to/ij.jar')
import ij
# import json

"""
for f in Y*; do
    ~/host/sgmlib/jgetimageproperties.py $f/${f}.tif > $f/$f-properties.txt;
done
"""

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("No file provided")
        exit(1)
    fname = sys.argv[1]

    im = ij.io.Opener().openImage(fname)
    if im is None:
        exit(2)

    fi = im.getFileInfo()

    fields = "pixelDepth pixelWidth pixelHeight frameInterval".split()
    export = {
        't_delta': fi.frameInterval,
        'x_delta': fi.pixelWidth,
        'y_delta': fi.pixelHeight,
        'z_delta': fi.pixelDepth,
    }
    print(export)