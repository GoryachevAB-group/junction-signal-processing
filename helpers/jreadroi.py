#!/usr/bin/jython
import sys
sys.path.append('/path/to/ij.jar')
import ij

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("No file provided")
        exit(1)
    fname = sys.argv[1]

    im = ij.io.Opener().openImage(fname)
    if im is None:
        exit(2)

    roi = im.getRoi()
    if roi.getType() != ij.gui.Roi.POLYLINE:
        exit(3)
    w, h, _, _, _ = im.getDimensions()
    x0, y0 = int(roi.getXBase()), int(roi.getYBase())
    coords = []
    for x, y in zip(roi.getXCoordinates(), roi.getYCoordinates()):
        coords.append([x + x0, y + y0])
    output = {'dims': (w, h), 'coords': coords}
    print(repr(output))
