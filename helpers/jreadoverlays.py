#!/usr/bin/jython
import sys
sys.path.append('/path/to/ij.jar')
import ij

if __name__ == '__main__':
    if len(sys.argv)<=1:
        print( "No file provided" )
        exit(1)
    fname = sys.argv[1]

    im = ij.io.Opener().openImage( fname )
    if im is None:
        exit(2)
    
    rois = im.getOverlay().toArray()
    
    roi = im.getRoi()
    if roi:
        rois.append(roi)
    
    w,h,_,_,_ = im.getDimensions()
    lines = []
    for roi in rois:
        if roi.getType() != ij.gui.Roi.POLYLINE:
            exit(3)
        x0, y0 = int(roi.getXBase()), int(roi.getYBase())
        coords = []
        for x, y in  zip(roi.getXCoordinates(), roi.getYCoordinates()):
            coords.append([x+x0, y+y0])
        lines.append(coords)
    output = {
        'dims': (w,h),
        'lines': lines
    }
    print( repr(output) )
