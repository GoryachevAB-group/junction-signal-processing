#!/usr/bin/jython
import sys
import os
import java.awt as awt
sys.path.append('/path/to/ij.jar')
import ij


def get_field(info, field):
    pos = -1
    for line in info.split('\n'):
        pos = line.find(field)
        if pos >= 0:
            break
    if pos < 0:
        exit(0)
    return eval(line[pos + len(field) + 1:])


def poly(polygon):
    xs, ys = [], []
    for x, y in polygon:
        xs.append(float(x))
        ys.append(float(y))
    if xs[0] == xs[-1] and ys[0] == ys[-1]:
        roi_type = ij.gui.Roi.POLYGON
    else:
        roi_type = ij.gui.Roi.POLYLINE
    return ij.gui.PolygonRoi(ys, xs, roi_type)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("No file provided")
        exit(1)
    fname = sys.argv[1]

    im = ij.io.Opener().openImage(os.path.abspath(fname))
    if im is None:
        exit(2)
    comp = ij.CompositeImage(im)
    comp.setMode(1)

    info = comp.getInfoProperty()
    flare_rois = get_field(info, 'flare')
    path_roi = get_field(info, 'path')
    ranges = get_field(info, 'ranges')

    for i, (rmin, rmax) in enumerate(ranges):
        comp.setC(i + 1)
        comp.setDisplayRange(rmin, rmax)

    roi = poly(path_roi)
    comp.setRoi(roi)
    overlay = ij.gui.Overlay()
    for flare_roi in flare_rois:
        roi = poly(flare_roi)
        overlay.addElement(roi)
    overlay.setStrokeColor(awt.Color.white)
    comp.setOverlay(overlay)

    ij.io.FileSaver(comp).saveAsTiff(os.path.abspath(fname) + '.tif')
