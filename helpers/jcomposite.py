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
    comp = ij.CompositeImage(im)
    comp.setMode(1)

    field = 'kymoranges:'
    info = comp.getInfoProperty()
    pos = -1
    for line in info.split('\n'):
        pos = line.find(field)
        if pos >= 0:
            break
    if pos < 0:
        exit(0)
    arr = eval(line[pos+len(field):])
    
    for i, (rmin, rmax) in  enumerate(arr):
        comp.setC(i+1)
        comp.setDisplayRange(rmin, rmax)
    
    ij.io.FileSaver(comp).saveAsTiff(fname)