import numpy as np
import matplotlib

def cell_centers(img):
    max_cells = img.max()

    m = np.zeros( img.shape, np.uint8 )
    vx = np.arange( img.shape[1] )
    vy = np.arange( img.shape[0] )
    dct = dict()
    for i in range(max_cells+1):
        m[:] = 0
        m[ img == i ] = 1
        S = m.sum()
        if S == 0:
            continue
        x = np.dot(m, vx).sum()/S
        y = np.dot(m.transpose(), vy).sum()/S
        dct[i] = (x, y)
        
    return dct

dcls = [
[2,63,165],[125,135,185],
# [190,193,212],
[214,188,192],[187,119,132],[142,6,59],
[74,111,227],[133,149,225],[181,187,227],[230,175,185],[224,123,145],[211,63,106],
[17,198,56],[141,213,147],[198,222,199],[234,211,198],[240,185,141],[239,151,8],
[15,207,192],[156,222,214],[213,234,231],[243,225,235],[246,196,225],[247,156,212]
]

dcls = [[255,255,255]] + dcls*10

dcls = np.array(dcls)/255.0

def blend_to_white(data):
    a = data[:,:,3]/255.
    a = np.array([a,a,a]).transpose(1,2,0)
    w = np.ones( a.shape )*255.
    return np.uint8( np.round(w*(1-a)+ a*data[:,:,:3] ) )

def two_scales(ax1, time, data1, data2, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp')

    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('sin')
    return ax1, ax2

huecolors =  np.array([[ 1.        ,  0.00784302,  0.        ],
                       [ 1.        ,  0.46655273,  0.        ],
                       [ 1.        ,  0.9921875 ,  0.        ],
                       [ 0.3371582 ,  1.        ,  0.        ],
                       [ 0.        ,  1.        ,  0.50976562],
                       [ 0.        ,  0.92919922,  1.        ],
                       [ 0.        ,  0.40795898,  1.        ],
                       [ 0.        ,  0.13330078,  1.        ],
                       [ 0.48242188,  0.        ,  1.        ],
                       [ 1.        ,  0.        ,  0.82763672]])
huecolors = np.tile(huecolors, (20,1) )
colorlist = np.concatenate( ( np.array([[1,1,1]]), huecolors) )
cmap = matplotlib.colors.ListedColormap( colorlist )

def put_markers_on_image(markers, img):
    img3 = np.tile(np.expand_dims(np.float16(img), axis=2), (1,1,3)) / np.max(img)
    img3 = (1+img3)/2
    return cmap(markers)[:,:,:3] * img3