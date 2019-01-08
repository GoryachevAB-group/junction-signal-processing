import numpy as np
import imageio


def logclip(arr, amin=1, amax=4):
    out = np.log2(arr)
    lmin, lmax = np.log2([amin, amax])
    out = (out - lmin) / (lmax - lmin)
    out = np.clip(out, a_min=0, a_max=1)
    return np.uint8(out * 255.0)


def write_logpack(fn, data):
    scale = max(np.log2(np.max(data)), -np.log2(np.min(data)))
    scale = int(np.ceil(100 * scale)) / 100.0
    idata = np.uint16((1 + np.log2(data) / scale) * 2**15)

    imageio.volwrite(fn + '_lp' + str(scale) + '.tif', idata)
    return scale


def read_logpack(fn):
    data = -1.0 + np.array(imageio.volread(fn), dtype='float') / 2**15
    scale = float(".".join(fn.split('.')[:-1]).split('_lp')[-1])
    return np.exp2(data * scale)


# write_logpack(main_imgfile[:-4]+'_fdata' , flepith.fdata3d)