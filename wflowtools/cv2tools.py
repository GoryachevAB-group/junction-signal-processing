import numpy as np
import cv2
import mahotas as mh


def otsu(img, k=1.0):
    T_otsu = mh.otsu(img)
    return img > k * T_otsu


def adaptive_thr(img, box_size=11, C=2):
    return cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, box_size, C)


def get_markers(eimg):
    return mh.label(np.array(eimg, np.bool))


def get_network_component(eimg):
    img, nr_markers = get_markers(eimg)
    n = nr_markers + 1
    size = 0
    imsize = img.shape[0] * img.shape[1]
    imask = -1
    for i in range(1, n):
        mask = np.zeros(img.shape, np.bool)
        mask[img == i] = True
        (xs, ys) = np.nonzero(mask)
        if xs:
            nsize = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
            if imsize == nsize:
                return mask
            elif nsize > size:
                size = nsize
                imask = i
    mask = np.zeros(img.shape, np.bool)
    mask[img == imask] = True
    return mask


def remove_small_components(markers, nr_markers, area=100):
    max_cells = 0

    for i in range(1, nr_markers + 1):
        if np.sum(markers == i) < area:
            markers[markers == i] = 0
        else:
            max_cells += 1
            markers[markers == i] = max_cells
    return max_cells


def permute_watershed(w):
    n = np.max(w) + 1
    ids = np.random.permutation(-1 - np.arange(n))
    for i, j in enumerate(ids):
        w[w == i] = j
    w[:] = -w[:] - 1
    return w


k5x5 = kernel = np.ones((5, 5), np.uint16)

cwatershed = mh.cwatershed


def get_watershed(dn):
    # otsu1 = otsu(dn, 1)
    thr = adaptive_thr(dn, 63, -10)
    cmp = np.array(get_network_component(thr), np.uint8)
    cmp = cv2.morphologyEx(cmp, cv2.MORPH_CLOSE, kernel=k5x5, iterations=2)

    markers, n_markers = get_markers(1 - cmp)

    n_cells = remove_small_components(markers, n_markers)
    W = mh.cwatershed(dn, markers)

    return W - 1, n_cells + 1


def get_watershed1(dn):
    # otsu1 = otsu(dn, 1)
    thr = adaptive_thr(dn, 93, -10)

    dt = cv2.distanceTransform(1 - thr, cv2.DIST_L2, 5)
    # ret, th2 = cv2.threshold(dt, 3, 1,0)
    th2 = adaptive_thr(np.array(dt, np.uint8), 63, 1)
    opn = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel=k5x5, iterations=1)

    markers, n_markers = get_markers(opn)

    # n_cells = remove_small_components(markers, n_markers)
    n_cells = n_markers

    W = mh.cwatershed(dn, markers)

    return W - 1, n_cells + 1


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def read_markers(mrks):
    m = np.uint8(mrks.sum(axis=2) / (3 * 255))
    markers, _ = get_markers(m)
    return markers


kernel3 = np.ones((3, 3))
mask_gauss_size = 31


def smooth_labels(wsh, force_erode=None):
    if force_erode is None:
        force_erode = dict()
    labels = np.zeros_like(wsh)
    mask = np.zeros_like(wsh, dtype=np.float32)
    uniq = frozenset(np.unique(wsh)) - frozenset({0})
    for i in uniq:
        mask[:] = 0
        mask[wsh == i] = 1
        if mask.sum() == 0:
            raise ValueError('No cell with index %d' % i)
        if i in force_erode:
            ker_size = 2 * force_erode[i] + 1
            ker = np.ones((ker_size, ker_size))
            p_mask = np.pad(mask, 1, "constant")
            p_mask = cv2.morphologyEx(p_mask, cv2.MORPH_ERODE, kernel=ker)
            mask = p_mask[1:-1, 1:-1]
            labels[mask > 0.5] = i
            continue
        maskg = cv2.GaussianBlur(mask, (mask_gauss_size, mask_gauss_size), 0)
        if np.count_nonzero(maskg > 0.95) > 0:
            labels[maskg > 0.95] = i
        else:
            maske = cv2.morphologyEx(
                mask, cv2.MORPH_ERODE, kernel=kernel3, iterations=2)
            if np.count_nonzero(maske > 0.5) > 0:
                labels[maske > 0.5] = i
            else:
                labels[mask > 0.5] = i
    return labels


def mask_from_comp(comp):
    mask = np.maximum(comp[:, :, 0], np.maximum(comp[:, :, 1], comp[:, :, 2]))
    return np.uint8(mask > 0.5)


# xx, yy = np.ogrid[-7:8, -7:8]
# kernel13 = np.uint8(xx * xx + yy * yy - 7.5**2 <= 0)


def add_new_cells(comp, labels, n_cells, debug=False):
    """
    @brief finds red pixels in `comp` and puts a new label there
    """
    if debug:
        print(comp)
    flt = np.int32(comp)
    red = flt[:, :, 0] - flt[:, :, 1] - flt[:, :, 2]
    mask = (red == 255)
    xy = np.where(mask)
    if xy[0].size == 0:
        return None, n_cells
    H, W = labels.shape
    xx, yy = np.ogrid[:H, :W]
    r2 = 10**2
    for x, y in zip(*xy):
        labels[(xx - x)**2 + (yy - y)**2 < r2] = 0  # erase space in radius r
        n_cells += 1
        labels[x, y] = n_cells
    return mask, n_cells


def add_divided_cells(comp, labels, n_cells, debug=False):
    if debug:
        print(comp)
    flt = np.int16(comp)
    green = flt[:, :, 1] - flt[:, :, 0] - flt[:, :, 2]
    mask = (green == 255)
    if mask.sum() == 0:
        return None, n_cells

    green_labels, nr_labels = get_markers(mask.astype('uint8'))

    # HACK -- change later
    oldcells = set(np.unique(mask * labels).tolist())
    oldcells -= frozenset({0})
    for cell in oldcells:  # delete oldcells
        labels[labels == cell] = 0
    H, W = labels.shape
    xx, yy = np.ogrid[:H, :W]
    for i in range(1, nr_labels + 1):
        n_cells += 1
        labels[green_labels == i] = n_cells
    return mask, n_cells

    ### old code vvv
    # green_labels, nr_labels = get_markers(mask)
    # print(green_labels)

    # floodfillmask = np.zeros((labels.shape[0] + 2, labels.shape[1] + 2),
    #                          np.uint8)
    # for i in range(1, nr_labels + 1):
    #     x, y = np.unravel_index(np.argmax(green_labels == i), labels.shape)
    #     if labels[x, y] > 0:
    #         cv2.floodFill(labels.astype('uint8'), floodfillmask, (y, x), 0)
    #     n_cells += 1
    #     labels[green_labels == i] = n_cells
    # return np.where(green_labels > 0.5), n_cells
