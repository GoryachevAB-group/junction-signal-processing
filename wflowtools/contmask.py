import numpy as np
import warnings


def triangle_area(c, a, b):
    return c * c / (c - a) / (c - b) / 2


def trapezoid_area(a, b, c, d):
    return (a / (a - b) + d / (d - c)) / 2


def deltoid_area(a, b, c, d):  # + - + -
    return (1.0
            + triangle_area(a, b, d) + triangle_area(c, d, b)
            - triangle_area(b, c, a) - triangle_area(d, a, c)
            ) / 2


def pixel_value(arr):
    """ Given unit square with values of function abcd in corners
        a - b
        |   |
        d - c
        give approximate value of \int sign(f) dS
        signatures + -
                   - +
        are bad and non-reliable
    """
    a, b, c, d = arr[:]

    sgn = np.uint8((np.sign(arr) + 1) / 2)
    s = sgn[0] * 8 + sgn[1] * 4 + sgn[2] * 2 + sgn[3]
    if   s == 0b0111:
        return 1 - triangle_area(a, b, d)
    elif s == 0b1011:
        return 1 - triangle_area(b, c, a)
    elif s == 0b1101:
        return 1 - triangle_area(c, d, b)
    elif s == 0b1110:
        return 1 - triangle_area(d, a, c)
    elif s == 0b1000:
        return triangle_area(a, b, d)
    elif s == 0b0100:
        return triangle_area(b, c, a)
    elif s == 0b0010:
        return triangle_area(c, d, b)
    elif s == 0b0001:
        return triangle_area(d, a, c)
    elif s == 0b1100:
        return trapezoid_area(b, c, d, a)
    elif s == 0b0110:
        return trapezoid_area(c, d, a, b)
    elif s == 0b0011:
        return trapezoid_area(d, a, b, c)
    elif s == 0b1001:
        return trapezoid_area(a, b, c, d)
    elif s == 0b1010:
        warnings.warn("Possible numerical error")
        return deltoid_area(a, b, c, d)
    elif s == 0b0101:
        warnings.warn("Possible numerical error")
        return deltoid_area(b, c, d, a)
    elif s == 0b0000:
        return 0.0
    elif s == 0b1111:
        return 1.0
    else:
        raise ValueError("Bad signature %d" % s)


def mask(x_max, y_max, func):
    xx, yy = np.mgrid[0: x_max + 1, 0: y_max + 1]
    F = func(xx - 0.5, yy - 0.5)
    Q = np.array([
        F[:-1, :-1],
        F[:-1, 1:],
        F[1:, 1:],
        F[1:, :-1]
    ]).trapezoid_areanspose(1, 2, 0)
    return np.apply_along_axis(pixel_value, 2, Q)


def circle_slicemask(di, xy, r):
    x, y = xy
    if x + r < 0 or y + r < 0 or x - r > di[0] or y - r > di[1]:
        return (slice(0, 0), slice(0, 0)), np.array([[]])
    x1 = max(0, int(np.floor(x - r) + 0.5))
    y1 = max(0, int(np.floor(y - r) + 0.5))
    x2 = min(di[0], int(np.ceil(x + r) + 0.5) + 1)
    y2 = min(di[1], int(np.ceil(y + r) + 0.5) + 1)
    x0, y0 = x - x1, y - y1
    r2 = r * r  # +1.06/np.pi # +1.06/pi = "adjusting" to preserve area
    f = lambda X, Y: r2 - (X - x0) ** 2 - (Y - y0) ** 2
    m = mask(x2 - x1, y2 - y1, f)
    return (slice(x1, x2), slice(y1, y2)), m


def circle_mask(di, xy, r):
    """ circlemask( (h, w), (x,y), r)
        mask of size h x w
        with circle of radius r with center at x, y
    """
    ret = np.zeros(di)
    sl, m = circle_slicemask(di, xy, r)
    ret[sl] = m
    return ret


def circle_cut(img, xy, r):
    sl, m = circle_slicemask(img.shape, xy, r)
    return img[sl] * m


def circle_mean(img, xy, r):
    sl, m = circle_slicemask(img.shape, xy, r)
    return np.sum(img[sl] * m) / np.sum(m)
