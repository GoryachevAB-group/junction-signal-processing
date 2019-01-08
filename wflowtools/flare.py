# import itertools
from collections import namedtuple, defaultdict, OrderedDict, Counter
from warnings import warn

import numpy as np
import pandas as pd  # pylint: disable=import-error
# import shapely as shp  # pylint: disable=import-error
from shapely.geometry import Point, Polygon, LineString  # pylint: disable=import-error
from shapely.ops import cascaded_union
import scipy.ndimage as ndi
from scipy import stats, optimize, interpolate
from objdict import ObjDict
from . import trace_image as TI
from . import activegraph as AG
from . import geometry as GM


def has_field(self, field):
    return field in self and self[field] is not None


ObjDict.has_field = has_field


def expected_position(m):
    X = np.dot(np.arange(m.shape[0], dtype=np.float16), m.sum(axis=1))
    Y = np.dot(np.arange(m.shape[1], dtype=np.float16), m.sum(axis=0))
    return np.array([X, Y]) / m.sum()


EdgeWeights = namedtuple('EdgeWeights', 'total summary data')
EdgeData = namedtuple('EdgeData', 'ids pts is_clipped')


def find_max_slope(data, d=2):
    n = len(data)
    x = np.arange(-d, d + 1)
    maxslope = -np.inf
    for i in range(d, n - d):
        slope = stats.linregress(x, data[i - d:i + d + 1])[0]
        if slope > maxslope:
            maxslope = slope
            pos = i
    return pos, maxslope


def _stepslope(x, x0, c, slope):
    if x < x0:
        return c
    else:
        return c + (x - x0) * slope


stepslope = np.vectorize(_stepslope)


class Flare:
    def __init__(self):
        self.data = ObjDict()
        self.fsdf = None
        self.kmdf = None
        self.index = None
        self.blob = np.array([], np.uint8).reshape(0, 0, 0)
        # self.blob_nt, self.blob_nx, self.blob_ny = self.blob.shape
        self.blob_slice = [slice(0, 0)] * 3
        self.blob_rect = np.zeros((3, 2), dtype=np.uint16)
        self.seed = None

    def set_index(self, i):
        self.index = i

    def set_name(self, name):
        self.name = name

    def set_directory(self, directory):
        self.directory = directory

    def add_blob(self, blob, blob_slice, thr=None):
        self.blob = np.array(blob, np.uint8)
        # self.blob_nt, self.blob_nx, self.blob_ny = blob.shape
        self.blob_slice = blob_slice
        self.blob_rect = np.array([[sl.start, sl.stop]
                                   for sl in blob_slice], np.uint16)
        self.threshold = thr
        if self.seed is None:
            dt = ndi.distance_transform_bf(blob)
            self.seed = np.unravel_index(dt.argmax(), blob.shape)
        self.create_geometry()
        return self

    def floodfill(self, data3d, seed, thr):
        T, H, W = data3d.shape

        def in_data(t, x, y):
            return t >= 0 and t < T and x >= 0 and x < H and y >= 0 and y < W

        def gen(t, x, y):
            return [(t - 1, x, y), (t + 1, x, y), (t, x - 1, y), (t, x + 1, y),
                    (t, x, y - 1), (t, x, y + 1)]

        pts = set()
        border = set({seed})
        while border:
            pt = border.pop()
            if in_data(*pt) and data3d[pt] >= thr:
                pts.add(pt)
                for p in gen(*pt):
                    if p not in pts and p not in border:
                        border.add(p)
        if not pts:
            raise ValueError("Bad seed")
        pts = np.array([list(p) for p in pts])
        t1, t2 = min(pts[:, 0]), max(pts[:, 0]) + 1
        x1, x2 = min(pts[:, 1]), max(pts[:, 1]) + 1
        y1, y2 = min(pts[:, 2]), max(pts[:, 2]) + 1
        blob = np.zeros((t2 - t1, x2 - x1, y2 - y1), dtype='uint8')
        wh = tuple((pts - np.array([t1, x1, y1])).transpose())
        blob[wh] = 1
        self.seed = seed
        self.add_blob(blob, (slice(t1, t2), slice(x1, x2), slice(y1, y2)), thr)
        return self

    def create_geometry(self, offset_distance=2.0):
        self.geo = []
        shift = self.blob_rect[1:, 0] - np.array([1.5, 1.5])

        def subpoly(blob):
            labels, nr_labels = ndi.label(blob)
            for i in range(1, nr_labels + 1):
                b = np.uint8(labels == i)
                b = np.pad(b, pad_width=1, mode='constant')
                rings = [shift + r for r in TI.trace_bw_image(b)]
                rings.sort(key=lambda a: -len(a))
                if rings:
                    # Loose criteria for inner/outer selection
                    # but it should work 99.9% of time
                    polygon = Polygon(rings[0], rings[1:])
                    if not polygon.is_valid:
                        coords = [(x, y) for x, y in rings[0]]
                        for pt, n in Counter(coords[:-1]).items():
                            if n == 2:
                                i1 = coords.index(pt)
                                i2 = coords.index(pt, i1 + 1)
                                rings[0][i1 + 1:i2] = rings[0][i2 - 1:i1:-1]
                        polygon = Polygon(rings[0], rings[1:])
                    yield polygon
                else:
                    raise ValueError('Cant process polygon with' +
                                     'no boundaries')

        for blob2d in self.blob:
            poly = cascaded_union(list(subpoly(blob2d)))
            poly = poly.buffer(offset_distance).simplify(0.2)
            self.geo.append(poly)

    def calculate_intersections(self, graphs):
        summary = defaultdict(float)
        data = []
        for ag, poly in zip(graphs[self.blob_slice[0]], self.geo):
            inside_ptids = set(i for i, p in enumerate(ag.pts)
                               if poly.contains(Point(*p)))
            edge_ids = set(i for i, e in enumerate(ag.edges)
                           if set(e) & inside_ptids)
            data.append(dict())
            for ei in edge_ids:
                label = tuple(ag.edge_labels[ei])
                edge = LineString(ag.pts[ag.edges[ei]])
                common_length = poly.intersection(edge).length
                data[-1][label] = common_length
                summary[label] += common_length
        summary = OrderedDict(sorted(summary.items(), key=lambda t: -t[1]))
        total = sum(summary.values())
        self.edge_weights = EdgeWeights(total, summary, data)

    def assign_category(self, graphs, max_weight_thr=0.9, wings=(100, 100)):
        before_frames, after_frames = wings
        if 'category' in self.data and self.data.category:
            return

        self.data.type = 'unknown'
        label = None

        summary = self.edge_weights.summary
        if not summary:
            category = '--orphan'
        elif len(summary) == 1:
            category = '+++'
            label = next(iter(summary))
        else:
            max_weight = next(iter(summary.values())) / self.edge_weights.total
            if max_weight >= max_weight_thr:
                category = '+++'
                label = next(iter(summary))
            else:
                category = '--multi'
        if self.blob.sum() <= 5:
            category = '--speckle'
        if self.blob_rect[0, 0] < before_frames:
            category = '--near beginning'
        elif self.blob_rect[0, 1] >= len(graphs) - after_frames:
            category = '--near end'
        if category == '+++':
            t1 = self.blob_rect[0, 0] - before_frames
            t2 = self.blob_rect[0, 1] + after_frames
            data = []
            for ag in graphs[t1:t2]:
                is_p = ag.is_peripheral(label)
                if is_p is None:
                    category = '--bad-edge'
                    break
                elif is_p:
                    data.append(True)
                else:
                    data.append(False)
            else:
                if not data:
                    msg = 'Cant find label %s in time %d-%d' % (label, t1, t2)
                    raise ValueError(msg)
                if not all(data):
                    category = '--peripheral'

        self.data.category = category
        self.data.edge_label = label
        if category == '+++':
            self.data.type = 'good'

    def add_edge_geometry(self,
                          graphs,
                          scale: tuple,
                          wings=(100, 100),
                          select_func=None):
        backward, forward = wings
        tscale, lscale = scale
        T = len(graphs)
        t1 = max(0, self.blob_rect[0, 0] - backward)
        t2 = min(T, self.blob_rect[0, 1] + forward)
        tslice = slice(t1, t2)
        self.data.trange = (t1, t2)
        self.data.tslice = tslice

        if 'edge_label' not in self.data:
            raise ValueError('No edge associated for flare %d' % self.index)

        if isinstance(self.data.edge_label, tuple):
            edge_labels = [self.data.edge_label]
        else:
            edge_labels = self.data.edge_label

        edge_data_dict = OrderedDict()
        for t in range(t1, t2):
            ids, pts, is_clipped = [], [], False
            for edge in edge_labels:
                edge_data = _get_edge_data(graphs[t], edge)
                if edge_data:
                    ids += list(edge_data.ids)[:-1]
                    pts += list(edge_data.pts)[:-1]
                    last_ids = edge_data.ids[-1]
                    last_pts = edge_data.pts[-1]
                    is_clipped = is_clipped or edge_data.is_clipped
                else:
                    msg = "Flare %d: no edge %s at time %d"
                    warn(msg % (self.index, edge, t))
            if ids:
                ids.append(last_ids)
                pts.append(last_pts)
                edge_data_dict[t] = EdgeData(
                    np.array(ids), np.array(pts), is_clipped)

        t_list = np.arange(t1, t2)
        self.fsdf = fsdf = pd.DataFrame(
            {
                'time': t_list * tscale,
            }, index=pd.Index(t_list, name='t'))
        self.fsdf['clipped_junction'] = np.nan

        tuples = []
        ds_ptids = []
        ds_pts = []
        ds_ss = []
        for t, ed in edge_data_dict.items():
            fsdf.at[t, 'clipped_junction'] = ed.is_clipped
            tuples += [(t, j) for j in range(len(ed.ids))]
            ds_ptids += list(ed.ids)
            ds_pts += list(ed.pts)
            ds_ss += GM.length_along(np.array(ed.pts)).tolist()
            assert len(tuples) == len(ds_ptids) == len(ds_pts)
        index = pd.MultiIndex.from_tuples(tuples, names=['t', 'ptn'])
        ds_pts = np.array(ds_pts).transpose()
        self.kmdf = kmdf = pd.DataFrame(
            {
                'ptids': ds_ptids,
                'x': ds_pts[0],
                'y': ds_pts[1],
                'ss': ds_ss
            },
            index=index)

        fsdf['jxn_length'] = pd.Series(kmdf.groupby(level=0)['ss'].max())
        fsdf['jxn_length_um'] = pd.Series(fsdf['jxn_length'] * lscale)

        kmdf['has_flare'] = False
        t_list = self.kmdf.index.get_level_values(0).unique()
        for t, poly in zip(range(*self.blob_rect[0]), self.geo):
            if t not in t_list:
                continue
            pts = kmdf.loc[t, ['x', 'y']].values
            kmdf.loc[t, 'has_flare'] = [
                poly.contains(Point(x, y)) for x, y in pts
            ]
        has_flare = kmdf['has_flare']
        fsdf['flare_s_min'] = kmdf.loc[has_flare].groupby('t')['ss'].min()
        fsdf['flare_s_max'] = kmdf.loc[has_flare].groupby('t')['ss'].max()

    def smart_align(self, T0=None, span=None):
        df = self.fsdf
        km = self.kmdf
        if T0 is None:
            fts = km[km.has_flare].index.get_level_values(0).unique()
            sub_has_flare = df.loc[fts]
            time1, time2 = min(sub_has_flare.time), max(sub_has_flare.time)

            sub_slope = df[df.time.between(time1 - 100, time2)]
            pos, slope = sub_slope.index.min() + find_max_slope(
                sub_slope.rho_norm)

            time1, time2 = df.loc[pos - 2].time - 100, df.loc[pos + 3].time + 1
            sub_fit1 = df[df.time.between(time1, time2)]
            tm = sub_fit1.time

            popt, _ = optimize.curve_fit(stepslope, tm, sub_fit1.rho_norm,
                                         (df.loc[pos - 2].time,
                                          df.loc[pos - 2].rho_norm, slope))
            T0 = popt[0]
            self.data.fit1_popt = popt
            self.data.fit1_times = (time1, time2)
        else:
            time1 = T0 - 100

        df.time_slope_align = df.time - T0

        jlf = interpolate.interp1d(df.time, df.jxn_length_um)
        df.jxn_length_diff_um = df.jxn_length_um - jlf(T0)

        # norming to beginning
        if span is None:
            selection = df.time < time1
        else:
            selection = (df.time >= span[0]) & (df.time <= span[1])
        sub_norm2 = df[selection]
        for col in df.columns:
            if col[-5:] == '_norm' or col[-6:] == '_total':
                sub = sub_norm2[sub_norm2[col] > 0]
                med = sub[col].median()
                df[col + '2'] = df[col] / med


def _get_edge_data(ag: AG.ActiveGraph, edge_label,
                   select_func=None) -> EdgeData:
    es = ag.edges_dct[edge_label]
    if not es:
        return None
    if len(es) > 1:
        if select_func:
            selection = select_func(ag, es)
            idx = es[selection]
        else:
            msg = "Edge (%d-%d) has %d components and there is no selection function"
            raise ValueError(msg % (*edge_label, len(es)))
    else:
        idx = es[0]
    # edge has 1-vertex as end
    pts = ag.pts[idx]
    if idx[0] in ag.ptids[0] or idx[-1] in ag.ptids[0]:
        return EdgeData(idx, pts, True)
    return EdgeData(idx, pts, False)


MAX_WHITE = 3.5
MIN_WHITE = 1.0
__forward_factor = 256.0 / (MAX_WHITE - MIN_WHITE)
__backward_factor = 1.0 / __forward_factor
float2byte = lambda f: (f - MIN_WHITE) * __forward_factor
byte2float = lambda b: MIN_WHITE + b * __backward_factor


def thresholded_flares(blur3d, threshold):
    """
    @brief threshold `blur3d` image and return segments
    """
    print('Labeling...', end='')
    blobs = np.uint8(blur3d >= threshold)
    labels3d, nr_flares = ndi.label(blobs, structure=np.ones((3, 3, 3)))
    flares_slices = ndi.find_objects(labels3d)
    print(' done. %d flares ' % nr_flares)

    print('Initialising...', end='')
    flares = []
    for i, slice3d in enumerate(flares_slices):
        mask = np.uint8(labels3d[slice3d] == i + 1)
        flare = Flare()
        ### Finding point with maximum blur3d value
        offset = np.array([s.start for s in slice3d])
        pts = np.array(np.nonzero(mask)).transpose() + offset
        arg = blur3d[tuple(pts.transpose())].argmax()
        flare.seed = pts[arg]
        ###
        flare.add_blob(mask, slice3d, threshold)
        flare.set_index(i + 1)
        flare.data.type = "unknown"
        flares.append(flare)
    print(' done')
    return flares, labels3d.astype('uint8')


def flares_info(flares):
    columns = """frame x y thr
                 category edge edge_num
                 edge_distribution""".split()
    data = pd.DataFrame(columns=columns)
    for flare in flares:
        label = dist = ''
        size = np.nan
        if flare.data.has_field('edge_label'):
            label = "%2d-%2d" % flare.data.edge_label

        if 'edge_weights' in flare.__dict__:
            summary = flare.edge_weights.summary
            size = len(summary)
            S = flare.edge_weights.total
            if len(summary) > 1:
                tuples = [(*l, int(100 * v / S)) for l, v in summary.items()]
                dist = ", ".join(["%2d-%2d(%2d%%)" % t for t in tuples])

        data.loc[flare.index] = [
            flare.seed[0] + 1, flare.seed[2], flare.seed[1],
            float2byte(flare.threshold), flare.data.category, label, size, dist
        ]
    return data


def read_flares(filename, blur3d, only_good=False):
    data = pd.read_excel(filename)
    flares = []
    for i, row in data.iterrows():
        flare = Flare()
        flare.data.type = 'good' if row.category[0] == '+' else 'unknown'
        if only_good and flare.data.type != 'good':
            continue
        flare.data.category = row.category
        flare.index = i

        seed = (row.frame - 1, row.y, row.x)
        thr = byte2float(row.thr)
        try:
            flare.floodfill(blur3d, seed, thr)
        except ValueError as e:
            print(str(e) + ' at row %d' % i)
        else:
            if flare.data.type == 'good':
                flare.data.edge_label = tuple(
                    int(s) for s in row.edge.split('-'))
            flares.append(flare)
    return flares


def find_shard_candidates(flares, labels3d, dt=3, dl=3):
    T, H, W = labels3d.shape
    pairs = set()
    add_pair = lambda a, b: pairs.add((min(a, b), max(a, b)))

    for flare in flares:
        tr, xr, yr = flare.blob_rect
        tr = np.clip(tr + np.array([-dt, dt]), a_min=0, a_max=T)
        xr = np.clip(xr + np.array([-dl, dl]), a_min=0, a_max=H)
        yr = np.clip(yr + np.array([-dl, dl]), a_min=0, a_max=W)
        sub = labels3d[tr[0]:tr[1], xr[0]:xr[1], yr[0]:yr[1]]
        s = set(np.unique(sub).tolist()) - set({0, flare.index})
        for e in s:
            add_pair(flare.index, e)

    def border_pts(blob):
        arr = np.pad(blob, pad_width=1, mode='constant')
        out = ((blob > arr[:-2, 1:-1, 1:-1]) + (blob > arr[2:, 1:-1, 1:-1]) +
               (blob > arr[1:-1, :-2, 1:-1]) + (blob > arr[1:-1, 2:, 1:-1]) +
               (blob > arr[1:-1, 1:-1, :-2]) + (blob > arr[1:-1, 1:-1, 2:]))
        return np.uint8(out)

    def closest(f1, f2):
        dist2 = lambda pt: 2 * pt[0]**2 + pt[1]**2 + pt[2]**2
        b1 = np.nonzero(border_pts(f1.blob))
        b2 = np.nonzero(border_pts(f2.blob))
        b1 = np.array(b1).transpose()
        b2 = np.array(b2).transpose()
        dmax = 1e99
        curpair = ()
        offset = 0.0 + f1.blob_rect[:, 0] - f2.blob_rect[:, 0]
        for pt1 in b1:
            for pt2 in b2:
                d = dist2(offset + pt1 - pt2)
                if d < dmax:
                    dmax = d
                    curpair = (f1.blob_rect[:, 0] + pt1,
                               f2.blob_rect[:, 0] + pt2)
        return np.sqrt(dmax), curpair

    fdict = dict((flare.index, flare) for flare in flares)
    pairs = sorted(list(pairs), key=lambda t: t[0])
    valid_categories = '--multi +++'.split()
    for i, j in pairs:
        cat1 = fdict[i].data.category
        cat2 = fdict[j].data.category
        if cat1 not in valid_categories and cat2 not in valid_categories:
            continue
        if cat1[:6] == '--near' and cat2[:6] == '--near':
            continue
        d, (p1, p2) = closest(fdict[i], fdict[j])
        dif = p1 - p2
        print("%3d - %3d\td = %4.1f (%4.1f v %4.1f) " %
              (i, j, d, abs(dif[0]), np.linalg.norm(dif[1:])))
    print('No more flares')
