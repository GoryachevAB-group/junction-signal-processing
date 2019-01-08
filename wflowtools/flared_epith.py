import os
from subprocess import call, Popen, PIPE
from itertools import repeat
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.signal as signal
import scipy.interpolate as intpl
import scipy.stats as stats
import imageio as iio

from shapely.geometry import Point, LineString
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

from . import geometry as GM
from . import flare as Flare

disk1x3x3 = np.array(
    [[
        [15384132, 27409613, 15384132],
        [27409613, 28206721, 27409613],
        [15384132, 27409613, 15384132],
    ]],
    dtype='float')
disk1x3x3 /= disk1x3x3.sum()

disk1x5x5 = np.array(
    [[
        [3594616, 20206818, 25825225, 20206818, 3594616],
        [20206818, 26265625, 26265625, 26265625, 20206818],
        [25825225, 26265625, 26265625, 26265625, 25825225],
        [20206818, 26265625, 26265625, 26265625, 20206818],
        [3594616, 20206818, 25825225, 20206818, 3594616],
    ]],
    dtype='float')
disk1x5x5 /= disk1x5x5.sum()


def get_mode(arr, bins=30, cutoff_coefficient=0.7, median_lowpass=0.5):
    """
    @brief Finds mode of distribution
    @param arr      1d array-like with data
    @param bins     number of bins
    @param cutoff_coefficient 
    @param median_lowpass 
    @return value of mode
    """
    S = np.median(arr) * 2
    rng = np.linspace(0, S, bins + 1)
    y, _ = np.histogram(arr, rng)
    x = (rng[1:] + rng[:-1]) / 2
    i1 = np.where(x > median_lowpass * S)[0][0]
    mx = y[i1:].max()
    i = np.where(y == mx)[0][0]
    i1, i2 = i, i + 1
    while i1 > 0 and y[i1 - 1] > cutoff_coefficient * mx:
        i1 -= 1
    while i2 < y.size and y[i2] > cutoff_coefficient * mx:
        i2 += 1
    if i2 - i1 <= 3:
        return x[y.argmax()]
    # vv fitting parabola to find maximum position
    xx, yy = x[i1:i2], y[i1:i2]
    x0 = xx.size
    x1 = xx.sum()
    x2 = (xx**2).sum()
    x3 = (xx**3).sum()
    x4 = (xx**4).sum()
    yx0 = yy.sum()
    yx1 = (xx * yy).sum()
    yx2 = (yy * xx**2).sum()
    _, b, c = np.dot(
        np.linalg.inv(np.array([
            [x0, x1, x2],
            [x1, x2, x3],
            [x2, x3, x4],
        ])), np.array([yx0, yx1, yx2]))
    mode = -b / (2 * c)
    if mode < x[i1] or mode > x[i2 - 1]:
        return x[y.argmax()]
    return mode


class FlaredEpithelium:
    """
    @brief main processor of vectorized cells geometry and flares
    """

    def __init__(self, fdata3d, graphs, tscale=1, lscale=1,
                 min_flare_volume=5):
        self.fdata3d = fdata3d
        self.T, self.H, self.W = fdata3d.shape
        self.shape3d = (self.T, self.H, self.W)
        self.shape2d = (self.H, self.W)
        self.xx = np.arange(self.H)
        self.yy = np.arange(self.W)
        self.tt = np.arange(self.T)
        self.graphs = graphs
        self.tscale, self.lscale = tscale, lscale
        self.min_flare_volume = min_flare_volume
        self.is_normalized = False

    def _internal_circles_gen(self, radius=3):
        for graph in self.graphs:
            nodes = np.concatenate(graph.ptids[2:])
            yield ((x, y, radius) for x, y in graph.pts[nodes])

    def _external_edges_gen(self, width=3):
        for graph in self.graphs:
            endnodes = set(graph.ptids[0])
            yield ((graph.pts[edge].flatten().tolist(), width)
                   for edge in graph.edges if set(edge[[0, -1]]) & endnodes)

    def _all_edges_gen(self, width=3):
        for graph in self.graphs:
            yield ((graph.pts[edge].flatten().tolist(), width)
                   for edge in graph.edges)

    def create_mask(self, circles_gen, lines_gen):
        if circles_gen is None:
            circles_gen = repeat([])
        if lines_gen is None:
            lines_gen = repeat([])
        ones2d = np.ones((self.W, self.H), dtype='uint8')
        mask3d = np.zeros((self.T, self.W, self.H), dtype='uint8')

        t_gen = range(self.T)

        for (t, circles, lines) in zip(t_gen, circles_gen, lines_gen):
            img = Image.fromarray(ones2d)
            draw = ImageDraw.Draw(img)
            for (x, y, r) in circles:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
            for (pts, w) in lines:
                draw.line(pts, width=w, fill=0)

            mask3d[t, :] = np.array(img)
        return mask3d.transpose(0, 2, 1)

    def generate_pt_values(self, data3d):
        return np.array([
            intpl.RectBivariateSpline(self.xx, self.yy, im, kx=1, ky=1) \
            .ev(*graph.pts.transpose())
            for im, graph in zip(data3d, self.graphs)
        ])

    def filter_graph_data(self, data3d):
        out = list()
        for ag, data in zip(self.graphs, data3d):
            degree = dict()
            for i, ptids in enumerate(ag.ptids):
                degree.update((ptid, i + 1) for ptid in ptids)
            fdata = np.zeros_like(data)
            for edge in ag.edges:
                if edge.size < 11:
                    x = np.arange(edge.size)
                    pars = stats.linregress(x, data[edge])
                    vals = pars[1] + pars[0] * x
                else:
                    vals = signal.savgol_filter(data[edge], 11, 2)
                fdata[edge[1:-1]] = vals[1:-1]
                fdata[edge[0]] += vals[0] / degree[edge[0]]
                fdata[edge[-1]] += vals[-1] / degree[edge[-1]]
            out.append(fdata)
        return np.array(out)

    def normalize(self,
                  mask3d,
                  approx_gridsize=50,
                  window=150,
                  linear_ts=frozenset()):
        if self.is_normalized:
            raise ValueError('Data is already normalized')
        window_radius = int(window / 2)
        blur = ndi.convolve(self.fdata3d.astype('float32'), weights=disk1x3x3)
        blur = blur.astype('float16')
        img3d = blur * mask3d
        T, H, W = img3d.shape
        xx, yy = np.mgrid[0:H, 0:W]
        bg = np.zeros_like(img3d)
        get_n = lambda n: 1 + np.round((n - 1) / approx_gridsize)
        linspace2 = lambda n: np.linspace(0, n - 1, get_n(n)).astype('uint16')
        gridxx = linspace2(H)
        gridyy = linspace2(W)
        bg_grid = np.zeros((gridxx.size, gridyy.size))
        for t, img in tqdm(enumerate(img3d), total=T):
            empty_ij = []
            for i, x in enumerate(gridxx):
                for j, y in enumerate(gridyy):
                    arr = img[max(0, x - window_radius):x + window_radius + 1,
                              max(0, y - window_radius):
                              y + window_radius + 1].flatten()
                    arr = arr[arr > 0]
                    if arr.size > 0:
                        bg_grid[i, j] = get_mode(arr)
                    else:
                        empty_ij.append((i, j))
            for i, j in empty_ij:
                neighbors = []
                if i > 0 and j > 0:
                    neighbors.append(bg_grid[i - 1, j - 1])
                if i > 0 and j < gridyy.size - 1:
                    neighbors.append(bg_grid[i - 1, j + 1])
                if i < gridxx.size - 1 and j > 0:
                    neighbors.append(bg_grid[i + 1, j - 1])
                if i < gridxx.size - 1 and j < gridyy.size - 1:
                    neighbors.append(bg_grid[i + 1, j + 1])
                neighbors = np.array(neighbors)
                bg_grid[i, j] = neighbors[neighbors > 0].mean()
            K = 1 if t in linear_ts else 3
            bg[t] = intpl.RectBivariateSpline(
                gridxx, gridyy, bg_grid, kx=K, ky=K).ev(xx, yy).reshape(H, W)
        self.fdata3d = self.fdata3d.astype('float16')
        self.fdata3d /= bg
        self.is_normalized = True
        return bg

    def test_threshold(self, frame=1, thr=2.0, kernel=(1, 2, 2)):
        sub = self.fdata3d[frame - kernel[0]:frame + kernel[0] + 1]
        blur = ndi.gaussian_filter(sub.astype('float32'), kernel)
        blobs = np.where(blur > thr, 1, 0).astype('uint8')
        return blobs[kernel[0]]

    def create_flares(self, thr=2.0, debug=False, kernel=(1, 2, 2)):
        if debug:
            print('Gauss filtering...', end='')
        blur = ndi.gaussian_filter(self.fdata3d.astype('float32'), kernel)
        if debug:
            print(' done')

        if debug:
            print('Labeling...', end='')
        blobs = np.where(blur > thr, 1, 0).astype('uint8')
        flares_labels, nr_flares = ndi.label(
            blobs, structure=np.ones((3, 3, 3)))
        flares_slices = ndi.find_objects(flares_labels)
        if debug:
            print(' done. %d flares ' % nr_flares)

        flares = []
        for i, sl in enumerate(flares_slices):
            mask = np.zeros_like(blobs[sl])
            mask[flares_labels[sl] == i + 1] = 1
            flare = Flare.Flare().add_blob(mask, sl, thr)
            flare.set_index(i + 1)
            flare.data.type = "unknown"
            flares.append(flare)

        if debug:
            print('Marking...', end='')
        flares_mask3d = self.create_mask(
            circles_gen=self._internal_circles_gen(),
            lines_gen=self._external_edges_gen(width=5))
        masked_labels = (1 - flares_mask3d) * flares_labels
        all_labels = list(np.arange(1, nr_flares + 1))
        bad_labels = set(np.unique(masked_labels)) - {0}
        good_labels = set(all_labels) - set(bad_labels)

        for flare in flares:
            if flare.index in good_labels:
                flare.data.type = 'good'
            elif flare.index in bad_labels:
                flare.data.type = 'bad'

            if flare.blob.sum() < self.min_flare_volume:
                flare.data.type = 'small'
                continue
            edge_labels = defaultdict(int)
            for graph, fl in zip(self.graphs[flare.blob_slice[0]], flare.geo):
                for i, edge in enumerate(graph.edges):
                    line = LineString(graph.pts[edge])
                    if fl.intersects(line):
                        el = tuple(graph.edge_labels[i])
                        edge_labels[el] += 1
            if edge_labels:
                el = max(edge_labels.items(), key=itemgetter(1))[0]
                flare.data.edge_label = el
            else:
                flare.data.type = 'inner'
        if debug:
            print(' done')
        return flares

    @staticmethod
    def get_edge_data(ag, edge_label, select_func=None):
        es = ag.edges_dct[edge_label]
        if not es:
            return np.zeros(0, dtype='uint8'), np.zeros((0, 2))
        if len(es) > 1:
            if select_func:
                selection = select_func(ag, es)
                idx = es[selection]
            else:
                raise ValueError("Edge (%d-%d) has %d components" %
                                 (*edge_label, len(es)))
        else:
            idx = es[0]
        # edge has 1-vertex as end
        if idx[0] in ag.ptids[0] or idx[-1] in ag.ptids[0]:
            return np.zeros(0, dtype='uint8'), np.zeros((0, 2))
        pts = ag.pts[idx]
        return idx, pts

    def init_flare(self,
                   flare,
                   backward=100,
                   forward=50,
                   select_func=None,
                   debug=False):
        t1 = max(0, flare.blob_rect[0, 0] - backward)
        t2 = min(self.T, flare.blob_rect[0, 1] + forward)
        tslice = slice(t1, t2)
        flare.data.trange = (t1, t2)
        flare.data.tslice = tslice

        t_list = self.tt[tslice]

        if 'edge_label' not in flare.data:
            print('No edge associated for flare %d' % flare.index)
            flare.fsdf = pd.DataFrame(
                {
                    'time': t_list * self.tscale,
                },
                index=pd.Index(t_list, name='t'))
            return

        if isinstance(flare.data.edge_label, tuple):
            data = []
            for graph in self.graphs[tslice]:
                ed = FlaredEpithelium.get_edge_data(
                    graph, flare.data.edge_label, select_func=select_func)
                data.append(list(ed))
            data = np.array(data)
        else:
            data = []
            for graph in self.graphs[tslice]:
                o1, o2 = [], []
                for edge in flare.data.edge_label:
                    i1, i2 = FlaredEpithelium.get_edge_data(graph, edge)
                    o1 += list(i1)
                    o2 += list(i2)
                data.append([np.array(o1), np.array(o2)])

        flare.fsdf = fsdf = pd.DataFrame(
            {
                'time': t_list * self.tscale,
            }, index=pd.Index(t_list, name='t'))

        tuples = []
        ds_ptids = []
        ds_pts = []
        ds_ss = []
        for t, (ptids, pts) in zip(t_list, data):
            if ptids is not None:
                tuples += [(t, j) for j in range(ptids.size)]
                ds_ptids += ptids.tolist()
                ds_pts += list(pts)
                ds_ss += GM.length_along(pts).tolist()
                assert len(tuples) == len(ds_ptids) == len(ds_pts)
        index = pd.MultiIndex.from_tuples(tuples, names=['t', 'ptn'])
        ds_pts = np.array(ds_pts).transpose()
        flare.kmdf = kmdf = pd.DataFrame(
            {
                'ptids': ds_ptids,
                'x': ds_pts[0],
                'y': ds_pts[1],
                'ss': ds_ss
            },
            index=index)

        fsdf['jxn_length'] = pd.Series(kmdf.groupby(level=0)['ss'].max())
        fsdf['jxn_length_um'] = pd.Series(fsdf['jxn_length'] * self.lscale)

        kmdf['has_flare'] = False
        blob_slice = flare.blob_slice[0]
        t_list = flare.kmdf.index.get_level_values(0).unique()

        for t, poly in zip(self.tt[blob_slice], flare.geo):
            if t not in t_list:
                continue
            pts = kmdf.loc[t, ['x', 'y']].values
            kmdf.loc[t, 'has_flare'] = [
                poly.contains(Point(x, y)) for x, y in pts
            ]
        has_flare = kmdf['has_flare']
        fsdf['flare_s_min'] = kmdf.loc[has_flare].groupby('t')['ss'].min()
        fsdf['flare_s_max'] = kmdf.loc[has_flare].groupby('t')['ss'].max()


class ChannelProcessor:
    def __init__(self,
                 img3d: np.ndarray,
                 bg_mask3d: np.ndarray,
                 flepith: FlaredEpithelium,
                 raw_kernel=disk1x3x3):
        """
            pt_sig 3px blur values
            pt_tnorm = ^ normed by mean on timestamp
        """
        self.flepith = flepith
        assert img3d.shape == self.flepith.shape3d

        self.img3d = ndi.convolve(img3d.astype('float32'), weights=raw_kernel)
        self.img3d = self.img3d.astype('float16')
        self.masked_img3d = self.img3d * bg_mask3d
        xyproj = self.img3d.mean(axis=(1, 2))
        self.img3d_tnorm = self.img3d / xyproj[:, None, None]

        self.pt_sig = self.flepith.generate_pt_values(self.img3d)
        self.pt_tnorm = self.flepith.generate_pt_values(self.img3d_tnorm)
        self.__bg_median_lowpass = 0
        self.T = self.flepith.T

    def get_reference(self, t, pt, r1=10, r2=70):
        if np.any(np.isnan(pt)):
            return np.nan
        pts = self.flepith.graphs[t].pts
        dist = ((pts - pt[None, :])**2).sum(axis=1)
        wh = (r1**2 < dist) & (dist < r2**2)
        arr = self.pt_sig[t][wh]
        return np.median(arr)

    def get_background(self, t, pt, window=150):
        if np.any(np.isnan(pt)):
            return np.nan
        d = int(window / 2)
        ix, iy = int(np.round(pt[0])), int(np.round(pt[1]))
        arr = self.masked_img3d[t,
                                max(0, ix - d):ix + d,
                                max(0, iy - d):iy + d].flatten()
        arr = arr[arr > 0]
        return get_mode(arr, median_lowpass=self.__bg_median_lowpass)

    def add_kymo_signal(self, flare):
        kmdf = flare.kmdf
        t_list = kmdf.index.get_level_values('t').unique()

        sig_chn, nrm_chn = self.ch + '_sig', self.ch + '_tnorm'
        kmdf[sig_chn], kmdf[nrm_chn] = None, None
        tgen = zip(t_list, self.pt_sig[t_list], self.pt_tnorm[t_list])
        for t, pt_sig1d, pt_sig_tnorm1d in tgen:
            ptids = kmdf.loc[t, 'ptids'].values
            kmdf.loc[t, sig_chn] = pt_sig1d[ptids]
            kmdf.loc[t, nrm_chn] = pt_sig_tnorm1d[ptids]

    # def add_constant_reference(self, flare, func=None):
    #     if func is None:
    #         func = (lambda larr, garr: np.median(larr[larr > 1]))

    #     df = flare.fsdf[['fc_x', 'fc_y']]
    #     t_list = df.index

    #     local_values = []
    #     all_values = []
    #     zipgen = zip(self.gp.gs[t_list], self.fdata_pts[t_list],
    #                  df.as_matrix())
    #     desc = "Flare %03d, building %s ref:" % (flare.index, self.ch)
    #     for ag, vals, pt in tqdm(zipgen, total=t_list.size, desc=desc):
    #         all_values += vals.tolist()
    #         cfilter = generate_circle_filter(pt)
    #         inside = np.apply_along_axis(cfilter, 1, ag.pts)
    #         local_values += vals[inside].tolist()
    #     ref = func(np.array(local_values), np.array(all_values))
    #     flare.fsdf[self.ch + '_ref'] = pd.Series(
    #         [ref] * t_list.size, index=t_list)

    def add_flare_signal(self, flare):
        fsdf = flare.fsdf

        def get_center_value(subdf):
            if np.isnan(subdf.name):
                return np.nan
            f = intpl.interp1d(subdf['ss'], subdf[self.ch + '_sig'])
            s = flare.fsdf.fc_s[subdf.name]
            return f(s)

        flare.fsdf[self.ch + '_sig'] = flare.kmdf.groupby('t').apply(
            get_center_value).astype('float')

        main_chn, ref_chn, bg_chn = self.ch + '_norm', self.ch + '_ref', self.ch + '_bg'
        f_ref = lambda row: self.get_reference(row.name, np.array([row.fc_x, row.fc_y]))
        f_bg = lambda row: self.get_background(row.name, np.array([row.fc_x, row.fc_y]))

        fsdf[ref_chn] = fsdf.apply(f_ref, axis=1)
        fsdf[bg_chn] = fsdf.apply(f_bg, axis=1)
        fsdf[main_chn] = ((fsdf[self.ch + '_sig'] - fsdf[bg_chn]) /
                          (fsdf[ref_chn] - fsdf[bg_chn]))

    def add_total_signal(self, flare):
        sum_chn, ref_chn, bg_chn = self.ch + '_total', self.ch + '_ref', self.ch + '_bg'
        group = flare.kmdf[self.ch
                           + '_sig'].dropna().astype(float).groupby('t')
        flare.fsdf[sum_chn] = group.agg('sum') / group.agg(
            'count') * flare.fsdf.jxn_length
        # flare.fsdf[sum_chn] /= flare.fsdf[sum_chn].dropna().mean()
        flare.fsdf[sum_chn] = ((flare.fsdf[sum_chn] / flare.fsdf[bg_chn] - 1) /
                               (flare.fsdf[ref_chn] / flare.fsdf[bg_chn] - 1))


class RhoProcessor(ChannelProcessor):
    def __init__(self, *args, **kwargs):
        self.ch = 'rho'
        super().__init__(*args, **kwargs)
        self.__bg_median_lowpass = 0.5


class OcclProcessor(ChannelProcessor):
    def __init__(self, *args, **kwargs):
        self.ch = 'jxn'
        super().__init__(*args, **kwargs)


class ZO1Processor(ChannelProcessor):
    def __init__(self, *args, **kwargs):
        self.ch = 'jxn'
        super().__init__(*args, **kwargs)


class ActinProcessor(ChannelProcessor):
    def __init__(self, *args, **kwargs):
        self.ch = 'jxn'
        super().__init__(*args, **kwargs)


def get_kymogram(df, scale=False):
    tidx = df.index.levels[0].values
    smax = int(np.ceil(df['ss'].max()) + 1)
    ctr = (smax - 1) / 2
    l = smax - 1

    kymo_arr = np.zeros((tidx.size, l))
    for t, km in zip(tidx, kymo_arr):
        d = df.loc[t]
        ss = d.iloc[:, 0].values.copy()
        ss -= ss[-1] / 2
        vs = d.iloc[:, 1].values
        f = intpl.interp1d(
            ss, vs, bounds_error=False, fill_value=(vs[0], vs[-1]))
        if scale:
            km[:] = f(np.linspace(ss[0], ss[-1], l))
        else:
            x0 = np.round(ctr + ss[0] - 0.5) - ctr + 0.5
            x1 = np.round(ctr + ss[-1] - 0.5) - ctr + 0.5
            i0 = int(ctr + x0 - 0.5)
            i1 = int(ctr + x1 - 0.5)
            km[i0:i1] = f(np.arange(x0, x1))
    return kymo_arr


def composite_kymogram(kmdf, scale=False):
    km = get_kymogram(kmdf[['ss', 'rho']], scale=scale)
    arr = km[km > 0]
    vmin = np.percentile(arr, q=40)
    vmax = np.percentile(arr, q=99.9)
    ch1 = np.uint8(255 * np.clip(
        (km - vmin) / (vmax - vmin), a_min=0, a_max=1))

    km = get_kymogram(kmdf[['ss', 'jxn']], scale=scale)
    arr = km[km > 0]
    vmin = np.percentile(arr, q=10)
    vmax = np.percentile(arr, q=99.9)
    ch2 = np.uint8(255 * np.clip(
        (km - vmin) / (vmax - vmin), a_min=0, a_max=1))

    return np.array([ch1, ch2, np.zeros_like(ch1)]).transpose(1, 2, 0)


def comp_kymo4x(df, mult=2, dtype='float'):
    if mult % 2 != 0:
        raise ValueError('Multiplier should be a multiple of 2')

    tidx = df.index.levels[0].values
    tsize = tidx.max() - tidx.min() + 1
    t0 = tidx[0]
    l1 = df.index.get_level_values('ptn').max() + 1
    km = np.zeros((tsize, 2 * l1, 3), dtype='float32')
    for t in tidx:
        d = df.loc[t][['rho_tnorm',
                       'jxn_tnorm']].as_matrix().astype('float').transpose()
        d1 = np.uint8(df.has_flare[t].values)
        d = np.concatenate((d, d1[None, :]))
        l0 = d.shape[1]
        d = np.tile(d[:, :, None], (1, 1, 2)).reshape(3, -1)
        km[t - t0, l1 - l0:l1 + l0] = d.transpose()
    km = km.transpose(2, 0, 1)
    ch = km[0]
    arr = ch[ch > 0]
    vmin0 = np.percentile(arr, q=35)
    vmax0 = np.percentile(arr, q=99.9)

    ch = km[1]
    arr = ch[ch > 0]
    vmin1 = np.percentile(arr, q=5)
    vmax1 = np.percentile(arr, q=99.9)

    rng = [[vmin0, vmax0], [vmin1, vmax1], [0, 2]]
    if dtype == 'float':
        pass
    elif dtype == 'uint':
        ch1 = np.uint8(255 * np.clip(
            (km[0] - vmin0) / (vmax0 - vmin0), a_min=0, a_max=1))
        ch2 = np.uint8(255 * np.clip(
            (km[1] - vmin1) / (vmax1 - vmin1), a_min=0, a_max=1))
        ch3 = 128 * km[2].astype('uint8')
        km = np.array([ch1, ch2, ch3])
        rng = [[0, 255], [0, 255], [0, 255]]
    else:
        raise ValueError('Unknown dtype = %s' % dtype)
    k = mult // 2
    km = np.tile(km[:, :, None, :, None], (1, 1, mult, 1, k)).reshape(
        3,
        -1,
        mult * l1, )
    return km, rng


def save3ch(fname, data, rng=None):
    wr = iio.get_writer(fname, 'tiff', 'v')
    meta = {'photometric': 'minisblack'}
    if rng is not None:
        meta['description'] = 'kymoranges: ' + repr(rng)
    wr.set_meta_data(meta)
    wr.append_data(data)
    wr.close()


jcomposite = '/path/to/jcomposite.py'
jreadroi = '/path/to/jreadroi.py'


def prepare_kymo(flare, folder=None):
    if folder is None:
        folder = flare.directory
    img, rng = comp_kymo4x(flare.kmdf, mult=2)
    fname = folder + "/%s.tiff" % flare.name
    if not os.path.exists(folder):
        os.makedirs(folder)
    save3ch(fname, img, rng)
    flare.data.kymo_filename = fname
    call([jcomposite, fname])


def set_fc_path_from_roi(flare):
    process = Popen([jreadroi, flare.data.kymo_filename], stdout=PIPE)
    output, _ = process.communicate()
    process.wait()
    try:
        data = eval(output)
    except SyntaxError as e:
        print(output)
        raise e
    dims = data['dims']
    coords = np.array(data['coords'])[:, ::-1]
    return dims, coords


def fc_from_roi(flare):
    kmdf = flare.kmdf
    dims, coords = set_fc_path_from_roi(flare)
    coords = coords.astype('float') - 0.5

    assert dims[1] == kmdf.index.get_level_values('t').unique().size * 2

    km = iio.volread(flare.data.kymo_filename)
    km = km.sum(axis=0)
    pos = np.array([np.where(l > 0)[0].min() for l in km[::2]])
    t0 = flare.kmdf.index.get_level_values(0).min()
    t1 = flare.kmdf.index.get_level_values(0).max()

    if coords[0, 0] > coords[-1, 0]:
        coords = coords[::-1]

    assert np.abs(sorted(coords[:, 0]) - coords[:, 0]).max() < 1e-5

    tmin = int(np.round(coords[0, 0] / 2))
    tmin = np.clip(tmin, a_min=0, a_max=t1 - t0)
    tmax = int(np.round(coords[-1, 0] / 2))
    tmax = np.clip(tmax, a_min=0, a_max=t1 - t0)

    f = intpl.interp1d(
        coords[:, 0],
        coords[:, 1],
        kind='linear',
        bounds_error=False,
        fill_value=(coords[0, 1], coords[-1, 1]))
    ts = np.arange(tmin, tmax + 1)
    x2s = f(2 * ts + 0.5)

    fc_ss = []
    for t, x2 in zip(ts, x2s):
        si = int(np.floor((x2 - pos[t]) / 2))
        # print(t0+t)
        si = np.clip(si, a_min=0, a_max=kmdf.loc[t0 + t].index[-1])
        rm = ((x2 - pos[t]) % 2) * 0.5
        s = kmdf.loc[t0 + t, si].ss + rm
        s = np.clip(s, a_min=0, a_max=flare.fsdf.jxn_length[t0 + t])
        fc_ss.append(s)

    # flare.fsdf['fc_s'] = None
    flare.fsdf['fc_s'] = pd.Series(fc_ss, index=ts + t0, dtype='float')

    fc_s = flare.fsdf.fc_s.dropna()
    flare.fsdf['fc_x'] = None
    flare.fsdf['fc_y'] = None
    fc_pts = []
    for t in fc_s.index:
        df = kmdf.loc[t]
        X = intpl.interp1d(df.ss, df.x)
        Y = intpl.interp1d(df.ss, df.y)
        xy = [float(X(fc_s[t])), float(Y(fc_s[t]))]
        fc_pts.append(xy)
    fc_pts = np.array(fc_pts)
    flare.fsdf[['fc_x', 'fc_y']] = pd.DataFrame(fc_pts, index=fc_s.index)


def align_fsdf_by_slope(fsdf, ch='rho_norm'):
    df = fsdf[['time', ch]].dropna()
    xy = df.as_matrix().astype('float')
    slopes = np.array([
        stats.linregress(xy[(i - 1):(i + 2)]).slope
        for i in range(1, len(df) - 1)
    ])
    ix = df.index[0] + slopes.argmax() + 1
    fsdf['t_slope_align'] = pd.Series(fsdf.index.values - ix, index=fsdf.index)
    fsdf['time_slope_align'] = fsdf.time - fsdf.time.loc[ix]


def diff_jxn_len(fsdf):
    zero_len = fsdf.jxn_length_um.dropna()[0:10].mean()
    fsdf['jxn_length_diff_um'] = fsdf.jxn_length_um - zero_len
