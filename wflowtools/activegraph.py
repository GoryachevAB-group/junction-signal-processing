import warnings
from collections import defaultdict

import cv2
import numpy as np
from scipy import interpolate

from .geometry import line_length

# import shapely as shp

CELL_LABEL_TYPE = np.int64
EDGE_LABEL_TYPE = np.uint64
PTID_TYPE = np.uint64
FLOAT = np.float64

PRECISION = 1 << 10


def __resample_line__(pts, spacing=1.0, n=None):
    vecs = pts[1:] - pts[:-1]
    lens = np.sqrt(np.sum(vecs**2, axis=1))
    s_values = np.concatenate((np.array([0], dtype=FLOAT), np.cumsum(lens)))
    s_max = s_values[-1]
    if n is None:
        n = int(round(s_max / spacing)) + 1
    if n <= 2:
        return pts[[0, -1]][:]
    f = interpolate.interp1d(s_values, pts.transpose())
    new_edge = np.array([f(s) for s in np.linspace(0, s_max, n)], dtype=FLOAT)
    return new_edge


class ActiveGraphPoint:
    def __init__(self, pt):
        self.pt = pt
        self.label = None
        self.index = None
        self.neighbors = set()
        self.degree = 0

    def add_neighbor(self, pt):
        self.neighbors.add(pt)
        self.degree += 1


class ActiveGraph:
    def __init__(self, W=None, E=None, fixed_edge_length=-1):
        self.pts = None  # n x 2 - points coordinates
        self.ptids = None  # [m1, m2, m3, m4] - lists of
        #    points of degree 1,2,3,4
        self.links = None  # [m1 x 1, m2 x 2, m3 x 3, m4 x 4] -
        #   lists of links of pidxs

        # boundary restriction
        # boundary pts = pts * M + V
        self.restriction_mult = None  # boundary restriction multiplier M

        self.fsx = None
        self.fsy = None

        if W is not None:
            assert isinstance(W, np.ndarray)
            self.__from_watershed__(W)
        if E is not None:
            self.set_potential(E, restrain_boundary=True)
        if fixed_edge_length > 0:
            self.fix_short_edges(fixed_edge_length)

    def __from_watershed__(self, W: np.ndarray):
        def connect(p1, p2):
            p1.add_neighbor(p2)
            p2.add_neighbor(p1)

        x0 = 1 - np.pad(
            W[:, 1:] == W[:, :-1], ((0, 0), (1, 1)),
            'constant',
            constant_values=True)
        x1 = np.pad(x0, ((1, 0), (0, 0)), 'constant', constant_values=0)
        x2 = np.pad(x0, ((0, 1), (0, 0)), 'constant', constant_values=0)
        y0 = 1 - np.pad(
            W[1:, :] == W[:-1, :], ((1, 1), (0, 0)),
            'constant',
            constant_values=True)
        y1 = np.pad(y0, ((0, 0), (1, 0)), 'constant', constant_values=0)
        y2 = np.pad(y0, ((0, 0), (0, 1)), 'constant', constant_values=0)

        pts = dict()
        for x, y in zip(*np.nonzero(x1 + x2 + y1 + y2)):
            pts[(x, y)] = ActiveGraphPoint(np.array([x, y], dtype=FLOAT) - 0.5)
        for x, y in zip(*np.nonzero(x0)):
            connect(pts[(x, y)], pts[(x + 1, y)])
        for x, y in zip(*np.nonzero(y0)):
            connect(pts[(x, y)], pts[(x, y + 1)])
        self.__from_agpoints__(pts.values())
        self.__label_edges__(W)
        return self

    def __from_agpoints__(self, ag_points):
        pts = list()
        ptids = [[], [], [], []]
        links = [[], [], [], []]
        for i, agp in enumerate(ag_points):
            agp.index = i
            agp.label = i
        for i, agp in enumerate(ag_points):
            pts.append(agp.pt.copy())
            ptids[agp.degree - 1].append(i)
            links[agp.degree
                  - 1].append([neighbor.index for neighbor in agp.neighbors])
        self.pts = np.array(pts, dtype=FLOAT)
        self.__set_ptids_links__(ptids, links)
        self.restriction_ptids = self.ptids[0]
        self.restriction_mult = np.ones((self.restriction_ptids.size, 2), dtype=FLOAT)

        self.__set_edges__(ag_points)
        return self

    def __set_ptids_links__(self, ptids, links):
        self.ptids = [np.array(ptid, dtype=PTID_TYPE) for ptid in ptids]
        self.links = [
            np.array(link, dtype=PTID_TYPE).reshape(-1, k + 1)
            for k, link in enumerate(links)
        ]
        for ptid, link in zip(self.ptids, self.links):
            srt = np.argsort(ptid)
            ptid[:] = ptid[srt]
            link[:] = link[srt]

    def __set_edges__(self, ag_ponts):
        edges = []
        fixed_label = -1

        def follow(agp):
            agp.label = fixed_label
            for neighbor in agp.neighbors:
                if neighbor.label != fixed_label:
                    edge = [agp.index]
                    cur_agp, prv_agp = neighbor, agp
                    while cur_agp.degree == 2 and cur_agp != agp:
                        edge.append(cur_agp.index)
                        cur_agp.label = fixed_label
                        cur_agp, prv_agp = \
                            [n for n in cur_agp.neighbors if n != prv_agp][0], \
                            cur_agp

                    edge.append(cur_agp.index)
                    edges.append(np.array(edge, dtype=PTID_TYPE))
                    follow(cur_agp)

        def find_start():
            for agp in ag_ponts:
                if agp.degree != 2 and agp.label != fixed_label:
                    return agp
            return None

        p = find_start()
        while p is not None:
            follow(p)
            p = find_start()
        self.edges = edges

    def __restore_from_edges__(self, edges: list):
        self.edges = [np.array(edge, dtype=PTID_TYPE) for edge in edges]
        ptids = [[], [], [], []]
        links = [[], [], [], []]
        links_dict = defaultdict(list)
        for edge in self.edges:
            links_dict[edge[0]].append(edge[1])
            links_dict[edge[-1]].append(edge[-2])
            ptids[1] += edge[1:-1].tolist()
            links[1] += np.array(
                [edge[:-2], edge[2:]], dtype=PTID_TYPE).transpose().tolist()
        for ptid, link in links_dict.items():
            k = len(link) - 1
            ptids[k].append(ptid)
            links[k].append(link)
        self.__set_ptids_links__(ptids, links)

    def __label_edges__(self, A):
        """ Labeling preserving orientation
            In direction of edge
            (left cell label, right cell label)
        """
        self.edge_labels = np.zeros(
            (len(self.edges), 2), dtype=CELL_LABEL_TYPE)
        tensor = np.array([[[0, -1], [1, 0]], [[0, 1], [-1, 0]]], dtype=FLOAT)
        for i, edge in enumerate(self.edges):
            pts = self.pts[edge[:2]]
            vecs = pts[1] - pts[0]
            vecs /= np.linalg.norm(vecs)
            label_pts = np.mean(pts[None, :, :], axis=1) + tensor.dot(vecs) / 2
            label_pts = np.round(label_pts).astype(CELL_LABEL_TYPE)

            label = tuple(A[tuple(c)] for c in label_pts)
            if label[0] == label[1]:
                raise ValueError("Bad edge labeling")
            elif label[0] > label[1]:
                self.edges[i] = edge[::-1]
                label = label[::-1]
            self.edge_labels[i, :] = label
        self.__set_edges_dct__()

    def __set_edges_dct__(self):
        self.edges_dct = defaultdict(list)
        for e, label_as_arr in zip(self.edges, self.edge_labels):
            label = tuple(label_as_arr)
            self.edges_dct[label].append(e[:])
            self.edges_dct[label[::-1]].append(e[::-1])

    def edge_pts(self, c1: int, c2: int, one_edge_or_error=False):
        """ Coordinates of edge with preserved orientation
            (c1 is left, c2 is right)
                        ^
                    c1  |  c2
                        |
        """
        es = self.edges_dct[(c1, c2)]
        if es is None:
            return None
        if not one_edge_or_error:
            return [self.pts[e] for e in es]
        if len(es) == 1:
            return self.pts[es[0]]
        else:
            raise ValueError("Edge (%d-%d) has %d components" % (c1, c2,
                                                                 len(es)))

    def is_peripheral(self, label):
        ptids = self.edges_dct.get(label, None)
        if ptids:
            start, end = ptids[0][[0, -1]]
            return start not in self.ptids[0] and end not in self.ptids[0]

    def redistribute_edge(self, edge: np.ndarray):
        # pts = __resample_line__(self.pts[edge], n=edge.size)
        # self.pts[edge[1:-1]] = pts[1:-1]
        pts = self.pts[edge].copy()
        vecs = pts[1:] - pts[:-1]
        lens = np.sqrt(np.sum(vecs**2, axis=1))
        unit_vecs = vecs / lens[:, None]
        dl = np.sum(lens) / lens.size
        i, j = 0, 1
        cl = dl
        while i < lens.size and j < lens.size:
            while cl < lens[i]:
                self.pts[edge[j]] = pts[i] + unit_vecs[i] * cl
                cl += dl
                j += 1
            cl -= lens[i]
            i += 1

    def set_potential(self, energy: np.ndarray, restrain_boundary=False):
        # transposed_energy = energy.transpose()
        x_max, y_max = energy.shape
        grad_x = cv2.Sobel(energy, cv2.CV_64F, 0, 1, ksize=5)
        grad_y = cv2.Sobel(energy, cv2.CV_64F, 1, 0, ksize=5)
        x_range = np.arange(x_max)
        y_range = np.arange(y_max)

        self.fsx = interpolate.RectBivariateSpline(x_range, y_range, grad_x)
        self.fsy = interpolate.RectBivariateSpline(x_range, y_range, grad_y)

        if restrain_boundary:
            for i, (x, y) in enumerate(self.pts[self.restriction_ptids]):
                if abs(x - 0) < 5 or abs(x - x_max) < 5:
                    self.restriction_mult[i, 0] = 0
                elif abs(y - 0) < 5 or abs(y - y_max) < 5:
                    self.restriction_mult[i, 1] = 0
                else:
                    warnings.warn('Boundary point not at border')

    def step(self, alpha=0.04, beta=0.01):
        displacements = np.zeros_like(self.pts)
        # length minimization displacement
        for p, l in zip(self.ptids, self.links):
            displacements[p] = alpha * (
                np.mean(self.pts[l], axis=1) - self.pts[p])
        # potential minimization displacement
        displacements[:, 0] += beta * self.fsx(
            self.pts[:, 0], self.pts[:, 1], grid=False)
        displacements[:, 1] += beta * self.fsy(
            self.pts[:, 0], self.pts[:, 1], grid=False)
        # restrict boundary points
        displacements[self.restriction_ptids] = displacements[self.restriction_ptids] * self.restriction_mult
        self.pts += displacements
        return displacements

    def fix_short_edges(self, max_length: FLOAT):
        ptids_set = set(self.ptids[0])
        dct = {ptid: index for index, ptid in enumerate(self.ptids[0])}
        for edge in self.edges:
            if edge[0] not in ptids_set and edge[-1] not in ptids_set:
                continue
            if line_length(self.pts[edge]) > max_length:
                continue
            for ptid in ptids_set & set(edge[[0, -1]]):
                self.restriction_mult[dct[ptid], :] = 0

    def redistribute(self):
        for e in self.edges:
            self.redistribute_edge(e)

    # matplotlib shortcut
    def plot(self, ax, **kwargs):
        for e in self.edges:
            ax.plot(*self.pts[e].transpose()[::-1], **kwargs)

    def __getstate__(self):
        pts = np.round(PRECISION * self.pts.flatten())
        return {
            'pts': pts.astype(np.int32),
            'edges': [edge.astype(np.uint32).tolist() for edge in self.edges],
            'edge_labels': self.edge_labels,
            'restriction_mult': self.restriction_mult,
            'restriction_ptids': self.ptids[0]
        }

    def __setstate__(self, state: dict):
        self.pts = np.array(
            state['pts'], dtype=FLOAT).reshape(-1, 2) / PRECISION
        self.__restore_from_edges__(state['edges'])
        self.edge_labels = state['edge_labels']
        self.__set_edges_dct__()

        if np.all(self.ptids[0] == state['restriction_ptids']):
            self.restriction_mult = state['restriction_mult']
            self.restriction_ptids = state['restriction_ptids']
        else:
            raise ValueError("Restriction points don't match")

    def set_spacing(self, spacing: float):
        dct = dict()
        pts = []
        # for k in (0, 2, 3):
        #     l = len(pts)
        #     pts += self.pts[self.ptids[k]].tolist()
        #     dct.update({p: l + i for i, p in enumerate(self.ptids[k])})
        
        for edge in self.edges:
            for ptid in edge[[0,-1]]:
                if ptid not in dct:
                    dct[ptid] = len(pts)
                    pts.append(self.pts[ptid])

        edges = []
        for edge in self.edges:
            l = len(pts)
            inner_pts = __resample_line__(
                self.pts[edge], spacing=spacing)[1:-1]
            pts += inner_pts.tolist()
            new_edge = ([dct[edge[0]]] + list(range(l, l + len(inner_pts))) +
                        [dct[edge[-1]]])
            edges.append(new_edge)

        self.pts = np.array(pts, dtype=FLOAT)
        self.__restore_from_edges__(edges)
