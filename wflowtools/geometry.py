import numpy as np
import shapely.ops as ops

def line_length(lst):
    return np.sum(np.sqrt(np.sum((lst[1:] - lst[:-1]) ** 2, axis=1)))


def length_along(lst):
    out = np.zeros(lst.shape[0], dtype=np.float32)
    ls = np.sqrt(np.sum((lst[1:] - lst[:-1]) ** 2, axis=1))
    out[1:] = np.cumsum(ls)
    return out


def get_rectpts(img):
    xmin, xmax = -0.5, img.shape[0]-0.5
    ymin, ymax = -0.5, img.shape[1]-0.5
    return np.array([[xmin, ymin],
                     [xmax, ymin],
                     [xmax, ymax],
                     [xmin, ymax]])

def get_geometry_poly(ag, rectpts=None):
    # We need to add edges along rectangle to close polygons
    if True:
        if rectpts is None:
            xmin = ag.pts[:, 0].min()
            xmax = ag.pts[:, 0].max()
            ymin = ag.pts[:, 1].min()
            ymax = ag.pts[:, 1].max()
            rectpts = np.array([[xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax]])

        # add lonely nodes, sort them by angle from central point
        pts = np.array(list(rectpts) + list(ag.pts[ag.ptids[0]]))
        cp = rectpts.mean(axis=0)
        angles = np.arctan2(*(pts - cp).transpose())
        angles -= angles[0]
        angles = np.mod(angles, 2 * np.pi)
        srt = angles.argsort()
        pts = np.array(pts[srt].tolist() + [rectpts[0]])

        # add cell labels around each point
        pt_labels = []
        for ptid in ag.ptids[0]:
            for i, edge in enumerate(ag.edges):
                if ptid == edge[0]:
                    pt_labels.append(tuple(ag.edge_labels[i]))
                if ptid == edge[-1]:
                    pt_labels.append(tuple(ag.edge_labels[i][::-1]))
        pt_labels = np.array(([None] * 4 + pt_labels))[srt]

        # create boundary edges, set labels
        b_edges = np.array([pts[:-1], pts[1:]]).transpose(1, 0, 2)
        b_labels = []
        for i, pt in enumerate(pts[:-1]):
            if pt_labels[i] is not None:
                b_labels.append((pt_labels[i][0],))
            elif pt_labels[i + 1] is not None:
                b_labels.append((pt_labels[i + 1][1],))
            else:
                raise ValueError("Unknown boundary label")

    total_edges_pts = [ag.pts[e] for e in ag.edges] + list(b_edges)
    total_labels = [tuple(l) for l in ag.edge_labels] + b_labels

    geom_dict = {}
    cells_labels = set(element for tupl in total_labels for element in tupl)
    for cell in cells_labels:
        edgespts = [total_edges_pts[i] for i, l in enumerate(total_labels) if cell in l]
        lines = list(ops.polygonize(edgespts))
        if not lines:
            l = ops.linemerge(edgespts)
            edgespts.append([l.boundary[0].coords[0], l.boundary[1].coords[0]])
            lines = list(ops.polygonize(edgespts))
        if len(lines) == 1:
            lines = lines[0]
        elif len(lines) > 1:
            lines = shp.geometry.MultiPolygon(lines)
        geom_dict[cell] = lines
    return geom_dict
