import numpy as np
# import scipy.interpolate as intpl

import evtk.hl as vtk
from tqdm import tqdm
from . import geometry as GM


def generate_facets(pts, list1, list2):
    facets = []
    i, j = 1, 1
    while i < len(list1) and j < len(list2):
        l1 = GM.line_length(pts[[list1[i], list2[j - 1]]])
        l2 = GM.line_length(pts[[list1[i - 1], list2[j]]])
        if l1 < l2:
            facets.append([list2[j - 1], list1[i - 1], list1[i]])
            i += 1
        else:
            facets.append([list2[j - 1], list1[i - 1], list2[j]])
            j += 1
    while i < len(list1):
        facets.append([list2[j - 1], list1[i - 1], list1[i]])
        i += 1
    while j < len(list2):
        facets.append([list2[j - 1], list1[i - 1], list2[j]])
        j += 1
    return facets


def save_mesh(fname, pts, facets, facet_type=5, **kwargs):
    x = pts[:, 0].astype('float')
    y = pts[:, 1].astype('float')
    if pts.shape[1] >= 3:
        z = pts[:, 2].astype('float')
    else:
        z = np.array([0] * x.size).astype('float')
    connectivity = facets.flatten()
    ncells = facets.shape[0]
    offsets = 3 + 3 * np.arange(ncells)
    types = np.array([facet_type] * ncells)
    return vtk.unstructuredGridToVTK(fname, x, y, z, connectivity, offsets,
                                     types, **kwargs)


def save_func(fname,
              gs,
              rho,
              jxn,
              edge_label,
              trange,
              l_scale=0.25,
              t_scale=0.25,
              transpose=True):
    t1, t2 = trange
    tslice = slice(t1, t2)
    bigT = np.arange(100000)

    PTS = []
    es = []
    rho_data, jxn_data = [], []

    for ag, rh, jx, t in tqdm(
            zip(gs[tslice], rho[tslice], jxn[tslice], bigT[tslice]),
            total=t2 - t1):
        idx = [
            i for i, label in enumerate(ag.edge_labels)
            if set(label) == set(edge_label)
        ][0]
        if tuple(ag.edge_labels[idx]) == edge_label:
            ptids = ag.edges[idx]
        else:
            ptids = ag.edges[idx][::-1]
        pts = ag.pts[ptids]
        if transpose:
            pts = pts[:, ::-1]

        x1 = len(PTS)
        PTS += list(
            np.pad(
                pts * l_scale, [[0, 0], [0, 1]],
                mode='constant',
                constant_values=t * t_scale))
        x2 = len(PTS)
        es.append(np.arange(x1, x2))
        rho_data += rh[ptids].tolist()
        jxn_data += jx[ptids].tolist()
    PTS = np.array(PTS)
    facets = []
    for e1, e2 in zip(es[:-1], es[1:]):
        facets += generate_facets(PTS, e1, e2)
    save_mesh(
        fname,
        PTS,
        np.array(facets, dtype='int'),
        pointData={"rho": np.array(rho_data),
                   "occl": np.array(jxn_data)})

def writePVD(pvd_fname, pvu_fnames, mult=1):
    import xml.dom.minidom

    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd_root.setAttribute("byte_order", "LittleEndian")
    pvd.appendChild(pvd_root)

    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)

    for t, fname in enumerate(pvu_fnames):
        dataSet = pvd.createElementNS("VTK", "DataSet")
        dataSet.setAttribute("timestep", "%3.1f" % (t*mult) )
        dataSet.setAttribute("group", "")
        dataSet.setAttribute("part", "0")
        dataSet.setAttribute("file", fname)
        collection.appendChild(dataSet)

    with open(pvd_fname, 'w') as f:
        pvd.writexml(f, newl='\n')

