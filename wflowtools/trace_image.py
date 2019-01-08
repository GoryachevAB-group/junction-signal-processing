from collections import defaultdict
import numpy as np


def trace_bw_image(W: np.ndarray):
    links = defaultdict(set)

    def connect(p1, p2):
        links[p1].add(p2)
        links[p2].add(p1)

    diff = lambda w, a, b: np.pad(w, ((a, a), (b, b)), 'constant', constant_values=False)

    arr = diff(W[:, 1:] != W[:, :-1], 0, 1)
    for x, y in zip(*np.nonzero(arr)):
        connect((x, y), (x + 1, y))
    arr = diff(W[1:, :] != W[:-1, :], 1, 0)
    for x, y in zip(*np.nonzero(arr)):
        connect((x, y), (x, y + 1))

    lines = []
    while links:
        pt1 = next(iter(links.keys()))
        pt2 = links[pt1].pop()
        links[pt2].remove(pt1)
        line = [pt1]
        while links[pt2]:
            line.append(pt2)
            pt1, pt2 = pt2, links[pt2].pop()
            links[pt2].remove(pt1)
        line.append(pt2)
        for pt in line:
            if pt in links and not links[pt]:
                del links[pt]
        lines.append(np.array(line))
    return lines
