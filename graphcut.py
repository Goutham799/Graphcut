from numpy import *
import numpy
import maxflow
from PIL import Image
from matplotlib import pyplot as plt
from pylab import *
import cv2


def graph(file,
          k,
          s,
          fore,
          back):
    I = (Image.open(file).convert('L'))
    If = I.crop(fore)
    Ib = I.crop(back)

    I, If, Ib = array(I), array(If), array(Ib)
    Ifmean, Ibmean = mean(cv2.calcHist([If], [0], None, [256], [0, 256])), mean(
        cv2.calcHist([Ib], [0], None, [256], [0, 256]))
    F, B = ones(shape=I.shape), ones(shape=I.shape)
    Im = I.reshape(-1, 1)
    m, n = I.shape[0], I.shape[1]
    g, pic = maxflow.Graph[int](m, n), maxflow.Graph[int]()

    structure = np.array([[inf, 0, 0],
                          [inf, 0, 0],
                          [inf, 0, 0]
                          ])

    source, sink, J = m * n, m * n + 1, I
    nodes, nodeids = g.add_nodes(m * n), pic.add_grid_nodes(J.shape)

    pic.add_grid_edges(nodeids, 0), pic.add_grid_tedges(nodeids, J, 255 - J)
    gr = pic.maxflow()
    IOut = pic.get_grid_segments(nodeids)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            F[i, j] = -log(abs(I[i, j] - Ifmean) / (abs(I[i, j] - Ifmean) + abs(I[i, j] - Ibmean)))
            B[i, j] = -log(abs(I[i, j] - Ibmean) / (abs(I[i, j] - Ibmean) + abs(I[i, j] - Ifmean)))

    F, B = F.reshape(-1, 1), B.reshape(-1, 1)

    for i in range(Im.shape[0]):
        Im[i] = Im[i] / linalg.norm(Im[i])
    w = structure

    for i in range(m * n):
        ws = (F[i] / (F[i] + B[i]))
        wt = (B[i] / (F[i] + B[i]))
        g.add_tedge(i, ws[0], wt)

        if i % n != 0:
            w = k * exp(-(abs(Im[i] - Im[i - 1]) ** 2) / s)
            g.add_edge(i, i - 1, w[0], k - w[0])

        if (i + 1) % n != 0:
            w = k * exp(-(abs(Im[i] - Im[i + 1]) ** 2) / s)
            g.add_edge(i, i + 1, w[0], k - w[0])

        if i // n != 0:
            w = k * exp(-(abs(Im[i] - Im[i - n]) ** 2) / s)
            g.add_edge(i, i - n, w[0], k - w[0])

        if i // n != m - 1:
            w = k * exp(-(abs(Im[i] - Im[i + n]) ** 2) / s)
            g.add_edge(i, i + n, w[0], k - w[0])

    I = array(Image.open(file))
    print(f"The maximum flow for %s is %d" % (file, gr))
    Iout = ones(shape=nodes.shape)

    for i in range(len(nodes)):
        Iout[i] = g.get_segment(nodes[i])
    out = 255 * ones((I.shape[0], I.shape[1], 3))

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if IOut[i, j] == False:
                if len(I.shape) == 2:
                    out[i, j, 0], out[i, j, 1], out[i, j, 2] = I[i, j], I[i, j], I[i, j]
                if len(I.shape) == 3:
                    out[i, j, 0], out[i, j, 1], out[i, j, 2] = I[i, j, 0], I[i, j, 1], I[i, j, 2]
            else:
                out[i, j, 0], out[i, j, 1], out[i, j, 2] = 1, 255, 255

    figure()
    plt.imshow(out, vmin=0, vmax=255)
    plt.show()

graph('C:/Users/Goutham/Downloads/GraphCut Assingnment (1)/GraphCut Assingnment/input1.jpg', 2, 100, (225, 142, 279, 185), (7, 120, 61, 163))
graph('C:/Users/Goutham/Downloads/GraphCut Assingnment (1)/GraphCut Assingnment/input2.jpg', 2, 120, (148, 105, 201, 165), (11, 12, 80, 52))
