import numpy as np
# from scipy.spatial.distance import cdist
from numba import njit, prange


def init_boids(boids: np.ndarray, asp: float, vrange: tuple[float, float]):
    '''

    :param boids:
    :param asp:
    :param vrange:
    :return:
    '''

    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s


def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
    :param boids:
    :param dt:
    :return: array N x (x0, y0, x1, y1) for arrow painting
    """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))


def vclip(v: np.ndarray, vrange: tuple[float, float]):
    norm = np.linalg.norm(v, axis=1)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)


def propagate(boids1: np.ndarray, boids2: np.ndarray, dt: float, vrange: tuple[float, float]):
    boids1[:, 2:4] += dt * boids1[:, 4:6]
    vclip(boids1[:, 2:4], vrange)
    boids1[:, 0:2] += dt * boids1[:, 2:4]

    boids2[:, 2:4] += dt * boids2[:, 4:6]
    vclip(boids2[:, 2:4], vrange)
    boids2[:, 0:2] += dt * boids2[:, 2:4]














@njit(parallel=True)
def dist(matrix1, matrix2):
    distances = np.zeros((matrix1.shape[0], matrix2.shape[0]))

    for i in prange(matrix1.shape[0]):
        for j in range(matrix2.shape[0]):
            distance = np.sqrt(np.sum((matrix1[i] - matrix2[j]) ** 2))
            distances[i, j] = distance

    return distances


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:

    center = np.array([np.mean(boids[neigh_mask, 0]), np.mean(boids[neigh_mask, 1])])
    a = (center - boids[idx, :2]) / perception
    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray,
               perception: float) -> np.ndarray:

    d = np.array([np.mean(boids[neigh_mask, 0] - boids[idx, 0]), np.mean(boids[neigh_mask, 1] - boids[idx, 1])])

    return -d / ((d[0] ** 2 + d[1] ** 2) + 1)


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    #
    v_mean = np.array([  np.mean(boids[neigh_mask, 2]), np.mean(boids[neigh_mask, 3])  ])

    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a


@njit(parallel=True)
def walls(boids: np.ndarray, asp: float):
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]

    a_left = 1 / (np.abs(x) + c) ** 2
    a_right = -1 / (np.abs(x - asp) + c) ** 2

    a_bottom = 1 / (np.abs(y) + c) ** 2
    a_top = -1 / (np.abs(y - 1.) + c) ** 2

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit()
def noise_generator():
    return np.random.rand(2)



@njit(parallel=True)
def flocking(boids1: np.ndarray,
             boids2: np.ndarray,
             perception: float,
             coeffs11: np.ndarray,
             coeffs12,
             coeffs21,
             coeffs22,
             asp: float,
             vrange: tuple):
    D1 = np.zeros((boids1.shape[0], boids1.shape[0]))
    D2 = np.zeros((boids2.shape[0], boids2.shape[0]))
    D12 = np.zeros((boids1.shape[0], boids2.shape[0]))

    D1 = dist(boids1[:, 0:2], boids1[:, 0:2])
    D2 = dist(boids2[:, 0:2], boids2[:, 0:2])
    D12 = dist(boids1[:, 0:2], boids2[:, 0:2])

    N1 = boids1.shape[0]
    N2 = boids2.shape[0]

    for i in prange(N1):
        D1[i, i] = perception + 1

    for i in prange(N2):
        D2[i, i] = perception + 1

    wal1 = walls(boids1, asp)
    wal2 = walls(boids2, asp)

    mask1 = D1 < perception
    mask12 = D12 < perception
    mask2 = D2 < perception







    for i in prange(N1):
        if not np.any(mask1[i]):
            coh11 = np.zeros(2)
            alg11 = np.zeros(2)
            sep11 = np.zeros(2)
        else:
            coh11 = cohesion(boids1, i, mask1[i], perception)
            alg11 = alignment(boids1, i, mask1[i], vrange)
            sep11 = separation(boids1, i, mask1[i], perception)

        if not np.any(mask12[i]):
            coh12 = np.zeros(2)
            alg12 = np.zeros(2)
            sep12 = np.zeros(2)

        else:
            coh12 = cohesion(boids2, i, mask12[i], perception)
            alg12 = alignment(boids2, i, mask12[i], vrange)
            sep12 = separation(boids2, i, mask12[i], perception)

        noise = np.random.rand()/10
        a = coeffs11[0] * coh11 + coeffs11[1] * alg11 + coeffs11[2] * sep11 + coeffs11[3] * wal1[i] + \
            coeffs12[0] * coh12 + coeffs12[1] * alg12 + coeffs12[2] * sep12 + coeffs11[4] * noise + coeffs12[4] * noise


        boids1[i, 4:6] = a

    for i in prange(N2):
        if not np.any(mask2[i]):
            coh22 = np.zeros(2)
            alg22 = np.zeros(2)
            sep22 = np.zeros(2)
        else:
            coh22 = cohesion(boids2, i, mask2[i], perception)
            alg22 = alignment(boids2, i, mask2[i], vrange)
            sep22 = separation(boids2, i, mask2[i], perception)

        if not np.any(mask12[:, i]):
            coh21 = np.zeros(2)
            alg21 = np.zeros(2)
            sep21 = np.zeros(2)
        else:
            coh21 = cohesion(boids1, i, mask12[:, i], perception)
            alg21 = alignment(boids1, i, mask12[:, i], vrange)
            sep21 = separation(boids1, i, mask12[:, i], perception)

        noise = np.random.rand() / 10
        a = coeffs22[0] * coh22 + coeffs22[1] * alg22 + coeffs22[2] * sep22 + coeffs22[3] * wal2[i] + \
            coeffs21[0] * coh21 + coeffs21[1] * alg21 + coeffs21[2] * sep21 + coeffs22[4] * noise + coeffs21[4] * noise

        boids2[i, 4:6] = a


if __name__ == "__main__":
    import numpy as np

    #
    #
    # def compute_distances(matrix1, matrix2):
    #     distances = cdist(matrix1, matrix2)
    #     return distances
    #
    #
    # # Пример использования
    # matrix1 = np.array([[1, 0], [3, 4], [5, 6]])
    # matrix2 = np.array([[2, 0], [4, 5]])
    #
    # result = compute_distances(matrix1, matrix2)
    # print(result)
    #
    # boids = np.array([
    #     [1, 2, 3, 4, 5, 6],
    #     [1, 2, 3, 4, 5, 6],
    #     [1, 2, 3, 4, 5, 6]
    # ])
    # neigh_mask = [True, True, False]
    # print(np.mean(boids[neigh_mask, 2:4]))
