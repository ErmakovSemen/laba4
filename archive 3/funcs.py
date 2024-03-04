import numpy as np
from numba import njit, prange

def init_boids(boids: np.ndarray, asp: float, vrange: tuple[float, float]):
    """
    This function initializes a set of boids with random positions within a specified aspect ratio and
    random velocities within a specified range.
    
    :param boids: The `boids` parameter is a numpy array that represents the boids in a simulation. It
    seems to have a shape of (n, 4) where n is the number of boids. Each row in the array represents a
    boid and contains the following information:
    :type boids: np.ndarray
    :param asp: The `asp` parameter in the `init_boids` function represents the aspect ratio of the
    simulation space. It is used to generate initial positions for the boids within the specified aspect
    ratio
    :type asp: float
    :param vrange: The `vrange` parameter is a tuple that specifies the range of initial velocities for
    the boids. The first element of the tuple represents the minimum velocity value, and the second
    element represents the maximum velocity value that the boids can have initially
    :type vrange: tuple[float, float]
    """
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
    """
    The function `vclip` clips the magnitude of vectors in an array `v` to a specified range defined by
    `vrange`.
    
    :param v: It looks like you were about to provide some information about the parameter `v`, but the
    message got cut off. Could you please provide more details or complete the sentence so that I can
    assist you better?
    :type v: np.ndarray
    :param vrange: The `vrange` parameter is a tuple containing two float values. The first value
    represents the minimum allowed norm for the vectors in the input array `v`, and the second value
    represents the maximum allowed norm. The function `vclip` clips the vectors in the input array `v`
    to ensure
    :type vrange: tuple[float, float]
    """
    norm = np.linalg.norm(v, axis=1)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)


def propagate(boids: np.ndarray,
              dt: float,
              vrange: tuple[float, float],
              arange: tuple[float, float]):
    """
    This function updates the positions and velocities of boids based on their current velocities and
    acceleration ranges.
    
    :param boids: It looks like the code you provided is a function named `propagate` that operates on a
    NumPy array `boids` along with some other parameters `dt`, `vrange`, and `arange`. The function
    seems to update the positions and velocities of the boids based on their current
    :type boids: np.ndarray
    :param dt: The `dt` parameter in the `propagate` function represents the time step for updating the
    positions of the boids. It is a float value that determines how much time elapses in each iteration
    of the simulation
    :type dt: float
    :param vrange: The `vrange` parameter likely represents the range of velocities that the boids can
    have. It is a tuple containing two float values representing the minimum and maximum velocities
    allowed for the boids
    :type vrange: tuple[float, float]
    :param arange: The `arange` parameter is a tuple containing two float values. These values represent
    the minimum and maximum acceleration values that can be applied to the boids during propagation. The
    `vclip` function is used to ensure that the acceleration values are within this range
    :type arange: tuple[float, float]
    """
    vclip(boids[:, 4:6], arange)
    boids[:, 2:4] += dt * boids[:, 4:6]
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]

@njit()
def distances(vecs: np.ndarray) -> np.ndarray:
    """
    The function calculates the pairwise distances between vectors in a given numpy array.
    
    :param vecs: The `vecs` parameter is a NumPy array containing vectors. The function `distances`
    calculates the pairwise distances between all vectors in the input array
    :type vecs: np.ndarray
    :return: The function `distances` takes an input numpy array `vecs` and calculates the pairwise
    distances between the vectors in the array. It returns a numpy array `D` where `D[i, j]` represents
    the distance between the vectors at indices `i` and `j` in the input array `vecs`.
    """
    n, m = vecs.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(m):
                s+= (vecs[i][k]- vecs[j][k])**2
            D[i,j] = s
    return D
@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    center = boids[neigh_mask, :2].mean(axis=0)
    a = (center - boids[idx, :2]) / perception
    return a

@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray,
               perception: float) -> np.ndarray:
    """
    This function calculates the separation vector for a boid based on its neighbors within a certain
    perception range.
    
    :param boids: The `separation` function you provided seems to calculate the separation vector for a
    boid at index `idx` based on its neighboring boids. The separation vector points away from the
    average position of neighboring boids to avoid collisions
    :type boids: np.ndarray
    :param idx: The `idx` parameter in the `separation` function represents the index of the current
    boid for which we are calculating separation from its neighbors
    :type idx: int
    :param neigh_mask: The `neigh_mask` parameter is likely a boolean mask that indicates which boids
    are considered neighbors of the boid at index `idx`. It is used to select the neighboring boids from
    the `boids` array for calculating separation behavior
    :type neigh_mask: np.ndarray
    :param perception: The `separation` function you provided seems to be calculating the separation
    vector for a boid at index `idx` from its neighboring boids based on a perception distance. The
    separation vector is calculated as the average direction away from the neighboring boids
    :type perception: float
    :return: the direction vector for separation between the current boid at index `idx` and its
    neighboring boids within the specified perception range. The direction vector is calculated based on
    the mean direction of separation from the neighboring boids. The returned vector is normalized to
    have a magnitude of 1.
    """
    neighbs = boids[neigh_mask, :2] - boids[idx, :2]
    norm = np.linalg.norm(neighbs, axis=1)
    mask = norm > 0
    if np.any(mask):
        neighbs[mask] /= norm[mask].reshape(-1, 1)
    d = neighbs.mean(axis=0)
    norm_d = np.linalg.norm(d)
    if norm_d > 0:
        d /= norm_d
    # d = (boids[neigh_mask, :2] - boids[idx, :2]).mean(axis=0)
    return -d  # / ((d[0] ** 2 + d[1] ** 2) + 1)

@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """
    The code includes functions for alignment, wall avoidance, and smoothstep calculations in a boids
    simulation.
    
    :param boids: `boids` is a numpy array containing information about the boids. Each row represents a
    boid, and the columns contain different attributes of the boids. The columns are structured as
    follows:
    :type boids: np.ndarray
    :param idx: The `idx` parameter in the `alignment` function represents the index of the current boid
    for which alignment is being calculated. It is used to access the specific boid's velocity
    information from the `boids` array
    :type idx: int
    :param neigh_mask: The `neigh_mask` parameter is a boolean mask array that indicates which boids are
    considered neighbors of the current boid at index `idx`. It is used to select the relevant rows from
    the `boids` array for calculating alignment in the `alignment` function
    :type neigh_mask: np.ndarray
    :param vrange: The `vrange` parameter in the `alignment` function is a tuple that likely represents
    the range of velocities for the boids. It is used to calculate the alignment force for a boid based
    on the average velocity of its neighbors
    :type vrange: tuple
    :return: The `alignment` function returns a numpy array `a` representing the alignment force for a
    specific boid at index `idx`. The `walls` function returns a numpy array representing the forces
    applied by the walls on all boids. The `smoothstep` function returns a numpy array or float
    depending on the input type, after applying the smoothstep function to the input values.
    """
    v_mean = boids[neigh_mask, 2:4].mean(axis=0)
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a
@njit()
def walls(boids: np.ndarray, asp: float, param: int):
    """
    This function calculates the forces exerted by walls on a group of boids in a 2D space.
    
    :param boids: The `boids` parameter is expected to be a numpy array containing the positions of the
    boids. The positions are assumed to be in 2D space, where the x-coordinate is in the first column
    and the y-coordinate is in the second column
    :type boids: np.ndarray
    :param asp: The `asp` parameter in the `walls` function represents the aspect ratio of the
    environment. It is used to calculate the distance to the right wall based on the x-coordinate of the
    boids
    :type asp: float
    :param param: The `param` parameter in the `walls` function is used to determine the order of the
    potential function calculation. It is used in the calculation of the attractive and repulsive forces
    from the walls based on the distance of the boids from the walls. The higher the `param` value, the
    :type param: int
    :return: The function `walls` takes in a numpy array `boids`, a float `asp`, and an integer `param`.
    It calculates the attraction forces towards the walls based on the positions of the boids in the
    `boids` array. The function then returns a numpy array where each row contains the sum of attraction
    forces towards the left and right walls in the first column, and the sum of
    """
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]
    order = param

    a_left = 1 / (np.abs(x) + c) ** order
    a_right = -1 / (np.abs(x - asp) + c) ** order

    a_bottom = 1 / (np.abs(y) + c) ** order
    a_top = -1 / (np.abs(y - 1.) + c) ** order

    return np.column_stack((a_left + a_right, a_bottom + a_top))

@njit()
def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    """
    The function `smoothstep` calculates smooth interpolation between two edges based on the input value
    `x`.
    
    :param edge0: The `edge0` parameter in the `smoothstep` function represents the lower edge of the
    smoothstep function. It is a float value that defines the starting point of the interpolation range
    :type edge0: float
    :param edge1: The `edge1` parameter in the `smoothstep` function represents the upper edge value of
    the interpolation range. It is used to define the range within which the interpolation occurs
    :type edge1: float
    :param x: The `x` parameter represents the input value or an array of input values for which you
    want to calculate the smoothstep function. It should be a float or a NumPy array of floats
    :type x: np.ndarray | float
    :return: The `smoothstep` function returns the smoothed interpolation value based on the input
    values `edge0`, `edge1`, and `x`. The returned value is the result of applying the smoothstep
    interpolation formula: `x * x * (3.0 - 2.0 * x)`.
    """
    x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
    return x * x * (3.0 - 2.0 * x)

@njit()
def better_walls(boids: np.ndarray, asp: float, param: float):
    """
    This function calculates the influence of walls on boids' movement based on their positions and
    specified parameters.
    
    :param boids: The function `better_walls` takes in a numpy array `boids`, a float `asp`, and a float
    `param` as input parameters. The `boids` array is assumed to have two columns representing the x and
    y coordinates of points
    :type boids: np.ndarray
    :param asp: The `asp` parameter seems to be used as a scaling factor in the `better_walls` function.
    It is multiplied with `w` in the calculations for `a_left` and `a_right`. The specific purpose or
    meaning of `asp` in this context would depend on the broader context
    :type asp: float
    :param param: The `param` variable in the `better_walls` function is used as a parameter to control
    the width of the walls. It is used in calculating the smoothstep functions for the left, right,
    bottom, and top walls based on the positions of the boids
    :type param: float
    :return: The function `better_walls` takes in a numpy array `boids`, a float `asp`, and a float
    `param`. It calculates the influence of walls on the boids based on their positions and returns a
    numpy array with the calculated wall forces for each boid. The returned array contains the combined
    influence of left and right walls in the first column and the combined influence of bottom and top
    """
    x = boids[:, 0]
    y = boids[:, 1]
    w = param

    a_left = smoothstep(asp * w, 0.0, x)
    a_right = -smoothstep(asp * (1.0 - w), asp, x)

    a_bottom = smoothstep(w, 0.0, y)
    a_top = -smoothstep(1.0 - w, 1.0, y)

    return np.column_stack((a_left + a_right, a_bottom + a_top))

@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple,
             order: int,
             cnt_rely_on: int):
    """
    The function `flocking` calculates the movement of boids based on cohesion, alignment, separation,
    and wall avoidance behaviors within a specified perception range.
    
    :param boids: The `boids` parameter is expected to be a numpy array representing the positions and
    velocities of the boids in the flock. The shape of the array should be `(number_of_boids, 6)` where
    each row represents a boid with the first two columns representing the x and y positions
    :type boids: np.ndarray
    :param perception: The `perception` parameter in the `flocking` function represents the distance
    within which a boid can perceive and interact with other boids. Boids within this perception range
    are considered neighbors and influence the behavior of the current boid based on the flocking rules
    :type perception: float
    :param coeffs: The `coeffs` parameter in the `flocking` function represents the coefficients used to
    control the influence of different behaviors on the boids. These coefficients are used to adjust the
    contribution of cohesion, alignment, separation, and wall avoidance behaviors in the flocking
    simulation
    :type coeffs: np.ndarray
    :param asp: The `asp` parameter in the `flocking` function seems to be related to the aspect ratio
    of the environment or the space in which the boids are moving. It is used in the `better_walls`
    function to help calculate the influence of walls on the boids' behavior. The
    :type asp: float
    :param vrange: The `vrange` parameter in the `flocking` function likely represents the range of
    velocities that the boids can have. It is a tuple that specifies the minimum and maximum values for
    the velocity components. This range can be used to limit the velocity of the boids during the
    flocking simulation
    :type vrange: tuple
    :param order: The `order` parameter in the `flocking` function seems to be used in the
    `better_walls` function. It likely determines the order of the wall interaction calculation or some
    aspect related to how the walls are handled in the simulation. Without seeing the implementation of
    the `better_walls`
    :type order: int
    """

    N = boids.shape[0]
    DistMatrix = np.zeros((N,N))

    DistMatrix = distances(boids[:, 0:2])

    # fill the D matrix wirh perception + 1 
    # D[range(N), range(N)] = perception + 1 
    for i in prange(N):
        for j in range(N):
            DistMatrix[i, j] = perception + 1


    mask = DistMatrix < perception 
    # print(D)
    # print("---->>")
    max_cnt = cnt_rely_on
    for i in range(N):
        distance_per_leader = np.array(sorted(list(enumerate(DistMatrix[i])), key =  lambda x: x[1])) # 
        distance_per_leader[max_cnt:,1] = 100
        distance_per_leader = list(sorted(distance_per_leader, key =  lambda x: x[0]))
        for j in range(N):
            DistMatrix[i,j] = distance_per_leader[j]
    # print(D)
    mask_rely = DistMatrix < perception 



    wal = better_walls(boids, asp, order)
    for i in prange(N):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i], perception)
        a = coeffs[0] * coh + coeffs[1] * alg + \
            coeffs[2] * sep + coeffs[3] * wal[i]
        boids[i, 4:6] = a
    return {
            "mask_rely": mask_rely,
            "mask_see" : mask
            }


def periodic_walls(boids: np.ndarray, asp: float):
    """Sets the position of boids with respect to periodic walls for them to not fly away"""
    boids[:, 0:2] %= np.array([asp, 1.])



def wall_avoidance(boids: np.ndarray, asp: float):
    """Implements wall avoidance component in acceleration logic for boids"""
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])
    ax = 1. / left**2 - 1. / right**2
    ay = 1. / bottom**2 - 1. / top**2
    boids[:, 4:6] += np.column_stack((ax, ay))
