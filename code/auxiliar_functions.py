from scipy.spatial.transform import Rotation as Rot
from datetime import datetime
import numpy as np


def execution_time(func):
    def wrapper(*args, **kwargs):
        initial_time = datetime.now()
        func(*args, **kwargs)
        final_time = datetime.now()
        time_elapsed = final_time - initial_time
        print(f'execution time {time_elapsed.total_seconds()} seconds')
    return wrapper


def error(func):
    def wrapper(*args, **kwargs):
        f = func(*args, **kwargs)
        print(f'rms error: {np.sqrt(np.mean(f ** 2))} mm')
        print(f'mae error: {np.mean(np.abs(f))} mm')
    return wrapper


# ------------------------------- Auxiliary function for tip transform -------------------------------
def tip(r_vec, t_vec):
    p_k_rel = np.array([20.42962901,
                        139.03692834,
                        22.34037493]
                        )
    p_k_rel = p_k_rel.reshape((3, 1))

    p_c_knife = np.array([t_vec[0][0],
                          t_vec[1][0],
                          t_vec[2][0]]
                        )
    p_c_knife = p_c_knife.reshape((3, 1))

    r = Rot.from_rotvec([r_vec[0][0], r_vec[1][0], r_vec[2][0]])
    R_c_knife = r.as_matrix()  # from vector Rodrigue's to rotation matrix
    return p_c_knife + R_c_knife @ p_k_rel


# ------------------------------- Auxiliary functions for fit plane -------------------------------
def svd(A):
    u, s, vh = np.linalg.svd(A)
    S = np.zeros(A.shape)
    S[:s.shape[0], :s.shape[0]] = np.diag(s)
    return u, S, vh


def fit_plane_lse(points):
    # points: Nx4 homogeneous 3d points
    # return: 1d array of four elements [a, b, c, d] of
    assert points.shape[0] >= 3  # at least 3 points needed
    U, S, Vt = svd(points)
    null_space = Vt[-1, :]
    return null_space


def plot_plane(a, b, c, d):
    # plot plane with parameters [a, b, c, d]
    xx, yy = np.mgrid[-80:0, 20:100]
    return xx, yy, (-d - a * xx - b * yy) / c


def get_point_dist(points, plane):
    # return: 1d array of size N with the distance of the points respect the plane
    dists = np.abs(points @ plane) / np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
    return dists


# ----------------------------- Auxiliary functions for fit sphere ---------------------------------
def fit_sphere(xyz):
    #   Assemble the A matrix
    spX = xyz[:, 0]
    spY = xyz[:, 1]
    spZ = xyz[:, 2]
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX*2
    A[:, 1] = spY*2
    A[:, 2] = spZ*2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residuals, rank, sing_val = np.linalg.lstsq(A, f, rcond=-1)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = np.sqrt(t)

    return radius, C[0], C[1], C[2]


# ----------------------------- Auxiliary functions for fit line -----------------------------------
def dist_points(a, b, p):
    return np.linalg.norm(np.cross(b - a, a - p)) / np.linalg.norm(b - a)
