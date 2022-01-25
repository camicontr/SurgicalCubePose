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


def tip(r_vec, t_vec):
    p_k_rel = np.array([20.62984144,
                        -95.29293926,
                        -27.32301939])
    p_k_rel = p_k_rel.reshape((3, 1))

    p_c_knife = np.array([t_vec[0][0],
                          t_vec[1][0],
                          t_vec[2][0]])
    p_c_knife = p_c_knife.reshape((3, 1))

    r = Rot.from_rotvec([r_vec[0][0], r_vec[1][0], r_vec[2][0]])
    R_c_knife = r.as_matrix()  # from vector Rodrigue's to rotation matrix
    return p_c_knife + R_c_knife @ p_k_rel


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


def get_point_dist(points, plane):
    # return: 1d array of size N with the distance of the points respect the plane
    dists = np.abs(points @ plane) / np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
    return dists


def plot_plane(a, b, c, d):
    # plot plane with parameters [a, b, c, d]
    xx, yy = np.mgrid[-80:0, 20:100]
    return xx, yy, (-d - a * xx - b * yy) / c


def radius_sphere(points, center):
    # get the radius of sphere
    radius = []
    for point in points:
        radius.append(np.sqrt((point[0]-center[0]) ** 2 + (point[1]-center[1]) ** 2 + (point[2]-center[2]) ** 2))
    return radius


def dist_points(a, b, p):
    return np.linalg.norm(np.cross(b-a, a-p))/np.linalg.norm(b-a)


def rms(dist):
    # error rms
    return np.sqrt(np.mean(dist ** 2))


def mae(dist):
    # error mae
    return np.mean(np.abs(dist))
