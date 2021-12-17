from scipy.spatial.transform import Rotation as Rot
import numpy as np
import cv2


class Pose:
    def __init__(self, c_mtx, c_dist, marker_sep, marker_size, marker_ids):
        self.c_mtx = c_mtx
        self.c_dist = c_dist
        self.marker_sep = marker_sep
        self.marker_size = marker_size
        self.marker_ids = marker_ids

    @staticmethod
    def _dict():
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    def _create_board(self):
        return [cv2.aruco.GridBoard_create(2, 2, self.marker_size, self.marker_sep, Pose._dict(), j) for j in
                self.marker_ids]

    def pose_board(self, corners, ids, num_board):
        board = self._create_board()

        if corners is not None:
            r_vec_, t_vec_, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.c_mtx, self.c_dist)
            r_, r_vector, t_vector = cv2.aruco.estimatePoseBoard(corners, ids, board[num_board], self.c_mtx,
                                                                 self.c_dist, r_vec_, t_vec_)
            if t_vector is None:
                r_vector = np.nan * np.ones((1, 1, 3))
                t_vector = np.nan * np.ones((1, 1, 3))
        return r_, r_vector, t_vector

    def pose_single(self, corners):
        # estimate pose of aruco
        if corners is not None:
            r_vec_, t_vec_, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength=self.marker_size,
                                                                    cameraMatrix=self.c_mtx, distCoeffs=self.c_dist)
            return r_vec_, t_vec_

    def draw_axis(self, frame, r_vec, t_vec, length):

        return cv2.aruco.drawAxis(frame, self.c_mtx, self.c_dist, r_vec, t_vec, length)


class Aruco:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # Use 4x4 dictionary to find markers
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshWinSizeMin = 3  # default 3
        self.parameters.adaptiveThreshWinSizeMax = 30  # default 23
        self.parameters.adaptiveThreshWinSizeStep = 10  # default is 10
        self.parameters.adaptiveThreshConstant = 7  # default 7
        self.parameters.minMarkerPerimeterRate = 0.05  # default  0.005
        # self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # self.parameters.cornerRefinementWinSize = 1
        # self.parameters.cornerRefinementMinAccuracy = 0.001
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.color = (255, 0, 0)

    def detection(self, frame):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)
        return corners, ids

    def draw_detections(self, frame, corners, ids):
        return cv2.aruco.drawDetectedMarkers(frame, corners, ids, self.color)


def tip(r_vec, t_vec):
    # input: rotation Rodriguez and translation vector from cube
    # output: position of tip
    p_k_rel = np.array([20.40362974,
                        -98.25048598,
                        -30.57315047])
    p_k_rel = p_k_rel.reshape((3, 1))

    p_c_knife = np.array([t_vec[0][0],
                          t_vec[1][0],
                          t_vec[2][0]])
    p_c_knife = p_c_knife.reshape((3, 1))

    r = Rot.from_rotvec([r_vec[0][0], r_vec[1][0], r_vec[2][0]])
    R_c_knife = r.as_matrix()  # from vector Rodrigue's to rotation matrix
    return p_c_knife + R_c_knife @ p_k_rel


def plot_plane(a, b, c, d):
    # plot plane with parameters [a, b, c, d]
    xx, yy = np.mgrid[-80:80, 20:120]
    return xx, yy, (-d - a * xx - b * yy) / c


def svd(A):
    u, s, vh = np.linalg.svd(A)
    S = np.zeros(A.shape)
    S[:s.shape[0], :s.shape[0]] = np.diag(s)
    return u, S, vh


def inverse_sigma(S):
    inv_S = S.copy().transpose()
    for i in range(min(S.shape)):
        if abs(inv_S[i, i]) > 0.00001:
            inv_S[i, i] = 1.0 / inv_S[i, i]
    return inv_S


def svd_solve(A, b):
    U, S, Vt = svd(A)
    inv_S = inverse_sigma(S)
    svd_solution = Vt.transpose() @ inv_S @ U.transpose() @ b
    return svd_solution


def fit_plane_LSE(points):
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


def radius_sphere(points, center):
    # get the radius of sphere
    radius = []
    for point in points:
        radius.append(np.sqrt((point[0]-center[0]) ** 2 + (point[1]-center[1]) ** 2 + (point[2]-center[2]) ** 2))
    return radius


def rms(dist):
    # error rms
    return np.sqrt(np.mean(dist ** 2))


def mae(dist):
    # error mae
    return np.mean(np.abs(dist))
