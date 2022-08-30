import cv2


class System:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshWinSizeMin = 3  # default 3
        self.parameters.adaptiveThreshWinSizeMax = 30  # default 23
        self.parameters.adaptiveThreshWinSizeStep = 10  # default is 10
        self.parameters.minMarkerPerimeterRate = 0.05  # default  0.005
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.color = (255, 0, 0)


    def detection(self, frame):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.aruco_dict,
            parameters=self.parameters
            )
        return corners, ids


    def draw_detections(self, frame, corners, ids):
        return cv2.aruco.drawDetectedMarkers(
            frame,
            corners,
            ids,
            self.color)


class CubeAruco(System):
    def __init__(self, c_mtx, c_dist, marker_sep, marker_size, marker_ids):
        super().__init__()
        self.c_mtx = c_mtx
        self.c_dist = c_dist
        self.marker_sep = marker_sep
        self.marker_size = marker_size
        self.marker_ids = marker_ids


    def boards(self):
        return [cv2.aruco.GridBoard_create(
            2,
            2,
            self.marker_size,
            self.marker_sep,
            self.aruco_dict,
            j
            ) for j in self.marker_ids]


    def cube_pose(self, corners, ids, face):
        r_, r_vector, t_vector = None, None, None
        board = self.boards()

        if corners is not None:
            r_vec_, t_vec_, _ = cv2.aruco.estimatePoseSingleMarkers(
             corners,
             self.marker_size,
             self.c_mtx,
             self.c_dist
             )
            r_, r_vector, t_vector = cv2.aruco.estimatePoseBoard(
                corners,
                ids,
                board[face],
                self.c_mtx,
                self.c_dist,
                r_vec_,
                t_vec_
                )
        return r_, r_vector, t_vector


    def draw_axis(self, frame, r_vec, t_vec, length):
        return cv2.drawFrameAxes(
            frame,
            self.c_mtx,
            self.c_dist,
            r_vec,
            t_vec,
            length
            )
