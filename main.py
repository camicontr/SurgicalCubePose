from utils import *
import pandas as pd
import pickle
import os
cal = pickle.load(open("basler.pickle", "rb"))


def main(root):
    x_cube = []
    y_cube = []
    z_cube = []

    detector = Aruco()
    board = Pose(cal["mtx"], cal["dist"], 3, 19, [0, 4, 8, 16])  # in mm

    images = np.array([root + f for f in os.listdir(root) if f.endswith(".PNG")])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]

    for im in images:
        frame = cv2.imread(im)
        corners, ids = detector.detection(frame)
        detector.draw_detections(frame, corners, ids)

        r_, r_cube, t_cube = board.pose_board(corners, ids, 2)

        if r_ >= 3:
            board.draw_axis(frame, r_cube, t_cube, 10)
            x_cube.append(t_cube[0][0])
            y_cube.append(t_cube[1][0])
            z_cube.append(t_cube[2][0])

        cv2.imshow("output", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return pd.DataFrame(data={"x axis tip": x_cube, "y axis tip": y_cube, "z axis tip": z_cube})


folder = './transl/'
df = main(folder)
df.to_csv("transl.csv")
