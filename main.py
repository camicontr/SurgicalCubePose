from auxiliar_functions import *
from classes import *
import pandas as pd
import numpy as np
import pickle
import cv2
import os
cal = pickle.load(open("basler.pickle", "rb"))


def main(root):
    x_tip = []
    y_tip = []
    z_tip = []

    detector = System()
    cube = CubeAruco(cal["mtx"], cal["dist"], 3, 19, [0, 4, 8, 16])  # in mm

    images = np.array([root + f for f in os.listdir(root) if f.endswith(".PNG")])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]

    for im in images:
        frame = cv2.imread(im)
        corners, ids = detector.detection(frame)
        detector.draw_detections(frame, corners, ids)
        r_, r_cube, t_cube = cube.cube_pose(corners, ids, 2)

        if r_ > 3:
            p_tip = tip(r_cube, t_cube)
            cube.draw_axis(frame, r_cube, p_tip, 5)
            cube.draw_axis(frame, r_cube, t_cube, 10)
            x_tip.append(p_tip[0][0])
            y_tip.append(p_tip[1][0])
            z_tip.append(p_tip[2][0])

        cv2.imshow("output", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return pd.DataFrame(data={"x axis tip": x_tip, "y axis tip": y_tip, "z axis tip": z_tip})


def run():
    folder = './images/'
    df = main(folder)
    df.to_csv("test.csv")


if __name__ == "__main__":
    run()
