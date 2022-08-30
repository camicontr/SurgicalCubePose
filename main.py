from code.auxiliar_functions import *
from code.setting import *
import pandas as pd
import numpy as np
import pickle
import cv2
import os


cal = pickle.load(open("basler.pickle", "rb"))


def main(root, mode:str):
    detector = System()
    cube = CubeAruco(cal["mtx"], cal["dist"], 3, 19, [0, 4, 8, 16])  # in mm

    images = np.array([root + f for f in os.listdir(root) if f.endswith(".PNG")])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]

    x_tip, y_tip, z_tip = [], [], []
    qo, qx_, qy_, qz_, tx, ty, tz = [], [], [], [], [], [], []

    for im in images:
        frame = cv2.imread(im)
        corners, ids = detector.detection(frame)
        detector.draw_detections(frame, corners, ids)
        r_, r_cube, t_cube = cube.cube_pose(corners, ids, 2)
     
        if mode == "calibration":
            rotation = Rot.from_rotvec([r_cube[0][0], r_cube[1][0], r_cube[2][0]])
            qua = rotation.as_quat()
            cube.draw_axis(frame, r_cube, t_cube, 10)
            tx.append(t_cube[0][0])
            ty.append(t_cube[1][0])
            tz.append(t_cube[2][0])
            qo.append(qua[0])
            qx_.append(qua[1])
            qy_.append(qua[2])
            qz_.append(qua[3])

        if mode == "eval":
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
    
    df = pd.DataFrame(data={
                "Qo": qo,
                "QX": qx_,
                "QY": qy_,
                "QZ": qz_,
                "Tx": tx,
                "Ty": ty,
                "Tz": tz})
    if df.empty == False:
        data = np.asarray(df)
        pickle.dump(data, open("data_tip.pickle", "wb"))

    return pd.DataFrame(data={
        "x axis tip": x_tip,
        "y axis tip": y_tip,
        "z axis tip": z_tip})


def run():
    folder = './data/refine_tip_1/'
    df = main(folder, "calibration")


if __name__ == "__main__":
    run()
