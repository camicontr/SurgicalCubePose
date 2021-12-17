from utils import *
import pandas as pd
import pickle
import os
cal = pickle.load(open("basler.pickle", "rb"))


def main(root):
    x_tip = []
    y_tip = []
    z_tip = []

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

        if r_ > 3:
            p_tip = tip(r_cube, t_cube)
            board.draw_axis(frame, r_cube, p_tip, 5)
            board.draw_axis(frame, r_cube, t_cube, 15)
            x_tip.append(p_tip[0][0])
            y_tip.append(p_tip[1][0])
            z_tip.append(p_tip[2][0])

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.destroyAllWindows()
    return pd.DataFrame(data={"x axis tip": x_tip, "y axis tip": y_tip, "z axis tip": z_tip})


folder = './refine_tip/'
df = main(folder)
