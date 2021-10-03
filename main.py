from utils import *
import pandas as pd
import pickle
cal = pickle.load(open("intrinsic.pickle", "rb"))


def main(root: str):
    x_tip = []
    y_tip = []
    z_tip = []

    detector = Aruco()
    board = Pose(cal["mtx"], cal["dist"], 3, 19, [0, 4, 8, 16])  # in mm

    frame_id = 0
    frame_rate = 1
    cap = cv2.VideoCapture(root)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_rate != 0:
            continue

        corners, ids = detector.detection(frame)
        detector.draw_detections(frame, corners, ids)

        r_, r_cube, t_cube = board.pose_board(corners, ids, 2)
        r1_, r_cube1, t_cube1 = board.pose_board(corners, ids, 0)
        if r_ > 3:
            p_tip = Tip(r_cube, t_cube)
            board.draw_axis(frame, r_cube, p_tip, 5)
            board.draw_axis(frame, r_cube, t_cube, 15)
            x_tip.append(p_tip[0][0])
            y_tip.append(p_tip[1][0])
            z_tip.append(p_tip[2][0])

        if r1_ > 3:
            p_tip = Tip(r_cube1, t_cube1)
            board.draw_axis(frame, r_cube1, p_tip, 5)
            board.draw_axis(frame, r_cube1, t_cube1, 15)

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.destroyAllWindows()
    return pd.DataFrame(data={"x axis tip": x_tip, "y axis tip": y_tip, "z axis tip": z_tip})


video = "/Users/pc/PycharmProjects/tg/main.mov"
df = main(video)
# df.to_excel(excel_writer="animation_blender.xlsx")
