import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import pickle
import scipy
import os
cal = pickle.load(open("basler.pickle", "rb"))  # parameters intrinsic of camera


def method_koeda(p_table, p_cube, rot_cube):
    # calibration tip with koeda method.
    r_ = Rot.from_rotvec([rot_cube[0][0], rot_cube[1][0], rot_cube[2][0]])
    r_cube = r_.as_matrix()  # convert to rotation matrix from Rodriguez vector .
    p_c_rel = np.subtract(p_table, p_cube)
    p_rel = np.matmul(np.linalg.inv(r_cube), p_c_rel)
    p_c_tip = np.dot(r_cube, p_rel) + p_cube  # position of tip
    return p_rel, p_c_tip


def method_lsq():
    # reading data for calibration
    data = pickle.load(open("data_tip.pickle", "rb"))
    qw = data[:, 0]
    qx = data[:, 1]
    qy = data[:, 2]
    qz = data[:, 3]
    p_cube = data[:, 4:7]

    # plot the data
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(p_cube[:, 0], p_cube[:, 1], p_cube[:, 2], c=p_cube[:, 2])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.title("Aruco cube Positions")
    plt.show()

    ref_p_cube = np.array(p_cube.reshape(-1, 1))
    ref_r_cube = []

    for i in range(len(p_cube)):
        r = Rot.from_quat([qw[i], qx[i], qy[i], qz[i]])  # conversion
        # matrix rotation from quaternions
        ref_r_cube.extend(np.concatenate((r.as_matrix(), -np.identity(3)), axis=1))
    opt = scipy.linalg.lstsq(ref_r_cube, np.negative(ref_p_cube))

    p_rel = opt[0][0:3]
    # p_tip = opt[0][3:6]

    # Calculate error
    residual_vectors = np.array((ref_p_cube + ref_r_cube @ opt[0]).reshape(len(p_cube), 3))  # error
    residual_norms = np.linalg.norm(residual_vectors, axis=1)
    print(residual_norms)

    mean_error = np.mean(residual_norms)
    residual_rms = rms(residual_norms)
    print("mean error: ", mean_error, "mm", "RMS error: ", residual_rms, "mm")
    print("relative vector ts: ", p_rel)


def preprocessing(root: str, m: int):
    # if type is 0 lsq method
    # root: folder with images (.bmp) of the poses of cube
    # if type is 1 koeda method
    # root: folder with image (.bmp) with table marker and cube marker

    qo = []
    qx_ = []
    qy_ = []
    qz_ = []
    tx = []
    ty = []
    tz = []

    # initialization of the classes
    detector_aruco = Aruco()
    board = Pose(cal["mtx"], cal["dist"], 3, 19, [0, 4, 8, 16])  # in mm

    images = np.array([root + f for f in os.listdir(root) if f.endswith(".PNG")])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]

    if m == 0:
        for im in images:
            frame = cv2.imread(im)
            corners, ids = detector_aruco.detection(frame)  # getting the corners and ids
            detector_aruco.draw_detections(frame, corners, ids)  # draw marker detections

            if np.all(ids is not None):
                r_, r_cube, t_cube = board.pose_board(corners, ids, 2)  # pose of cube
                rotation = Rot.from_rotvec([r_cube[0][0], r_cube[1][0], r_cube[2][0]])
                qua = rotation.as_quat()

                tx.append(t_cube[0][0])
                ty.append(t_cube[1][0])
                tz.append(t_cube[2][0])
                qo.append(qua[0])
                qx_.append(qua[1])
                qy_.append(qua[2])
                qz_.append(qua[3])

                if r_ > 3:
                    board.draw_axis(frame, r_cube, t_cube, 10)

                cv2.imshow("output", frame)
                cv2.waitKey(0)

        df = pd.DataFrame(data={"Qo": qo, "QX": qx_, "QY": qy_, "QZ": qz_, "Tx": tx, "Ty": ty, "Tz": tz})
        data = np.asarray(df)
        pickle.dump(data, open("data_tip.pickle", "wb"))
        method_lsq()

    if m == 1:
        frame = cv2.imread(images[1])
        corners, ids = detector_aruco.detection(frame)
        detector_aruco.draw_detections(frame, corners, ids)  # draw marker detections

        if np.all(ids is not None):  # If there are markers found by detector
            ret, r_table, t_table = board.pose_board(corners, ids, 3)  # pose of table
            if ret:
                board.draw_axis(frame, r_table, t_table, 10)

            ret1, r_cube, t_cube = board.pose_board(corners, ids, 2)  # pose of cube
            if ret1:
                board.draw_axis(frame, r_cube, t_cube, 10)

            if ret > 2 and ret1 > 2:
                relative, p_tip = method_koeda(t_table, t_cube, r_cube)
                print("relative vector p_rel: ", relative)

            #  Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


images_cal = './Images_cal_2/'
preprocessing(images_cal, 0)
