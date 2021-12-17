import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import pickle
import scipy
import os
cal = pickle.load(open("intrinsic.pickle", "rb"))  # parameters intrinsic of camera


def Koeda(p_table, p_cube, rot_cube):
    # calibration tip with koeda method.
    r_ = Rot.from_rotvec([rot_cube[0][0], rot_cube[1][0], rot_cube[2][0]])
    R_cube = r_.as_matrix()  # convert to rotation matrix from Rodriguez vector .
    p_c_rel = np.subtract(p_table, p_cube)
    p_rel = np.matmul(np.linalg.inv(R_cube), p_c_rel)
    p_c_tip = np.dot(R_cube, p_rel) + p_cube  # position of tip

    return p_rel, p_c_tip


def lsq():
    # reading data for calibration
    data = pickle.load(open("data_tip.pickle", "rb"))
    Qw = data[:, 0]
    Qx = data[:, 1]
    Qy = data[:, 2]
    Qz = data[:, 3]
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

    refP_cube = np.array(p_cube.reshape(-1, 1))
    refR_cube = []

    for i in range(len(p_cube)):
        r = Rot.from_quat([Qw[i], Qx[i], Qy[i], Qz[i]])  # conversion
        # matrix rotation from quaternions
        refR_cube.extend(np.concatenate((r.as_matrix(), -np.identity(3)), axis=1))

    opt = scipy.linalg.lstsq(refR_cube, np.negative(refP_cube))

    p_rel = opt[0][0:3]
    # p_tip = optiT[0][3:6]

    # Calculate error
    residual_vectors = np.array((refP_cube + refR_cube @ opt[0]).reshape(len(p_cube), 3))  # error
    residual_norms = np.linalg.norm(residual_vectors, axis=1)

    mean_error = np.mean(residual_norms)
    residual_rms = rms(residual_norms)
    print("mean error: ", mean_error, "mm", "RMS error: ", residual_rms, "mm")
    print("relative vector ts: ", p_rel)


def preprocessing(root: str, m: int):
    # if type is 0 lsq method
    # root: folder with images (.bmp) of the poses of cube
    # if type is 1 koeda method
    # root: folder with image (.bmp) with table marker and cube marker

    Qo = []
    QX = []
    QY = []
    QZ = []
    tx = []
    ty = []
    tz = []

    # initialization of the classes
    detector_aruco = Aruco()
    board = Pose(cal["mtx"], cal["dist"], 3, 19, [0, 4, 8, 16])  # in mm

    images = np.array([root + f for f in os.listdir(root) if f.endswith(".bmp")])
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
                Qo.append(qua[0])
                QX.append(qua[1])
                QY.append(qua[2])
                QZ.append(qua[3])

                if r_ > 3:
                    board.draw_axis(frame, r_cube, t_cube, 10)

                cv2.imshow("output", frame)
                cv2.waitKey(0)

        df = pd.DataFrame(data={"Qo": Qo, "QX": QX, "QY": QY, "QZ": QZ, "Tx": tx, "Ty": ty, "Tz": tz})
        data = np.asarray(df)
        pickle.dump(data, open("data_tip.pickle", "wb"))
        lsq()

    if m == 1:
        frame = cv2.imread(images[1])  # only a image
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
                relative, p_tip = Koeda(t_table, t_cube, r_cube)
                print("relative vector p_rel: ", relative)

            #  Display the resulting frame
            cv2.imshow('frame', frame)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


images_cal = '/Users/pc/PycharmProjects/tg/cal/'
preprocessing(images_cal, 1)
