import matplotlib.pyplot as plt
from modules.utils import *
import pandas as pd
import pickle
import scipy
import os
cal = pickle.load(open("intrinsic.pickle", "rb"))  # parameters intrinsic of camera


def processing_tip(root: str, save):
    # input: library with images (.bmp) of the poses
    # with the fixed tip different poses of the cube are obtained and the data is saved
    # save == 1, for save the data
    Qo = []
    QX = []
    QY = []
    QZ = []
    tx = []
    ty = []
    tz = []

    # initialization of the classes
    detector_aruco = Aruco()
    board_pose = Pose(cal["mtx"], cal["dist"], 3, 19, [0, 4, 8, 12])  # in mm

    images = np.array([root + f for f in os.listdir(root) if f.endswith(".bmp")])
    order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
    images = images[order]

    for im in images:
        frame = cv2.imread(im)
        corners, ids = detector_aruco.detection(frame)  # getting the corners and ids
        detector_aruco.draw_detections(frame, corners, ids)  # draw marker detections

        if np.all(ids is not None):
            r_, r_cube, t_cube = board_pose.pose_board(corners, ids, 2)  # pose of cube
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
                board_pose.draw_axis(frame, r_cube, t_cube, 10)

            cv2.imshow("output", frame)
            cv2.waitKey(0)

    if save == 1:
        df = pd.DataFrame(data={"Qo": Qo, "QX": QX, "QY": QY, "QZ": QZ, "Tx": tx, "Ty": ty, "Tz": tz})
        data = np.asarray(df)
        pickle.dump(data, open("relative.pickle", "wb"))


# get the pose data to calibrate the cube
# images_cal = 'folder path with calibration images'
# processing_tip(images_cal, 1)

# reading data for calibration
tip = pickle.load(open("relative.pickle", "rb"))
Qw = tip[:, 0]
Qx = tip[:, 1]
Qy = tip[:, 2]
Qz = tip[:, 3]
Ts = tip[:, 4:7]

# plot the data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(Ts[:, 0], Ts[:, 1], Ts[:, 2], c=Ts[:, 2])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.title("Aruco cube Positions")
plt.show()

refTs = np.array(Ts.reshape(-1, 1))
refRs = []

for measurementOP in range(len(Ts)):
    r = Rot.from_quat([Qw[measurementOP], Qx[measurementOP], Qy[measurementOP], Qz[measurementOP]])
    refRs.extend(np.concatenate((r.as_matrix(), -np.identity(3)), axis=1))

optiT = scipy.linalg.lstsq(refRs, np.negative(refTs))

t_s = optiT[0][0:3]
t_p = optiT[0][3:6]

# Calculate error
residual_vectors = np.array((refTs + refRs @ optiT[0]).reshape(len(Ts), 3))  # error
residual_norms = np.linalg.norm(residual_vectors, axis=1)

mean_error = np.mean(residual_norms)
residual_rms = rms(residual_norms)
print("mean error: ", mean_error, "mm", "RMS error: ", residual_rms, "mm")
print("relative vector ts: ", t_s)
