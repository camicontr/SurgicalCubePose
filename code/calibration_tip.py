from auxiliar_functions import *
import matplotlib.pyplot as plt
import pickle
import scipy


def koeda_method(p_table, p_cube, rot_cube):
    # calibration tip with koeda method.
    r_ = Rot.from_rotvec([rot_cube[0][0], rot_cube[1][0], rot_cube[2][0]])
    r_cube = r_.as_matrix()  # convert to rotation matrix from Rodriguez vector .
    p_c_rel = np.subtract(p_table, p_cube)
    p_rel = np.matmul(np.linalg.inv(r_cube), p_c_rel)
    p_c_tip = r_cube @ p_rel + p_cube  # position of tip
    return p_rel, p_c_tip


def lsq_method():
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
        ref_r_cube.extend(np.concatenate((r.as_matrix(), -np.identity(3)), axis=1))  # matrix rotation from quaternions

    opt = scipy.linalg.lstsq(ref_r_cube, np.negative(ref_p_cube))
    p_rel = opt[0][0:3]
    print("relative vector ts: ", p_rel)

    # Calculate error
    residual_vectors = np.array((ref_p_cube + ref_r_cube @ opt[0]).reshape(len(p_cube), 3))
    residual_norms = np.linalg.norm(residual_vectors, axis=1)

    return residual_norms


def run():
    lsq_method()


if __name__ == "__main__":
    run()
