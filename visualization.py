from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import circle_fit


def transl_analysis():
    df = pd.read_csv("./transl.csv")
    xyz = df.iloc[:, 1:4]
    xyz = np.asarray(xyz)

    # Calculate the mean of the points, i.e. the 'center' of the cloud
    data_mean = xyz.mean(axis=0)
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(xyz - data_mean)

    line_pts = vv[0] * np.mgrid[-6:6:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    line_pts += data_mean

    # error calculation
    dis = []
    for i in range(0, len(xyz)):
        dis.append(dist_seg(line_pts[0], line_pts[1], xyz[i]))
    dis = np.asarray(dis)
    print("mean error: ", mae(dis), "rms error fit line: ", rms(dis))

    # plot line
    fig = plt.figure(figsize=(6, 12))
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter3D(*xyz.T)
    ax.plot3D(*line_pts.T)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax = fig.add_subplot(212)
    plt.title("variación en z")
    plt.scatter(xyz[:, 0], xyz[:, 2], color="blue")
    plt.xlabel("eje x (mm)")
    plt.ylabel("eje z (mm)")
    plt.axis('equal')
    plt.show()


def plane_analysis(n_example):
    # read data
    df = pd.read_csv("./circle{n}.csv".format(n=n_example))
    xyz = df.iloc[:, 1:4]
    xyz = np.asarray(xyz)
    # homogeneous coord
    hm = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)

    # plane fit
    p = fit_plane_LSE(hm)

    # calculate error:
    dists = get_point_dist(hm, p)
    print("mean error of plane fit", mae(dists), "rms error of plane fit", rms(dists))

    # plot plane
    fig = plt.figure(figsize=(6, 12))
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter3D(xyz.T[0], xyz.T[1], xyz.T[2], color="black", label="points")
    xx, yy, zz = plot_plane(p[0], p[1], p[2], p[3])
    ax.plot_surface(xx, yy, zz, color='blue', alpha=0.5)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()

    # dimension reduction
    pca = PCA(n_components=2, svd_solver="full")
    xyz_pca = pca.fit_transform(xyz)
    # circle fit
    data_c = circle_fit.least_squares_circle(xyz_pca)
    r = data_c[2]  # radius from circle fit
    r_ = np.sqrt((xyz_pca[:, 0]) ** 2 + (xyz_pca[:, 1]) ** 2)
    print("mean error: ", mae(r - r_), "rms error fit circle: ", rms(r - r_), "estimate radius in mm : ", r)

    ax = fig.add_subplot(212)
    plt.scatter(xyz_pca[:, 0], xyz_pca[:, 1], color="black")
    plt.title('Proyección en el plano ajustado con centro en (0, 0, 0)')
    plt.axis('equal')
    plt.show()


def sphere_analysis(n_example):
    # read data
    df = pd.read_excel("/Users/pc/PycharmProjects/tg/sphere/sphere{n}.xlsx".
                       format(n=n_example))
    xyz = df.iloc[:, 1:4]
    xyz = np.asarray(xyz)

    # plot sphere
    fig2 = plt.figure(2, figsize=(8, 8))
    ax = fig2.add_subplot(projection='3d')
    ax.scatter3D(xyz.T[0], xyz.T[1], xyz.T[2], color="black", label="points")

    # Create a sphere
    """
    r = parameters[3]
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r*np.sin(phi)*np.cos(theta) + parameters[0]
    y = r*np.sin(phi)*np.sin(theta) + parameters[1]
    z = r*np.cos(phi) + parameters[2]
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    """
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('fit sphere example: sphere{}'.format(n_example))
    plt.show()
    # r_calc = radius_sphere(xyz, parameters)
    # r_calc = np.asarray(r_calc)
    # print("El error rms es de {:.4f} mm".format(rms(r_calc - parameters[3])))
