import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction


s = 10
b = 2.667
dt = Fraction(1, 1000)
time = 30
deviance = 0.1
standard_dt = Fraction(1, 64000)
#standard_dt = Fraction(1, 512000)

rk4 = [
    np.array([]),
    np.array([0.5]),
    np.array([0, 0.5]),
    np.array([0, 0, 1])
]
weights_rk4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])

dormand_prince8 = [  np.array([]),
                    np.array([0.055555556]),
                    np.array([0.020833333, 0.0625]),
                    np.array([0.03125, 0, 0.09375,]),
                    np.array([0.3125, 0, -1.171875, 1.171875]),
                    np.array([0.0375, 0, 0, 0.1875, 0.15]),
                    np.array([0.047910137, 0, 0, 0.112248713, -0.025505674, 0.012846824]),
                    np.array([0.01691799, 0, 0, 0.387848278, 0.03597737, 0.196970214, -0.172713852]),
                    np.array([0.069095753, 0, 0, -0.634247977, -0.161197575, 0.138650309, 0.940928614, 0.211636326]),
                    np.array([0.183556997, 0, 0, -2.468768084, -0.291286888, -0.02647302, 2.847838764, 0.281387331, 0.1237449]),
                    np.array([-1.215424817, 0, 0, 16.672608666, 0.915741828, -6.056605804, -16.003573594, 14.849303086, -13.371575735, 5.134182648]),
                    np.array([0.258860916, 0, 0, -4.774485785, -0.435093014, -3.049483332, 5.57792004, 6.15583159, -5.062104587, 2.193926173, 0.134627999]),
                    np.array([0.8224276, 0, 0, -11.658673257, -0.757622117, 0.713973588, 12.075774987, -2.127659114, 1.990166207, -0.234286472, 0.175898578, 0]),
                    np.array([0.041747491, 0, 0, 0, 0, -0.055452329, 0.239312807, 0.703510669, -0.759759614, 0.660563031, 0.158187483, -0.238109539, 0.25])
]

weights_rkdp8 = np.array([0.041747491, 0, 0, 0, 0, -0.055452329, 0.239312807, 0.703510669, -0.759759614, 0.660563031, 0.158187483, -0.238109539, 0.25, 0])

#arr_r = [0.3, 1.8, 3.7, 10, 16, 24.06, 28, 100]
r = float(input())


def lorenz_x(xyz):
    x_dot = s*(xyz[1] - xyz[0])
    return x_dot


def lorenz_y(xyz):
    y_dot = r * xyz[0] - xyz[1] - xyz[0] * xyz[2]
    return y_dot


def lorenz_z(xyz):
    z_dot = xyz[0] * xyz[1] - b * xyz[2]
    return z_dot


lorenz = [lorenz_x, lorenz_y, lorenz_z]


def rk_step(dt: Fraction, xyz, s: int, method, weights):
    k = np.zeros((s, 3))
    for i in range(s):
        for j in range(3):
            k[i, j] = lorenz[j](xyz + float(dt) * (method[i] @ k[:i, j]))
    step = xyz + float(dt) * (weights @ k)
    return step


def simulation(dt, method, weights, s, time):
    xyzs = np.zeros((int(time / dt) + 1, 3))  # Need one more for the initial values
    xyzs[0] = (10., 10., 10.)  # Set initial values
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    curr_time = dt
    i = 1
    while curr_time <= time:
        xyzs[i] = rk_step(dt, xyzs[i - 1], s, method, weights)
        curr_time += dt
        i += 1
    return xyzs


def compute_deviation(xyz1, xyz2, step):
    d_xyz = np.abs(xyz2[::step] - xyz1)
    return np.max(d_xyz)


def compute_error(standard):
    xyz = simulation(dt, rk4, weights_rk4, 4, time)
    accurate_xyz = standard[::int(standard.shape[0] / xyz.shape[0])]
    diff = np.abs(accurate_xyz.shape[0] - xyz.shape[0])
    accurate_xyz = accurate_xyz[diff:] if accurate_xyz.shape > xyz.shape else xyz[diff:]
    d_xyz = accurate_xyz - xyz
    d_xyz /= np.maximum(xyz, accurate_xyz)
    global_error = np.sqrt(np.mean(d_xyz**2))
    return global_error


def show_pic(r: float):
    xyzs = simulation(dt, rk4, weights_rk4, 4, time)

    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    ax.text(9, -10,0, f'r={r}')
    plt.show()

if __name__ == '__main__':
    show_pic(r)
    standard = np.load("standard.txt.npy")
    print(compute_error(standard))
