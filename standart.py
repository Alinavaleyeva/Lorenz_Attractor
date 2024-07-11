from Lorenz_attractor import simulation, dormand_prince8, weights_rkdp8, compute_deviation
from fractions import Fraction
import numpy as np


def compute_standard():
    st_dt = Fraction(1, 1000)
    st_time = 5
    st_deviance = 0.001
    xyzs1 = simulation(st_dt, dormand_prince8, weights_rkdp8, 14, st_time)
    cur_dt = st_dt
    while True:
        xyzs2 = simulation(cur_dt/2, dormand_prince8, weights_rkdp8, 14, st_time)
        cur_dev = compute_deviation(xyzs1, xyzs2, 2)
        if round(cur_dev, 2) <= st_deviance:
            standard = xyzs2
            res_dt = cur_dt
            break
        xyzs1 = xyzs2
        cur_dt /= 2

    np.save("standard.txt", standard)
    return res_dt

if __name__ == '__main__':
    print(compute_standard())