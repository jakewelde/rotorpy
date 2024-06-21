import math

import numpy as np
import osqp
from matplotlib import colors
from scipy import sparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def plot_vector(ax, xs, ys, zs, xl, yl, zl, c):
    xe = xs + xl
    ye = ys + yl
    ze = zs + zl
    ax.plot3D([xs, xe], [ys, ye], [zs, ze], c)
    ax.quiver(xs, ys, zs, xl, yl, zl, color=c, arrow_length_ratio=0.1)
    label_position = [(xs + xe) / 2, (ys + ye) / 2, (zs + ze) / 2]
    ax.text(label_position[0], label_position[1], label_position[2],
            str(round(math.sqrt(xl ** 2 + yl ** 2 + zl ** 2), 3)), color=c)


class ControlAllocation:
    def __init__(self, r, a, kQ, phi, min_f, max_f):
        # np.set_printoptions(precision=3, suppress=True, edgeitems=100, linewidth=100)
        self.r = r
        self.a = a
        self.kQ = kQ
        self.phi = phi
        self.min_f = min_f
        self.max_f = max_f
        self.B = []
        self.C = []
        self.F_d = []
        self.T_d = []
        self.pyr_lines = []
        self.prob = None
        self.res = None

        self.solve_prob(None, None)

    # Constants
    EPSILON = 0.01
    ALPHA = 1.0
    N = 8  # Number of Pyramid Sides

    def solve_prob(self, F_d, T_d):
        if F_d is None and T_d is None:
            self.prob = osqp.OSQP()
            self.compute_b()
            self.compute_c()

            self.B = np.matrix(self.B)
            self.C = np.matrix(self.C)

            print("B=\n", self.B)
            print("C=\n", self.C)
            print("CB=\n", self.C * self.B)

        self.F_d = F_d
        self.T_d = T_d
        # self.minimize_f()  # Cost Function- minimize F
        self.square_error_ft()  # Cost Function- minimize (F - F_des)^2 + (T - T_des)^2

        self.res = self.prob.solve()
        print(self.res.x)

    def pyramid_constraint(self):
        if self.phi > 0:
            A = np.zeros((len(self.a) * (self.N + 1), 3 * len(self.a)))
            theta = 2 * math.pi / self.N
            for i in range(len(self.a)):
                p_n = math.sqrt(self.a[i][0] ** 2 + self.a[i][1] ** 2 + self.a[i][2] ** 2)
                phi_n = math.acos(self.a[i][2] / p_n) + self.phi
                for j in range(1, int(self.N) + 1):
                    theta_n = j * theta
                    if self.a[i][0] != 0:
                        theta_n = math.atan(self.a[i][1] / self.a[i][0]) + j * theta
                    self.pyr_lines.append(
                        [p_n * math.sin(phi_n) * math.cos(theta_n) * self.max_f, p_n * math.sin(phi_n) *
                         math.sin(theta_n) * self.max_f, p_n * math.cos(phi_n) * self.max_f])

            for j in range(0, len(self.pyr_lines)):
                a = np.cross(self.pyr_lines[0], self.pyr_lines[j])
                if j + 1 < len(self.pyr_lines):
                    a = np.cross(self.pyr_lines[j + 1], self.pyr_lines[j])
                for k in range(3):
                    A[j][3 * int(j / self.N) + k] = a[k]

            U = np.zeros(len(self.a) * self.N + 4)
            L = np.full(len(self.a) * self.N + 4, -float('inf'))
            for i in range(len(self.a)):
                L[-len(self.a) + i] = self.min_f
                U[-len(self.a) + i] = self.max_f
                A[-len(self.a) + i][i * 3 + 2] = 1

        else:
            A = np.zeros((len(self.a) * 3, len(self.a) * 3))
            for i in range(len(A)):
                A[i][i] = 1
            U = np.zeros((len(A)))
            L = np.zeros((len(A)))
            for i in range(len(self.a)):
                L[i * 3 + 2] = self.min_f
                U[i * 3 + 2] = self.max_f

        # print("A=\n", A)
        # print("U=\n", U)
        # print("L=\n", L)
        return sparse.csc_matrix(A), L, U

    def square_error_ft(self):
        if self.F_d is not None and self.T_d is not None:
            q = -2 * self.B.transpose() * self.C.transpose() * np.matrix(np.array([self.F_d + self.T_d])).transpose()
            # print("q=\n", q)
            self.prob.update(q=q)
        else:
            P = 2 * self.B.transpose() * self.C.transpose() * self.C * self.B
            ep = np.zeros((len(P), len(P)))
            for i in range(len(P)):
                ep[i][i] = self.EPSILON
            # print("ep=\n", ep)
            P += ep
            # print("P=\n", P)
            A, L, U = self.pyramid_constraint()
            self.prob.setup(P=sparse.csc_matrix(P), A=A, l=L, u=U, alpha=self.ALPHA)

    def minimize_f(self):
        if self.F_d is not None and self.T_d is not None:
            e = np.array(self.F_d + self.T_d)
            self.prob.update(l=e, u=e)
        else:
            P = np.zeros((len(self.C[0]), len(self.C[0])))
            for i in range(len(self.C[0])):
                P[i][i] = 2
            P = sparse.csc_matrix(P)
            q = np.zeros((len(self.C[0]), 1))

            A = sparse.csc_matrix(self.C) * sparse.csc_matrix(self.B)
            self.prob.setup(sparse.csc_matrix(P), q, A, np.zeros(len(self.C)), alpha=self.ALPHA)

    def compute_b(self):
        self.B = np.zeros((len(self.a) * 3, len(self.a) * 3))
        if self.phi > 0:
            for i in range(0, len(self.a)):
                for j in range(0, 3):
                    self.B[i * 3 + j][i * 3 + j] = 1
        else:
            for i in range(0, len(self.a)):
                for j in range(0, 3):
                    self.B[i * 3 + j][i * 3 + j] = self.a[i][j]
        B = np.matrix(self.B)

    def compute_c(self):
        self.C = np.zeros((6, len(self.r) * 3))
        for i in range(0, len(self.C[0])):
            self.C[i % 3][i] = 1
            if i % 3 == 0:
                self.C[4][i] = self.r[int(i / 3)][2]
                self.C[5][i] = -self.r[int(i / 3)][1]
            if i % 3 == 1:
                self.C[3][i] = -self.r[int(i / 3)][2]
                self.C[5][i] = self.r[int(i / 3)][0]
            if i % 3 == 2:
                self.C[3][i] = self.r[int(i / 3)][1]
                self.C[4][i] = -self.r[int(i / 3)][0]
                self.C[5][i] = self.kQ[int(i / 3)]
        C = np.matrix(self.C)

    def plot_forces(self):
        if self.res.x is not None:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            plot_vector(ax, 0, 0, 0, self.F_d[0], self.F_d[1], self.F_d[2], 'red')
            plot_vector(ax, 0, 0, 0, self.T_d[0], self.T_d[1], self.T_d[2], 'green')

            # plot_vector(ax, 0, 0, 0, self.res.x[0], self.res.x[1], self.res.x[2], 'red')
            # plot_vector(ax, 0, 0, 0, self.T_d[0], self.T_d[1], self.T_d[2], 'green')

            for i in range(len(self.a)):
                ax.plot3D([0, self.r[i][0]], [0, self.r[i][1]], [0, self.r[i][2]], 'blue')

                plot_vector(ax, self.r[i][0], self.r[i][1], self.r[i][2], self.res.x[i * 3],
                            self.res.x[i * 3 + 1], self.res.x[i * 3 + 2], 'red')
                plot_vector(ax, self.r[i][0], self.r[i][1], self.r[i][2], 0, 0, self.res.x[i * 3 + 2] * self.kQ[i], 'green')

                if self.phi is not 0:
                    theta = np.linspace(0, 2 * np.pi, self.N)
                    x = self.min_f * math.tan(self.phi) * np.cos(theta) + self.r[i][0]
                    y = self.min_f * math.tan(self.phi) * np.sin(theta) + self.r[i][1]
                    X, Y = np.meshgrid(x, y)
                    Z = np.full((len(X), len(X[0])), self.min_f + self.r[i][2])
                    circle_mask = (X - self.r[i][0]) ** 2 + (Y - self.r[i][1]) ** 2 <= (
                                self.min_f * math.tan(self.phi)) ** 2
                    X = X[circle_mask]
                    Y = Y[circle_mask]
                    Z = Z[circle_mask]

                    X_p = np.array([])
                    Y_p = np.array([])
                    Z_p = np.array([])
                    for j in range(len(self.pyr_lines) - 1):
                        X_p = np.append(X_p, self.r[i][0] + self.pyr_lines[j][0])
                        X_p = np.append(X_p, self.r[i][0] + self.pyr_lines[j + 1][0])
                        Y_p = np.append(Y_p, self.r[i][1] + self.pyr_lines[j][1])
                        Y_p = np.append(Y_p, self.r[i][1] + self.pyr_lines[j + 1][1])
                        Z_p = np.append(Z_p, self.r[i][2] + self.pyr_lines[j][2])
                        Z_p = np.append(Z_p, self.r[i][2] + self.pyr_lines[j + 1][2])
                    ax.plot_trisurf(np.concatenate((X, X_p)),
                                    np.concatenate((Y, Y_p)), np.concatenate((Z, Z_p)), color='skyblue', alpha=0.5)
                    ax.plot_trisurf(X_p, Y_p, Z_p, color='skyblue', alpha=0.5)
            plt.show()
