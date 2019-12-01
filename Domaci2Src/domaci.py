import numpy as np
import math
# slike na github nalogu https://github.com/jboskovic/PPGR


def Euler2A(fi, teta, psi):
    # fi = (fi*math.pi)/180
    # teta = (teta*math.pi)/180
    # psi = (psi*math.pi)/180

    Rz = np.array([[math.cos(psi), -math.sin(psi), 0],
                   [math.sin(psi), math.cos(psi), 0],
                   [0, 0, 1]])

    Ry = np.array([[math.cos(teta), 0, math.sin(teta)],
                   [0, 1, 0],
                   [-math.sin(teta), 0, math.cos(teta)]])

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(fi), -math.sin(fi)],
                   [0, math.sin(fi), math.cos(fi)]])

    return Rz.dot(Ry).dot(Rx)


A = Euler2A(-math.pi/6, -math.pi/6, math.pi/6)
# print(np.linalg.det(A))
# print(A)
# print(np.linalg.det(A))


def Rodrigez(p, fi):
    # fi = (fi*math.pi)/180

    p1 = p[0]
    p2 = p[1]
    p3 = p[2]

    Px = np.array([[0, -p3, p2],
                   [p3, 0, -p1],
                   [-p2, p1, 0]])

    E = np.eye(3)
    p = np.reshape(p, (3, 1))
    Rp = p.dot(p.T) + math.cos(fi)*(E - p.dot(p.T)) + math.sin(fi)*Px
    # print(Rp)
    return Rp


Rp = Rodrigez([math.sqrt(2)/3, math.sqrt(2)/3, math.sqrt(5)/3], math.pi/3)
# print(Rp)


def AxisAngle(A):
    lambdas, vector = np.linalg.eig(A, )
    for i in range(len(lambdas)):
        if round(lambdas[i], 6) == 1.0:
            p = np.real(vector[:, i])

    p1 = p[0]
    p2 = p[1]
    p3 = p[2]

    u = np.cross(p, np.array([p1, p2, -p3]))
    u = u/math.sqrt(u[0]**2+u[1]**2+u[2]**2)

    up = A.dot(u)

    fi = round(math.acos(np.sum(u*up)), 5)
    if round(np.sum(np.cross(u, up)*p), 5) < 0:
        p = (-1)*p

    return [p, fi]


# A je matrica za koju vazi det(A)
A = Euler2A(-math.pi/6, -math.pi/6, math.pi/6)
p, fi = AxisAngle(A)
# print(p)
# print(fi)


def A2Euler(A):
    fi, teta, psi = 0, 0, 0
    if A[2, 0] < 1:
        if A[2, 0] > -1:
            psi = math.atan2(A[1, 0], A[0, 0])
            teta = math.asin((-1)*A[2, 0])
            fi = math.atan2(A[2, 1], A[2, 2])
        else:
            psi = math.atan2((-1)*A[0, 1], A[1, 1])
            teta = math.pi/2.0
            fi = 0.0
    else:
        psi = math.atan2((-1)*A[0, 1], A[1, 1])
        teta = (-1.0)*math.pi/2.0
        fi = 0

    return([fi, teta, psi])


# det(A) = 1
A = Euler2A(-math.pi/6, -math.pi/6, math.pi/6)
B = A2Euler(A)
# print(B)


def AngleAxis2Q(p, fi):
    w = math.cos(fi/2.0)
    norm = np.linalg.norm(p)
    if norm != 0:
        p = p/norm
    [x, y, z] = math.sin(fi/2.0) * p
    return [x, y, z, w]


q = AngleAxis2Q([0, 1, 0], math.pi/3)
# print(q)


def Q2AxisAngle(q):
    norm = np.linalg.norm(q)
    if norm != 0:
        q = q/norm

    fi = 2*math.acos(q[3])
    if abs(q[3]) == 1:
        p = [1, 0, 0]
    else:
        norm = np.linalg.norm(np.array([q[0], q[1], q[2]]))
        p = np.array([q[0], q[1], q[2]])
        if norm != 0:
            p = p / norm

    return [p, fi]


q = AngleAxis2Q([1/2, 1/2, 1/math.sqrt(2)], math.pi/3)
p, fi = Q2AxisAngle(q)
print(p)
print(fi)


#q = AngleAxis2Q([1.0/3.0, -2.0/3.0, 2.0/3.0], math.pi/2.0)
# print(Q2AxisAngle(q))


#print(AngleAxis2Q([1.0/3.0, -2.0/3.0, 2.0/3.0], math.pi/2.0))
#print(AxisAngle(Rodrigez(np.array([1.0/3.0, -2.0/3.0, 2/3]), math.pi/2.0)))
#print(A2Euler(Rodrigez(np.array([1.0/3.0, -2.0/3.0, 2/3]), math.pi/2.0)))
#A = Euler2A(-math.atan(1/4), -math.asin(8/9), math.atan(4))
# print(A)
