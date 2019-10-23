import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import tkinter as tk
import copy
import math


def solve_system(points):
    A,B,C,D = points
    result = np.array(D)
    matrix = np.array([[A[0], B[0], C[0]], [A[1], B[1], C[1]], [A[2], B[2], C[2]]])
    xs = np.linalg.solve(matrix, result)
    print(xs)
    return xs


def naive_algorithm(originals, pictures):
    A,B,C,D = originals
    Ap,Bp,Cp,Dp = pictures
    lambde = solve_system(originals)
    lambdep = solve_system(pictures)
    P1 = np.array([[lambde[0]*A[0], lambde[1]*B[0], lambde[2]*C[0]], [lambde[0]
                                                                      * A[1], lambde[1]*B[1], lambde[2]*C[1]], [lambde[0]*A[2], lambde[1]*B[2], lambde[2]*C[2]]])

    P2 = np.array([[lambdep[0]*Ap[0], lambdep[1]*Bp[0], lambdep[2]*Cp[0]], [lambdep[0]
                                                                            * Ap[1], lambdep[1]*Bp[1], lambdep[2]*Cp[1]], [lambdep[0]*Ap[2], lambdep[1]*B[2], lambdep[2]*Cp[2]]])

    P = np.dot(P2, np.linalg.inv(P1))
    return P


def normalization(points):
    x = 0.0
    y = 0.0
    num_points = len(points)
    for i in range(num_points):
        x = x + float(points[i][0]/points[i][2])
        y = y + float(points[i][1]/points[i][2])


    x = float(x) / float(num_points)
    y = float(y) / float(num_points)

    distance_sum = 0.0

    for i in range(num_points):
        tmp1 = points[i][0]/points[i][2] - x
        tmp2 = points[i][1]/points[i][2] - y
        distance_sum = distance_sum + \
            math.sqrt(tmp1**2+tmp2**2)

    distance_sum = float(distance_sum) / float(num_points)
    k = float(math.sqrt(2)) / distance_sum
    return np.array([[k, 0, -k*x], [0, k, -k*y], [0, 0, 1]])

def dlt(originals, pictures):
    x1 = originals[0][0]
    x2 = originals[0][1]
    x3 = originals[0][2]


    x1p = pictures[0][0]
    x2p = pictures[0][1]
    x3p = pictures[0][2]

    A = np.array([[0,0,0,(-1)*x3p*x1,(-1)*x3p*x2,(-1)*x3p*x3,x2p*x1,x2p*x2,x2p*x3],[x3p*x1,x3p*x2,x3p*x3,0,0,0,(-1)*x1p*x1,(-1)*x1p*x2,(-1)*x1p*x3]])

    for i in range(1, len(originals)):
        x1 = originals[i][0]
        x2 = originals[i][1]
        x3 = originals[i][2]


        x1p = pictures[i][0]
        x2p = pictures[i][1]
        x3p = pictures[i][2]
        
        r1 = np.array([0,0,0,(-1)*x3p*x1,(-1)*x3p*x2,(-1)*x3p*x3,x2p*x1,x2p*x2,x2p*x3])
        r2 = np.array([x3p*x1,x3p*x2,x3p*x3,0,0,0,(-1)*x1p*x1,(-1)*x1p*x2,(-1)*x1p*x3])
        A = np.vstack((A,r1))
        A = np.vstack((A,r2))

  
    U, D, V = np.linalg.svd(A, full_matrices = False)

    P = V[-1].reshape(3,3)
  
    return P

def dlt_normalize(originals, pictures):
    T = normalization(originals)
    Tp = normalization(pictures)

 
    originals_copy = copy.deepcopy(originals)
    pictures_copy = copy.deepcopy(pictures)
    for i in range(len(originals)):
        [x, y, z] = np.dot(T, [originals[i][0], originals[i][1], originals[i][2]]) 
        originals_copy[i][0] = x
        originals_copy[i][1] = y
        originals_copy[i][2] = z

    print(originals_copy)
    for i in range(len(pictures)):
        [x, y, z] = np.dot(Tp, [pictures[i][0], pictures[i][1], pictures[i][2]]) 
        pictures_copy[i][0] = x
        pictures_copy[i][1] = y
        pictures_copy[i][2] = z

    print(pictures_copy)

    Pp = dlt(originals_copy,pictures_copy)

    print("Pp")
    print(Pp)
   
    tmp = np.dot(np.linalg.inv(Tp), Pp)
    P = np.dot(tmp, T)

    for i in range(len(originals)):
        a = np.dot(P, [originals[i][0], originals[i][1], 1])
        print(a[0]*1.0/a[2])
        print(a[1]*1.0/a[2])

    print("P")
    print(P)

    
  
    return P



#dlt([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1],[1,2,3],[-8,-2,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1],[2,1,4],[-16,-5,4]])
#dlt_normalize([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1],[1,2,3],[-8,-2,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1],[2,1,4],[-16,-5,4]])
#print(naive_algorithm([[-3,-1,1],[3,-1,1], [1,1,1], [-1,1,1]], [[-2,-1,1], [2,-1,1], [2,1,1], [-2,1,1]]))
#print(naive_algorithm([[1,1,1],[5,2,1], [6,4,1], [-1,7,1]], [[0,0,1], [10,0,1], [10,5,1], [0,5,1]]))
#print(dlt([[1,1,1],[5,2,1], [6,4,1], [-1,7,1]], [[0,0,1], [10,0,1], [10,5,1], [0,5,1]]))0
#print(dlt([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1],[1,2,3],[-8,-2,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1],[2,1,4],[-16,-5,4]]))