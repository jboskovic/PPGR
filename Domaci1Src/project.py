import numpy as np
import numpy.linalg
from matplotlib import pyplot
import copy
import math
from PIL import Image
import tkinter


def solve_system(points):
    # lambda1*A+lambda2*B+lambda3*C = D
    A, B, C, D = points
    result = np.array(D)
    matrix = np.array(
        [[A[0], B[0], C[0]], [A[1], B[1], C[1]], [A[2], B[2], C[2]]])
    lambdas = np.linalg.solve(matrix, result)

    return lambdas


def naive_algorithm(originals, pictures):
    A, B, C, D = originals
    Ap, Bp, Cp, Dp = pictures
    lambde = solve_system(originals)
    lambdep = solve_system(pictures)
    P1 = np.array([[lambde[0]*A[0], lambde[1]*B[0], lambde[2]*C[0]], [lambde[0]
                                                                      * A[1], lambde[1]*B[1], lambde[2]*C[1]], [lambde[0]*A[2], lambde[1]*B[2], lambde[2]*C[2]]])

    P2 = np.array([[lambdep[0]*Ap[0], lambdep[1]*Bp[0], lambdep[2]*Cp[0]], [lambdep[0]
                                                                            * Ap[1], lambdep[1]*Bp[1], lambdep[2]*Cp[1]], [lambdep[0]*Ap[2], lambdep[1]*B[2], lambdep[2]*Cp[2]]])

    P = np.dot(P2, np.linalg.inv(P1))
    return P


def normalization(points):
    # (x,y) - teziste sistema tacaka
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
        tmp1 = float(points[i][0])/float(points[i][2]) - x
        tmp2 = float(points[i][1])/float(points[i][2]) - y
        distance_sum = distance_sum + \
            math.sqrt(tmp1**2+tmp2**2)

    distance_sum = float(distance_sum) / float(num_points)
    k = float(math.sqrt(2)) / distance_sum
    return np.array([[k, 0, -k*x], [0, k, -k*y], [0, 0, 1]])


def dlt(originals, pictures):
    x1 = float(originals[0][0])
    x2 = float(originals[0][1])
    x3 = float(originals[0][2])

    x1p = float(pictures[0][0])
    x2p = float(pictures[0][1])
    x3p = float(pictures[0][2])

    A = np.array([
        [0, 0, 0, (-1)*x3p*x1, (-1)*x3p*x2, (-1)*x3p*x3, x2p*x1, x2p*x2, x2p*x3],
        [x3p*x1, x3p*x2, x3p*x3, 0, 0, 0, (-1)*x1p*x1, (-1)*x1p*x2, (-1)*x1p*x3]
    ])

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

    # python numpy biblioteka ima funkciju koja radi svd dekompoziciju
    U, D, V = np.linalg.svd(A)


    P = V[-1].reshape(3,3)
  

    return P

def dlt_normalize(originals, pictures):
    T = normalization(originals)
    Tp = normalization(pictures)
 
    # pravimo deepcopy da ne bismo menjali koordinate tacaka (lista se prenosi po referenci)
    originals_copy = copy.deepcopy(originals)
    pictures_copy = copy.deepcopy(pictures)

    for i in range(len(originals)):
        [x, y, z] = np.dot(T, [originals[i][0], originals[i][1], originals[i][2]]) 
        originals_copy[i][0] = float(x)
        originals_copy[i][1] = float(y)
        originals_copy[i][2] = float(z)

    for i in range(len(pictures)):
        [x, y, z] = np.dot(Tp, [pictures[i][0], pictures[i][1], pictures[i][2]]) 
        pictures_copy[i][0] = float(x)
        pictures_copy[i][1] = float(y)
        pictures_copy[i][2] = float(z)



    Pp = dlt(originals_copy,pictures_copy)

    P = np.dot(np.linalg.inv(Tp), Pp)
    P = np.dot(P, T)

    return P


def enter_coordinates():
    coordinates = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    print("Enter 4 random points ")
    for i in range(4):
        coordinates[i][0] = float(input("Enter x coordinate "))
        coordinates[i][1] = float(input("Enter y coordinate "))
        coordinates[i][2] = 1

    return coordinates


def picture_edit(name, algorithm):
    picture_old = Image.open(name)
    picture_old.show()
    dimensions = picture_old.size
    picture_new = Image.new('RGB', dimensions, 'black')

    pixel_old = picture_old.load()
    pixel_new = picture_new.load()


    original = enter_coordinates()

    A = original[0]
    B = original[1]
    C = original[2]
    D = original[3]

    AD_len = math.sqrt((A[0]-D[0])**2 +(A[1]-D[1])**2)
    BC_len = math.sqrt((B[0]-C[0])**2 +(B[1]-C[1])**2)

    AB_len = math.sqrt((A[0]-B[0])**2 +(A[1]-B[1])**2)
    DC_len = math.sqrt((D[0]-C[0])**2 +(D[1]-C[1])**2)

    vertic_mean = (AD_len + BC_len)/2
    horizontal_mean = (AB_len+DC_len)/2


    # koordinate tacaka slike odredjuemo tako sto izracunamo prosecnu duzinu vertiklanih (horizontalnih) duzi
    # gornja leva tacka je ista kao gornja leva tacka originalne slike
    # ostale tacke se racunaju tako da obrazuju pravougaonik i da su udaljene od gornje leve tacke za prosecnu duzinu vertikale(horizontale)
 
    picture = [[A[0],A[1],1],[A[0]+horizontal_mean, A[1], 1], [A[0]+horizontal_mean,A[1]+vertic_mean, 1], [A[0], A[1]+vertic_mean,1]]

    # u ovom delu koda biramo koji algoritam zelimo da primenimo
    if algorithm == 0:
        P = naive_algorithm(original, picture)
    elif algorithm == 1:
        P = dlt(original, picture)
    elif algorithm == 2:
        P = dlt_normalize(original, picture)
    else:
        P = naive_algorithm(original, picture)

   
    P = np.linalg.inv(P)

    # prolazimo kroz koordinate nove slike
    # za svaku koordinatu nove slike racunamo koje su to kooridanate u originalnoj slici
    # vrednost pikslea na novoj slici je vrednost piksel koji se nalazi na koordinatama stare slike
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            new_coordinates = np.dot(P, [i,j,1])
    

            x = round(new_coordinates[0] / new_coordinates[2])
            y = round(new_coordinates[1] / new_coordinates[2])

            if x<0 or x>=dimensions[0]:
                continue
            elif y<0 or y>=dimensions[1]:
                continue
            else:
                pixel_new[i,j] = pixel_old[x,y]
            
    picture_new.save('building_edited.bmp')
    picture_new.show()
    return

def comparasion_between_algorithms():
    original = np.array([[1.0, 1.0, 1.0], [5.0, 2.0,1.0], [6.0, 4.0,1.0],[-1.0, 7.0,1.0],[3.0,1.0,1.0]])
    picture =  np.array([[0.0,0.0,1.0],[10.0,0.0,1.0],[10.0,5.0,1.0],[0.0,5.0,1.0],[3.0,-1.0,1.0]])


    C1 = np.array([[0, 1, 2], [-1, 0, 3], [0, 0, 1]])
    C2 = np.array([[1, -1, 5], [1, 1, -2], [0, 0, 1]])


    original_new = []
    picture_new = []

    for i in range(len(original)):
        original_new.append(np.dot(C1, original[i]))
        picture_new.append(np.dot(C2, picture[i]))
 
    original_new = np.array(original_new)
    picture_new = np.array(picture_new)

    P = dlt(original,picture)
 
    Pp = dlt(original_new, picture_new)
  

    P_tmp = np.dot(np.linalg.inv(C2), Pp)
    P_tmp= np.dot(P_tmp, C1)


 

    P_tmp = (P[0]/P_tmp[0])*P_tmp
    
    print("DLT algorithm: ")
    print(P)
    print(P_tmp)


    P = dlt_normalize(original,picture)
    Pp = dlt_normalize(original_new, picture_new)
    


    P_tmp= np.dot(np.linalg.inv(C2), Pp)
    P_tmp = np.dot(P_tmp, C1)
    
    print("DLT modified algorithm: ")
    print(P)
    print(P_tmp)


#

# dlt([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1],[1,2,3],[-8,-2,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1],[2,1,4],[-16,-5,4]])
# print(dlt_normalize([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1],[1,2,3],[-8,-2,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1],[2,1,4],[-16,-5,4]]))
# print(naive_algorithm([[-3,-1,1],[3,-1,1], [1,1,1], [-1,1,1]], [[-2,-1,1], [2,-1,1], [2,1,1], [-2,1,1]]))
# print(naive_algorithm([[1,1,1],[5,2,1], [6,4,1], [-1,7,1]], [[0,0,1], [10,0,1], [10,5,1], [0,5,1]]))
# print(dlt([[1,1,1],[5,2,1], [6,4,1], [-1,7,1]], [[0,0,1], [10,0,1], [10,5,1], [0,5,1]]))

# P = dlt([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1],[1,2,3],[-8,-2,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1],[2,1,4],[-16,-5,4]])
# print(np.round(P, 5))
# P_naive =  naive_algorithm([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1]])

# P = (P/P[0,0])*P_naive[0,0]

# print(np.round(P, 5))

# P = dlt_normalize([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1],[1,2,3],[-8,-2,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1],[2,1,4],[-16,-5,4]])
# print(np.round(P, 5))


# P_naive =  naive_algorithm([[-3, -1,1], [3, -1,1], [1, 1,1],[-1, 1,1]], [[-2, -1,1], [2, -1,1], [2, 1,1], [-2, 1,1]])

# P = (P/P[0,0])*P_naive[0,0]

# print(np.round(P, 5))


# originals = np.array([[1,1,1],[5,2,1],[6,4,1],[-1,7,1]])
# pictures = np.array([[0,0,1],[10,0,1],[10,5,1],[0,5,1]])

# P = naive_algorithm(originals,pictures)
# print(P)

# P_dlt = dlt(originals, pictures)
# print(P_dlt)

picture_name = input("Enter a name of a picture you want to edit: ")
algorithm = int(input('Choose algorithm you want to use.\n0 - naive algorithm\n1 - dlt algorithm\n2 - modified dlt algorithm\n'))
picture_edit(picture_name, algorithm)























