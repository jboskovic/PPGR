import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(vector):
    #return np.array([x / vector[2] for x in vector])
    return [x / vector[2] for x in vector]

def test(F, x, y):
    for i in range(8):
        if round(y[i].T.dot(FF).dot(x[i]), 5) != 0.0:
            return False
    return True


def jed(x, y):
    a1 = x[0]
    b1 = y[0]

    a2 = x[1]
    b2 = y[1]

    a3 = x[2]
    b3 = y[2]

    return np.array([a1*b1, a2*b1, a3*b1, a1*b2, a2*b2, a3*b2, a1*b3, a2*b3, a3*b3])


def vec(p):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    return np.array([[0,-p3,p2],[p3,0,-p1], [-p2,p1,0]])

def jednacine(xx, yy, T1, T2):
    return np.array([xx[1]*T1[2]-xx[2]*T1[1],-xx[0]*T1[2]+xx[2]*T1[0],yy[1]*T2[2]-yy[2]*T2[1], -yy[0]*T2[2]+yy[2]*T2[0]])

def UAfine(XX):
    XX = XX/XX[3]
    XX = np.array([XX[0],XX[1],XX[2]])
    #print("UAfine XX", XX)
    return XX

#Koordinate tacaka sa slike 2 kutije
#x1 = np.array([958, 38, 1])
#y1 = np.array([933, 33, 1])

#x2 = np.array([1117, 111, 1])
#y2 = np.array([1027, 132, 1])

#x3 = np.array([874, 285, 1])
#y3 = np.array([692, 223, 1])

#x4 = np.array([707, 218, 1])
#y4 = np.array([595, 123, 1])

#x9 = np.array([292, 569, 1])
#y9 = np.array([272, 360, 1])

#x10 = np.array([770, 969, 1])
#y10 = np.array([432, 814, 1])

#x11 = np.array([770, 1465, 1])
#y11 = np.array([414, 1284, 1])

#x12 = np.array([317, 1057, 1])
#y12 = np.array([258, 818, 1])


#xx = np.array([x1, x2, x3, x4, x9, x10, x11, x12])
#yy = np.array([y1, y2, y3, y4, y9, y10, y11, y12])


#x6 = np.array([1094, 536, 1])
#y6 = np.array([980, 535, 1])

#x7 = np.array([862, 729, 1])
#y7 = np.array([652, 638, 1])

#x8 = np.array([710, 648, 1])
#y8 = np.array([567, 532, 1])

#x14 = np.array([1487, 598, 1])
#y14 = np.array([1303, 700, 1])

#x15 = np.array([1462, 1079, 1])
#y15 = np.array([1257, 1165, 1])

#y13 = np.array([1077, 269, 1])

#skrivene tacke slika 2 kutije

#x5 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x4, x8)), f(np.cross(x6, x2)))), x1)),
             # f(np.cross(f(np.cross(f(np.cross(x1, x4)), f(np.cross(x3, x2)))), x8))))
#x5 = np.round(x5)
#print("X5:")
#print(x5)


#x13 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x9, x10)), f(np.cross(x11, x12)))), x14)),
    #          f(np.cross(f(np.cross(f(np.cross(x11, x15)), f(np.cross(x10, x14)))), x9))))

#x13 = np.round(x13)

#x16 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x10, x14)), f(np.cross(x11, x15)))), x12)),
     #         f(np.cross(f(np.cross(f(np.cross(x9, x10)), f(np.cross(x11, x12)))), x15))))

#x16 = np.round(x16)
#print("X13:")
#print(x13)

#print("X16:")
#print(x16)


#y5 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y4, y8)), f(np.cross(y6, y2)))), y1)),
      #        f(np.cross(f(np.cross(f(np.cross(y1, y4)), f(np.cross(y3, y2)))), y8))))

#y5 = np.round(y5)
#print("Y5", y5)

#y16 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y10, y14)), f(np.cross(y11, y15)))), y12)),
       #       f(np.cross(f(np.cross(f(np.cross(y9, y10)), f(np.cross(y11, y12)))), y15))))

#y16 = np.round(y16)
#print("Y16",y16)



x1 = np.array([335,75,1])
x2 = np.array([555,55,1])
x3 = np.array([720,170,1])
x4 = np.array([540,190,1])
x5 = np.array([333,295,1])
x6 = np.array([0,0,1]) #nevidljiva tacka
x7 = np.array([715,400,1])
x8 = np.array([540,430,1])
x9 = np.array([260,340,1])
x10 = np.array([0,0,1]) #nevidljiva tacka
x11 = np.array([780,365,1])
x12 = np.array([315,410,1])
x13 = np.array([270,590,1])
x14 = np.array([0,0,1]) #nevidljiva tacka
x15 = np.array([770,615,1])
x16 = np.array([315,670,1])
x17 = np.array([95,630,1])
x18 = np.array([0,0,1]) #nevidljiva tacka
x19 = np.array([930,600,1])
x20 = np.array([700,780,1])
x21 = np.array([96,826,1])
x22 = np.array([555,55,1]) #nevidljiva tacka
x23 = np.array([920,788,1])
x24 = np.array([698,990,1])


y1 = np.array([396,77,1])
y2 = np.array([564,80,1])
y3 = np.array([568,198,1])
y4 = np.array([373,198,1])
y5 = np.array([0,0,1]) #nevidljiva tacka
y6 = np.array([0,0,1]) #nevidljiva tacka
y7 = np.array([568,423,1])
y8 = np.array([379,422,1])
y9 = np.array([284,315,1])
y10 = np.array([237, 376, 1])
y11 = np.array([690, 403, 1])
y12 = np.array([717, 335, 1])
y13 = np.array([0, 0, 1]) #nevidljiva tacka
y14 = np.array([719, 555, 1]) 
y15 = np.array([688, 639, 1])
y16 = np.array([251, 617, 1])
y17 = np.array([127, 550, 1])
y18 = np.array([0, 0, 1]) #nevidljiva tacka
y19 = np.array([863, 654, 1])
y20 = np.array([463, 783, 1])
y21 = np.array([133, 721, 1])
y22 = np.array([0, 0, 1]) #nevidljiva tacka
y23 = np.array([858, 838, 1])
y24 = np.array([466, 975, 1])


#skrivene tacke 

x6 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x1, x5)), f(np.cross(x7, x3)))), x2)),
              f(np.cross(f(np.cross(f(np.cross(x1, x4)), f(np.cross(x3, x2)))), x7))))
x6 = np.round(x6)
    
x10 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x9, x12)), f(np.cross(x16, x13)))), x11)),
              f(np.cross(f(np.cross(f(np.cross(x12, x11)), f(np.cross(x15, x16)))), x9))))
x10 = np.round(x10)
    
x14 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x12, x11)), f(np.cross(x16, x15)))), x13)),
              f(np.cross(f(np.cross(f(np.cross(x9, x12)), f(np.cross(x16, x13)))), x15))))
x14 = np.round(x14)
    
x18 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x17, x20)), f(np.cross(x21, x24)))), x19)),
              f(np.cross(f(np.cross(f(np.cross(x20, x19)), f(np.cross(x23, x24)))), x17))))
x18 = np.round(x18)
   
x22 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x20, x19)), f(np.cross(x23, x24)))), x21)),
              f(np.cross(f(np.cross(f(np.cross(x17, x20)), f(np.cross(x24, x21)))), x23))))
x22 = np.round(x22)
    
    
y5 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y4, y8)), f(np.cross(y7, y3)))), y1)),
              f(np.cross(f(np.cross(f(np.cross(y1, y4)), f(np.cross(y3, y2)))), y8))))
y5 = np.round(y5)
    
y6 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y1, y5)), f(np.cross(y7, y3)))), y2)),
              f(np.cross(f(np.cross(f(np.cross(y1, y4)), f(np.cross(y3, y2)))), y7))))
y6 = np.round(y6)
    
y13 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y9, y10)), f(np.cross(y11, y12)))), y16)),
              f(np.cross(f(np.cross(f(np.cross(y9, y12)), f(np.cross(y11, y10)))), y16))))
y13 = np.round(y13)
    
y18 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y17, y20)), f(np.cross(y21, y24)))), y19)),
              f(np.cross(f(np.cross(f(np.cross(y20, y19)), f(np.cross(y23, y24)))), y17))))
y18 = np.round(y18)
    
y22 = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y20, y19)), f(np.cross(y23, y24)))), y21)),
              f(np.cross(f(np.cross(f(np.cross(y17, y20)), f(np.cross(y24, y21)))), y23))))
y22 = np.round(y22)




jed8 = []


num_of_visibaly_points = 20

xx = np.array([x1,x2,x3,x4,x7,x8,x9,x11])
yy = np.array([y1,y2,y3,y4,y7,y8,y9,y11])

print("MatrixForm[jed8]: ")
for i in range(8):  
    tmp = jed(xx[i], yy[i])
    jed8.append(tmp)
    print(tmp)

print()


# for i in range(8):
 #   for j in range(9):
  #      print(jed8[i][j], end=' ')
   # print()

U, Dp, V = np.linalg.svd(jed8)
D = np.zeros((8, 9), float)
np.fill_diagonal(D, Dp)

print("MatrixForm[SVDJed8]")
print("U")
print(U)
print("D")
print(D)
print("V")
print(V)
print()


Fvector = V[-1:]
Fvector = Fvector[0]

FF = np.array([[Fvector[0], Fvector[1], Fvector[2]], [
              Fvector[3], Fvector[4], Fvector[5]], [Fvector[6], Fvector[7], Fvector[8]]])
print("Fundamentalna matrica FF: ")
print(FF)
print()

print("Det(FF):")
print(np.linalg.det(FF))
print()

print("Testing:")
print(test(FF, xx, yy))
print()


FFX = FF
SVDFF = np.linalg.svd(FFX)

U = SVDFF[0]
D = SVDFF[1]
V = SVDFF[2]

print("MatrixForm[SVDFF]")
print("U")
print(U)
print("D")
print(D)
print("V")
print(V)
print()


D[2] = 0 
DD = [[D[0], 0,0],[0, D[1], 0], [0,0,0]]
FF1 = U @ DD @ V
print("Fundamentalna matrica FF1: ")
print(FF1)
print()

print("Det(FF1)")
print(np.linalg.det(FF1))
print()

e1 = SVDFF[2][-1, :]
print("e1:")
print(e1)

e1 = (1/e1[2])*e1
print("e1 afino:")
print(e1)
print()
# print(e1)


e2 = SVDFF[0][:, -1]
print("e2")
print(e2)
e2 = (1/e2[2])*e2
print("e2 afino")
print(e2)
print()
# print(e2)

#DD1 = np.array([[1,0,0], [0,9.88647*pow(10,-6),0],[0,0,1]])
#print(SVDFF[0])
#FF1 = SVDFF[0].dot(DD1)
#FF1 = FF1.dot(SVDFF[2])
#print(FF1)

#print("FF", FF)
#print("FF1: ",FF1)

T1 = np.zeros((3, 4), float)
np.fill_diagonal(T1, 1)
print("T1:")
print(T1)
print()
#print("T1:")


E2 = vec(e2)
#print(E2.dot(FF1))
print("E2:")
print(E2)
print()

T2 = E2.dot(FF1).T
T2 = np.vstack([T2, e2])

T2 = T2.T
print("T2: ")
print(T2)
print()

C1 = np.array([0,0,0,1])
U,D,V = np.linalg.svd(T2)
C2 = V[:][-1]

print("Koordinate prve kamere C1:")
print(C1)
print("Koordinate druge kamere C2:")
print(C2)
print()


tmp = jednacine(x1,y1,T1, T2)
U, D, V = np.linalg.svd(tmp)
afino = UAfine(V[3])

def TriD(xx, yy, T1, T2):
    U, D, V = np.linalg.svd(jednacine(xx,yy,T1, T2))
    Vp = V[3]
    Vafino = UAfine(Vp)
    return Vafino


slika1 = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16])
slika2 = np.array([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16])


slika1 = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24])
slika2 = np.array([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24])

rekonstruisane = []
for i in range(len(slika1)):
    tmp = TriD(slika1[i], slika2[i], T1, T2)
    rekonstruisane.append(tmp)
    #print(tmp)


dig = np.eye(3)
dig[2][2] = 400
rekonstruisane400 = np.zeros((len(slika1),3))
print("Rekonstruisano400:")
for i in range(len(slika1)):
    rekonstruisane400[i]=dig.dot(rekonstruisane[i])
    print(rekonstruisane400[i])

X = rekonstruisane400

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:4,0], X[:4,1], X[:4,2], color = "blue")
ax.scatter(X[4:8,0], X[4:8,1], X[4:8,2], color = "blue")
ax.scatter(X[8:12,0], X[8:12,1], X[8:12,2], color = "blue")
ax.scatter(X[12:16,0], X[12:16,1], X[12:16,2], color = "blue")
ax.scatter(X[16:20,0], X[16:20,1], X[16:20,2], color = "blue")
ax.scatter(X[20:,0], X[20:,1], X[20:,2], color = "blue")

plt.gca().invert_yaxis()
plt.show()
