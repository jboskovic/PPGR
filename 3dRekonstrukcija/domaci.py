import numpy as np


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

    
x1 = np.array([958, 38, 1])
y1 = np.array([933, 33, 1])

x2 = np.array([1117, 111, 1])
y2 = np.array([1027, 132, 1])

x3 = np.array([874, 285, 1])
y3 = np.array([692, 223, 1])

x4 = np.array([707, 218, 1])
y4 = np.array([595, 123, 1])

x9 = np.array([292, 569, 1])
y9 = np.array([272, 360, 1])

x10 = np.array([770, 969, 1])
y10 = np.array([432, 814, 1])

x11 = np.array([770, 1465, 1])
y11 = np.array([414, 1284, 1])

x12 = np.array([317, 1057, 1])
y12 = np.array([258, 818, 1])


xx = np.array([x1, x2, x3, x4, x9, x10, x11, x12])
yy = np.array([y1, y2, y3, y4, y9, y10, y11, y12])


x6 = np.array([1094, 536, 1])
y6 = np.array([980, 535, 1])

x7 = np.array([862, 729, 1])
y7 = np.array([652, 638, 1])

x8 = np.array([710, 648, 1])
y8 = np.array([567, 532, 1])

x14 = np.array([1487, 598, 1])
y14 = np.array([1303, 700, 1])

x15 = np.array([1462, 1079, 1])
y15 = np.array([1257, 1165, 1])

y13 = np.array([1077, 269, 1])


x5 = np.cross(np.cross(np.cross(np.cross(x4, x8), np.cross(x6, x2)), x1),
              np.cross(np.cross(np.cross(x1, x4), np.cross(x3, x2)), x8))
x5 = np.round(x5/x5[2])
print("X5:")
print(x5)


x13 = np.cross(np.cross(np.cross(np.cross(x9, x10), np.cross(x11, x12)), x14),
               np.cross(np.cross(np.cross(x11, x15), np.cross(x10, x14)), x9))
x13 = np.round(x13/x13[2])

x16 = np.cross(np.cross(np.cross(np.cross(x10, x14), np.cross(x11, x15)), x12),
               np.cross(np.cross(np.cross(x9, x10), np.cross(x11, x12)), x15))
x16 = np.round(x16/x16[2])
print("X13:")
print(x13)

print("X16:")
print(x16)




y5 = np.cross(np.cross(np.cross(np.cross(y4, y8), np.cross(y6, y2)), y1),
              np.cross(np.cross(np.cross(y1, y4), np.cross(y3, y2)), y8))
y5 = np.round(y5/y5[2])
print("Y5", y5)

y16 = np.cross(np.cross(np.cross(np.cross(y10, y14), np.cross(y11, y15)), y12),
               np.cross(np.cross(np.cross(y9, y10), np.cross(y11, y12)), y15))
y16 = np.round(y16/y16[2])
print("Y16",y16)

jed8 = []

for i in range(8):
    tmp = jed(xx[i], yy[i])
    jed8.append(tmp)


# for i in range(8):
 #   for j in range(9):
  #      print(jed8[i][j], end=' ')
   # print()

U, Dp, V = np.linalg.svd(jed8)
D = np.zeros((8, 9), float)
np.fill_diagonal(D, Dp)

# print("U")
# print(U)
# print("D")
# print(D)
# print("V")
# print(V)

Fvector = V[-1:]
Fvector = Fvector[0]

FF = np.array([[Fvector[0], Fvector[1], Fvector[2]], [
              Fvector[3], Fvector[4], Fvector[5]], [Fvector[6], Fvector[7], Fvector[8]]])

print("Testing:")
print(test(FF, xx, yy))

print("Det(F):")
print(np.linalg.det(FF))

FFX = FF
SVDFF = np.linalg.svd(FFX)

U = SVDFF[0]
D = SVDFF[1]
V = SVDFF[2]

D[2] = 0 
DD = [[D[0], 0,0],[0, D[1], 0], [0,0,0]]
FF1 = U @ DD @ V

e1 = SVDFF[2][-1, :]
# print(e1)

e1 = (1/e1[2])*e1
# print(e1)


e2 = SVDFF[0][:, -1]
e2 = (1/e2[2])*e2
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
#print(T1)


E2 = vec(e2)
#print(E2.dot(FF1))

T2 = E2.dot(FF1).T
print(T2)
T2 = np.vstack([T2, e2])
print(T2)
T2 = T2.T
print(T2)

tmp = jednacine(x1,y1,T1, T2)
print("tmp")
print(tmp)
U, D, V = np.linalg.svd(tmp)
print(U)
print(D)
print(V)
print("Jednacina V[3]")
print(V[3])
afino = UAfine(V[3])
print("Afino ", afino)

def TriD(xx, yy, T1, T2):
    U,D,V = np.linalg.svd(jednacine(xx,yy,T1,T2))
    Vp = V[3]
    VAfino = UAfine(Vp)
    return VAfino

#print(TriD(x1,y1,T1, T2))
#print("U")
#print(U)
#print()
#print("D")
#print(D)
#print()
#print("V")
#print(V)

slika1 = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16])
slika2 = np.array([y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16])


rekonstruisane = []
for i in range(len(slika1)):
    tmp = TriD(slika1[i], slika2[i], T1, T2)
    rekonstruisane.append(tmp)
    print(tmp)

#print(rekonstruisane)
dig = np.eye(3)
dig[2][2] = 400
rekonstruisane400 = np.zeros((16,3))
print("Rekonstruisano400:")
for i in range(len(rekonstruisane)):
    rekonstruisane400[i]=dig.dot(rekonstruisane[i])
    print(rekonstruisane400[i])
