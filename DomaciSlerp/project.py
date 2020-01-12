
import functions as f 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import math

#animation settings

TIMER_ID = 0
TIMER_INTERVAL = 20
t = 0
tm = 30
animation = False

#object parameters

c1 = [-4, 5,5]  #begin
c2 = [1, -5, -5]  #end

angles1 = [math.pi/6, math.pi/3, math.pi/3]
angles2 = [math.pi, math.pi/3, 2*math.pi/5]

q = []
quat1 = []
quat2 = []


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(500, 500)
    glutCreateWindow("TeapotSlerp")
    
    glLineWidth(1.5)
    
    glutKeyboardFunc(keyboard)
    glutDisplayFunc(display)
    if animation:
        glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)
    
    
    glClearColor(1, 1, 1, 1)
    glEnable(GL_DEPTH_TEST)
    
    global quat1
    global quat2
    
    A = f.Euler2A(angles1[0], angles1[1], angles1[2])
    p, angle = f.AxisAngle(A)
    quat1 = f.AngleAxis2Q(p, angle)
    
    A = f.Euler2A(angles2[0], angles2[1], angles2[2])
    p, angle = f.AxisAngle(A)
    quat2 = f.AngleAxis2Q(p, angle)
    
    
    glutMainLoop()
    return


def keyboard(key, x, y):
    
    global animation
    
    if ord(key) == 27:
        sys.exit(0)
        
    if ord(key) == ord('g'):
        if not animation:
            glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)
            animation = True
        animation = True
    
    if ord(key) == ord('s'):
        animation = False
            
            
def timer(value):
    if value != TIMER_ID:
        return
    
    global t
    global tm 
    global animation
    global q
    
    t += 0.2
        
    if t >= tm:
        t = 0
        animation = False
        return
    
    glutPostRedisplay()
    
    if animation:
        glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)

def draw(position, angles):
    glPushMatrix()
    
    glColor3f(0.33, 0.75, 0.88)
    
    glTranslatef(position[0], position[1], position[2])
    
    A = f.Euler2A(angles[0], angles[1], angles[2])
    p, angle = f.AxisAngle(A)
    
    glRotatef(angle/math.pi*180, p[0], p[1], p[2])
    glutSolidTeapot(1)
    
    draw_axis(2)
    
    glPopMatrix()

def draw_animated():
    global q
    global t
    global tm
    
    glPushMatrix()
    glColor3f(0.33, 0.75, 0.88)
    
    position = []
    
    position.append((1-t/tm)*c1[0] + (t/tm)*c2[0])
    position.append((1-t/tm)*c1[1] + (t/tm)*c2[1])
    position.append((1-t/tm)*c1[2] + (t/tm)*c2[2])

    glTranslatef(position[0], position[1], position[2])
    
    q = f.slerp(quat1, quat2, tm, t)
    
    p, angle = f.Q2AxisAngle(q)
    
    glRotatef(angle/math.pi*180, p[0], p[1], p[2])
    
    glutSolidTeapot(1)
    draw_axis(2)
    
    glPopMatrix()
    
def draw_axis(size):
    
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0,0,0)
    glVertex3f(size,0,0)
        
    glColor3f(0,1,0)
    glVertex3f(0,0,0)
    glVertex3f(0,size,0)
        
    glColor3f(0,0,1)
    glVertex3f(0,0,0)
    glVertex3f(0,0,size)
    
    glEnd()
    

def display():
    
    global q
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    window_width = 500
    window_height = 500
    
    glViewport(0, 0, window_width, window_height)
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45,float( window_width) /  window_height, 1, 30)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(15, 12, 15, 0, 0, 0, 0, 1, 0)
    
    #draw_axis(15) 
    
    draw(c1, angles1)
    draw(c2, angles2) 
    
    draw_animated()
    
    glutSwapBuffers()


if __name__ == '__main__': 
    main()