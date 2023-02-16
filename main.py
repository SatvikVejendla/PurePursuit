import matplotlib.pyplot as plt
import math
import numpy as np

with open("points.txt") as f:
    data = f.read().splitlines()

    data = [(i.split(",")) for i in data]


    data = list(map(lambda x: ( float(x[0]), float(x[1]), float(x[2]) ), data))


def degrees(rads):
    return rads * 180 / np.pi

class Vector:

    def __init__(self, p1, p2):

        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]

        self.p1 = p1
        self.p2 = p2

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.magnitude = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        self.slope = (y2 - y1)/(x2 - x1)


        self.angle = math.atan((y2-y1)/(x2-x1))

    def __str__(self):
        return str(self.magnitude) + " magnitude,  " + str(self.angle * 180 / np.pi) + " degrees"


def dist2Points(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
def dist(vector, cur):
    slope = vector.slope

    v = Vector([vector.x1, vector.y1], [cur[0], cur[1]])
    ov = vector

    proj = projection(v, ov)

    newy = proj * np.sin(degrees(ov.angle))
    newx = proj * np.cos(degrees(ov.angle))
    return [newx + vector.x1, newy + vector.y1]


lookaheadD = 0.7

def lookahead_point(vector, cur):

    e = vector.p1
    l = vector.p2
    c = cur
    r = lookaheadD


    d = vector
    f = Vector(cur, e)

    a = dot(d,d)
    b = 2*dot(f, d)
    c = dot(f, f) - r * r

    discriminant = b*b - 4 * a * c

    if(discriminant < 0):
        return [[0,0]]

    else:
        
        discriminant = math.sqrt(discriminant)

        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)


        ans = []
        if(t1 >= 0 and t1 <= 1):
            ans.append([e[0] + (t1 * (d.x2 - d.x1)), e[1] + (t1 * (d.y2 - d.y1))])
        if(t2 >= 0 and t2 <= 1):
            ans.append([e[0] + (t2 * (d.x2 - d.x1)), e[1] + (t2 * (d.y2 - d.y1))])
        
        return ans

def closest_point(path, cur):
    
    d = dist(Vector(path[0], path[1]), cur)
    index = 0
    for i in range(1, len(path-1)):
        
        segment = Vector(path[i], path[i+1])

        distance = dist(segment, cur)

        if dist(path[i], cur) < d:
            d = dist(path[i], cur)
            index = i
    return data[index]


def dot(a, b):
    return (a.x2 - a.x1) * (b.x2 - b.x1) + (a.y2 - a.y1) * (b.y2 - b.y1)

def projection(a, b):
    dAngle = a.angle - b.angle

    
    vec = dot(a, b) / (a.magnitude**2)

    return vec



"""
cur = [2,1]

x = [i[0] for i in data]
y = [i[1] for i in data]

plt.scatter(x, y, color='g')
plt.plot(x, y, color='r')

#circle around cur variable
angle = np.linspace( 0 , 2 * np.pi , 150 ) 
x = lookaheadD * np.cos( angle ) 
y = lookaheadD * np.sin( angle ) 

plt.plot( x + cur[0] , y + cur[1] , color='b' )
closest = closest_point(data, cur)
plt.plot([cur[0], closest[0]], [cur[1], closest[1]], color='g')


print(lookahead_point(data, cur))
plt.axis('scaled')
plt.show()
"""

def drawVector(vec):
    plt.plot([vec.x1, vec.x2], [vec.y1, vec.y2], color='b')


def segmentDistance(vec, point):
    start = vec.p1

    dist = math.sqrt((point[0] - start[0])**2 + (point[1] - start[1])**2)

    return dist



cur = [1, 0]

stage = 0

path = [[0.5, 0], [0,0], [3,3], [5,2], [6,5], [7,3], [8, 9], [9, 6], [3,7]]




fig = plt.figure()


plt.xlim(0, 10)

plt.ylim(0, 10)

path = []



def main():
    global stage
    global cur

    cur = [path[0][0] - 0.2, path[0][1]]
    for i in range(len(path)-1):
        drawVector(Vector(path[i], path[i+1]))
        
    while(stage < len(path)-1):


        segend = path[stage+1]

        if(dist2Points(cur, segend) < lookaheadD):
            stage += 1
            if stage == len(path)-1:
                break

        vec = Vector(path[stage], path[stage+1])
        

        drawVector(vec)
        plt.plot(cur[0], cur[1], 'ro')



        closest = lookahead_point(vec, cur)

        lp = 0
        maxval = 0

        curDist = Vector(cur, [0,0]).magnitude

        for i in closest:
            segDist = segmentDistance(vec, i)
            
            if segDist > maxval:
                maxval = segDist
                lp = i


        plt.axis('scaled')

        speed = 0.7


        if(lp == 0):
            lp = [0,0]
        dx = lp[0] - cur[0]
        dy = lp[1] - cur[1]

        cur[0] += speed * dx
        cur[1] += speed * dy
        




        plt.pause(0.001)

        


        #0.5, 0.5
    print("FINAL POINT")
    while(cur != path[-1]):
        plt.plot(cur[0], cur[1], 'ro')
        plt.axis('scaled')
        plt.pause(0.00001)

        dx = path[-1][0] - cur[0]
        dy = path[-1][1] - cur[1]

        cur[0] += speed * dx
        cur[1] += speed * dy

    print("DONE")
    plt.show()

def onclick(event):

    if(event.button == 1):
        global ix, iy

        ix, iy = event.xdata, event.ydata

        print('x = %d, y = %d'%(ix, iy))

        global path
        path.append([ix, iy])

    if(len(path) > 1 and event.button == 3):
        fig.canvas.mpl_disconnect(cid)

        main()



cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


