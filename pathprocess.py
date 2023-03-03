from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Array
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


# custom process class
class CustomProcess(Process):
    # override the constructor
    def __init__(self, obstaclelist, arrresolution):
        # execute the base constructor
        Process.__init__(self)
        # initialize integer attribute
        self.obstaclelist = obstaclelist
        self.arrresolution = arrresolution
        self.beginning = (arrresolution//2 -1, 0)
        self.end = (arrresolution//2, arrresolution)
        self.path = Array('i', self.run())
        #self.stringpath = Value('s',"hello")

    # override the run function
    def run(self):
        # block for a moment
        #x1 = 2
        #x2 = 5
        #y1 = 5
        #y2 = 9
        obstaclelist = [[20, 45, 35, 40], [5, 20, 20, 30]]
        map = np.full([self.arrresolution, self.arrresolution], 1)
        for i in self.obstaclelist:

            innerarray = np.full(self.arrresolution, 1)
            print(innerarray)
            innerarray[i[0]:i[1]] = 0
            print(innerarray)
            map[i[2]:i[3]] = innerarray
            print(map)

        grid = Grid(matrix=map)

        start = grid.node(self.beginning[0], self.beginning[1])
        end = grid.node(self.end[0] - 1, self.end[1] - 1)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, runs = finder.find_path(start, end, grid)
        print('operations:', runs, 'path length:', len(path))
        print(grid.grid_str(path=path, start=start, end=end))
        print(f"path:{path}")

        thelongpath = []
        foundpath = []
        for i in path:
            thelongpath.append(path[path.index(i)][0])
        for i in path:
            thelongpath.append(path[path.index(i)][1])
            #for i in thelongpath:
                #foundpath.append(list(i))

        print([i for i in thelongpath])
        print(len(thelongpath))
        return list(thelongpath)

# entry point
if __name__ == '__main__':
    # create the process
    mylist = []
    process = CustomProcess(mylist, 50)
    # start the process
    process.start()
    # wait for the process to finish
    print('Waiting for the child process to finish')
    # block until child process is terminated
    process.join()
    # report the process attribute
    print(f'Parent got: {process.obstaclelist}')
    print(process.path)
    pathlistx = [process.path[i]*(1000//50) for i in range(len(process.path)//2)]
    pathlisty = [process.path[i]*(1000//50) for i in range(len(process.path)//2, len(process.path))]
    print(f"pathlistx:{pathlistx}, len:{len(pathlistx)}")
    print(f"pathlisty:{pathlisty}, len:{len(pathlisty)}")
    pathlist = []
    for i in range(len(pathlistx)):
        pathlist.append((pathlistx[i], pathlisty[i]))

    print(pathlist)


    #print([(i.x, i.y) for i in process.path])

"""
from multiprocessing import Process, Manager

def f(general_exchange, path_exchange, x1, x2, y1, y2):

    x1 = 2
    x2 = 5
    y1 = 5
    y2 = 9

    map = np.full([100, 100], 1)
    # innerarray = np.zeros([1,10], dtype=int)
    # innerarray = np.empty(10)
    # innerarray.fill(1)
    innerarray = np.full(10, 1)
    print(innerarray)
    innerarray[x1 - 1:x2] = 0
    print(innerarray)
    map[y1 - 1:y2] = innerarray
    print(map)

    grid = Grid(matrix=map)

    start = grid.node(0, 0)
    end = grid.node(10 - 1, 10 - 1)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)
    print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start, end=end))
    print(f"path:{path}")

    for i in path:
        path_exchange.append(path[path.index(i)])

if __name__ == '__main__':
    with Manager() as manager:
        general_exchange = manager.dict()
        path_exchange = manager.list()

        p = Process(target=f, args=(general_exchange, path_exchange, mapx1[0], mapx2[0], mapy1[1], mapy2[1]))
        p.start()
        p.join()

        print(general_exchange)
        print(path_exchange)
"""