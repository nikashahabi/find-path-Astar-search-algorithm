from math import sqrt
import math


class FibonacciHeap:

    # internal node class
    class Node:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.parent = self.child = self.left = self.right = None
            self.degree = 0
            self.mark = False

    # function to iterate through a doubly linked list
    def iterate(self, head):
        node = stop = head
        flag = False
        while True:
            if node == stop and flag is True:
                break
            elif node == stop:
                flag = True
            yield node
            node = node.right

    # pointer to the head and minimum node in the root list
    root_list, min_node = None, None

    # maintain total node count in full fibonacci heap
    total_nodes = 0

    # return min node in O(1) time
    def find_min(self):
        return self.min_node

    # extract (delete) the min node from the heap in O(log n) time
    # amortized cost analysis can be found here (http://bit.ly/1ow1Clm)
    def extract_min(self):
        z = self.min_node
        if z is not None:
            if z.child is not None:
                # attach child nodes to root list
                children = [x for x in self.iterate(z.child)]
                for i in range(0, len(children)):
                    self.merge_with_root_list(children[i])
                    children[i].parent = None
            self.remove_from_root_list(z)
            # set new min node in heap
            if z == z.right:
                self.min_node = self.root_list = None
            else:
                self.min_node = z.right
                self.consolidate()
            self.total_nodes -= 1
        return z

    # insert new node into the unordered root list in O(1) time
    # returns the node so that it can be used for decrease_key later
    def insert(self, key, value=None):
        n = self.Node(key, value)
        n.left = n.right = n
        self.merge_with_root_list(n)
        if self.min_node is None or n.key < self.min_node.key:
            self.min_node = n
        self.total_nodes += 1
        return n

    # modify the key of some node in the heap in O(1) time
    def decrease_key(self, x, k):
        if k > x.key:
            return None
        x.key = k
        y = x.parent
        if y is not None and x.key < y.key:
            self.cut(x, y)
            self.cascading_cut(y)
        if x.key < self.min_node.key:
            self.min_node = x

    # merge two fibonacci heaps in O(1) time by concatenating the root lists
    # the root of the new root list becomes equal to the first list and the second
    # list is simply appended to the end (then the proper min node is determined)
    def merge(self, h2):
        H = FibonacciHeap()
        H.root_list, H.min_node = self.root_list, self.min_node
        # fix pointers when merging the two heaps
        last = h2.root_list.left
        h2.root_list.left = H.root_list.left
        H.root_list.left.right = h2.root_list
        H.root_list.left = last
        H.root_list.left.right = H.root_list
        # update min node if needed
        if h2.min_node.key < H.min_node.key:
            H.min_node = h2.min_node
        # update total nodes
        H.total_nodes = self.total_nodes + h2.total_nodes
        return H

    # if a child node becomes smaller than its parent node we
    # cut this child node off and bring it up to the root list
    def cut(self, x, y):
        self.remove_from_child_list(y, x)
        y.degree -= 1
        self.merge_with_root_list(x)
        x.parent = None
        x.mark = False

    # cascading cut of parent node to obtain good time bounds
    def cascading_cut(self, y):
        z = y.parent
        if z is not None:
            if y.mark is False:
                y.mark = True
            else:
                self.cut(y, z)
                self.cascading_cut(z)

    # combine root nodes of equal degree to consolidate the heap
    # by creating a list of unordered binomial trees
    def consolidate(self):
        A = [None] * int(math.log(self.total_nodes) * 2)
        nodes = [w for w in self.iterate(self.root_list)]
        for w in range(0, len(nodes)):
            x = nodes[w]
            d = x.degree
            while A[d] != None:
                y = A[d]
                if x.key > y.key:
                    temp = x
                    x, y = y, temp
                self.heap_link(y, x)
                A[d] = None
                d += 1
            A[d] = x
        # find new min node - no need to reconstruct new root list below
        # because root list was iteratively changing as we were moving
        # nodes around in the above loop
        for i in range(0, len(A)):
            if A[i] is not None:
                if A[i].key < self.min_node.key:
                    self.min_node = A[i]

    # actual linking of one node to another in the root list
    # while also updating the child linked list
    def heap_link(self, y, x):
        self.remove_from_root_list(y)
        y.left = y.right = y
        self.merge_with_child_list(x, y)
        x.degree += 1
        y.parent = x
        y.mark = False

    # merge a node with the doubly linked root list
    def merge_with_root_list(self, node):
        if self.root_list is None:
            self.root_list = node
        else:
            node.right = self.root_list.right
            node.left = self.root_list
            self.root_list.right.left = node
            self.root_list.right = node

    # merge a node with the doubly linked child list of a root node
    def merge_with_child_list(self, parent, node):
        if parent.child is None:
            parent.child = node
        else:
            node.right = parent.child.right
            node.left = parent.child
            parent.child.right.left = node
            parent.child.right = node

    # remove a node from the doubly linked root list
    def remove_from_root_list(self, node):
        if node == self.root_list:
            self.root_list = node.right
        node.left.right = node.right
        node.right.left = node.left

    # remove a node from the doubly linked child list
    def remove_from_child_list(self, parent, node):
        if parent.child == parent.child.right:
            parent.child = None
        elif parent.child == node:
            parent.child = node.right
            node.right.parent = parent
        node.left.right = node.right
        node.right.left = node.left

    # checks for a node in the heap with the given value
    def searchWithValue(self, value, currentNode):
        if currentNode is not None:
            # print(currentNode.value)
            # print(currentNode.key)
            if currentNode.value == value:
                return currentNode
            siblings = [x for x in self.iterate(currentNode)]
            for i in range(0, len(siblings)):
                # print(siblings[i].value)
                # print(siblings[i].key)
                if siblings[i].value == value:
                    return siblings[i]
                if self.searchWithValue(value, siblings[i].child) is not None:
                    return self.searchWithValue(value, siblings[i].child)
        return None




class Cell:
    'class for each cell. represents its state'
    def __init__(self):
        self.parent = False
        self.parenti = None
        self.parentj = None
        self.f = None
        'f = g + h'
        self.g = None
        'g is the cost function.'
        self.h = None
        'h is the heuristic function and equals the Euclidean Distance to the goal.'


def isCellDestination(row, col, rowDes, colDes):
    'if a cell is our destination.'
    return row == rowDes and col == colDes


def isCellValid(r, c, row, col):
    'if a cell is valid in the grid.'
    return 0 <= r < row and 0 <= c < col


def isCellBlocked(row, col, grid):
    'if a cell is blocked'
    return grid[row][col] == 0


def HValue(row, col, rowDes, colDes):
    'calculates the H value of a cell'
    return sqrt((row - rowDes) * (row - rowDes) + (col - colDes) * (col - colDes))


def tracePath(cells, rowDes, colDes):
    'traces the path from source to the destination'
    stack = []
    currentRow = rowDes
    currentCol = colDes
    while cells[currentRow][currentCol].parent is True:
        stack.append((currentRow, currentCol))
        tempRow = currentRow
        currentRow = cells[currentRow][currentCol].parenti
        currentCol = cells[tempRow][currentCol].parentj
    stack.append((currentRow, currentCol))
    print("The path to the destination:")
    while stack:
        print(stack[-1])
        stack.pop()


def calculateCost(pi, pj, i, j):
    'calculates the cost from a cell in (pi, pj) to a cell in (i, j)'
    if abs(pi - i) + abs(pj - j) == 2:
        return 1.414
    else:
        return 1


def inserIntoFrontier(i, j, pi, pj, cells, grid, frontier, explored, rowDes, colDes):
    'checks if the cell is valid and not blocked and not in explored.'
    if isCellValid(i, j, len(grid), len(grid[0])) and isCellBlocked(i, j, grid) is False and explored[i][j] is False:
        print("the (" , str(i), ", ", str(j), ") neighbor is not in explored")
        gNew = cells[pi][pj].g + calculateCost(pi, pj, i, j)
        hNew = HValue(i, j, rowDes, colDes)
        fNew = gNew + hNew
        'if the accepted cell is not in frontier insert it in'
        'if the accepted cell in in frontier with higher f decrease its key'
        if not cells[i][j].parent:
            print("the cell was not in frontier. it is now inserted into frontier")
            cells[i][j].parenti = pi
            cells[i][j].parentj = pj
            cells[i][j].f = fNew
            cells[i][j].g = gNew
            cells[i][j].h = hNew
            print("its f, g and h are:",fNew, gNew, hNew)
            cells[i][j].parent = True
            frontier.insert(cells[i][j].f, (i, j))
        elif frontier.searchWithValue((i, j), frontier.root_list) is not None:
            print("the cell is in frontier")
            node = frontier.searchWithValue((i, j),frontier.root_list)
            if frontier.decrease_key(node, fNew) is not None:
                print("the cell is in frontier with higher f. its key will decrease")
                print("its f, g and h are:",fNew, gNew, hNew)
                cells[i][j].parenti = pi
                cells[i][j].parentj = pj
                cells[i][j].f = fNew
                cells[i][j].g = gNew
                cells[i][j].h = hNew
                cells[i][j].parent = True
            else:
                print("the cell is in frontier but not with higher f. nothing will happen")







def AStarSearch (grid, rowDes, colDes, rowStart, colStart):
    'performs the a star search algorithm. start to des'
    rows = len(grid)
    columns = len(grid[0])
    if not isCellValid(rowDes, colDes, rows, columns):
        print("Destination is not valid")
        return
    if not isCellValid(rowStart, colStart, rows, columns):
        print("Source is not valid")
        return
    if isCellBlocked(rowDes, colDes, grid):
        print("Destination is blocked")
        return
    if isCellBlocked(rowStart, colStart, grid):
        print("Source is blocked")
        return
    'explored is the closed list. a binary hash'
    explored = [[False] * columns for i in range(rows)]
    'cells is used to store info of each cell in the grid'
    cells = [[Cell() for i in range(columns)] for j in range(rows)]
    cells[rowStart][colStart].f = 0
    cells[rowStart][colStart].g = 0
    'frontier is the open list. a fibonacci heap'
    frontier = FibonacciHeap()
    frontier.insert(cells[rowStart][colStart].f, (rowStart, colStart))
    while frontier.total_nodes > 0:
        minNode = frontier.extract_min()
        r, c = minNode.value
        print("extracted node")
        print(r, c)
        'we have now explored the min node in the fibonacci heap'
        explored[r][c] = True
        'checks to see if minNode is the destination'
        if isCellDestination(r, c, rowDes, colDes):
            print("the optimal path to the destination is found")
            tracePath(cells, rowDes, colDes)
            return
        'check to see if you can insert each of 8 successors into the frontier'
        inserIntoFrontier(r - 1, c - 1, r, c, cells, grid, frontier, explored, rowDes, colDes)
        inserIntoFrontier(r - 1, c, r, c, cells, grid, frontier, explored, rowDes, colDes)
        inserIntoFrontier(r - 1, c + 1, r, c, cells, grid, frontier, explored, rowDes, colDes)
        inserIntoFrontier(r, c - 1, r, c, cells, grid, frontier, explored, rowDes, colDes)
        inserIntoFrontier(r, c + 1, r, c, cells, grid, frontier, explored, rowDes, colDes)
        inserIntoFrontier(r + 1, c - 1, r, c, cells, grid, frontier, explored, rowDes, colDes)
        inserIntoFrontier(r + 1, c, r, c, cells, grid, frontier, explored, rowDes, colDes)
        inserIntoFrontier(r + 1, c + 1, r, c, cells, grid, frontier, explored, rowDes, colDes)
    print("No path was found to the destination")



grid = [ [ 1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
         [ 1, 1, 1, 0, 1, 1, 1, 0, 1, 1 ],
         [ 1, 1, 1, 0, 1, 1, 0, 1, 0, 1 ],
         [ 0, 0, 1, 0, 1, 0, 0, 0, 0, 1 ],
         [ 1, 1, 1, 0, 1, 1, 1, 0, 1, 0 ],
         [ 1, 0, 1, 1, 1, 1, 0, 1, 0, 0 ],
         [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 1 ],
         [ 1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
         [ 1, 1, 1, 0, 0, 0, 1, 0, 0, 1 ] ]
grid2 = [[1, 0, 0],
        [1, 1, 1],
        [0, 0, 1]]
AStarSearch(grid, 0, 2, 8, 6)

