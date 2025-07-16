import numpy as np
import random
import math
import cv2
import os

class Nodes:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_parent = []
        self.y_parent = []

class RRTree:
    def __init__(self, map, offset, iteration):
        # create black obstacle map
        img = cv2.imread(map, cv2.IMREAD_GRAYSCALE) 
        img[np.where((img[:,:] != 255))] = 0
        dilation = cv2.erode(img, np.ones((3,3), np.uint8), iterations=12)
        self.dilation = dilation 
        self.map = cv2.imread(map) 
        self.iteration = iteration
        self.offset = offset
        self.start_nodes = []  # Initialize empty list for start nodes
        self.goal_nodes = []   # Initialize empty list for goal nodes
        self.target = " "
        self.start = None
        self.goal = None

    def set_target(self, target):
        object = {
            "rack": (748, 284),
            "cushion": (1039, 454),
            "stair": (1075, 80),
            "cooktop": (380, 545),
            "sofa": (1074, 471)
        }
        x = object[target][0]
        y = object[target][1]
        self.target = target
        self.goal = (x, y)
    
    # generate a random point in the 2D image pixel
    def random_point(self, height, width):
        random_x = random.randint(0, width)
        random_y = random.randint(0, height)
        return (random_x, random_y)
    
    # Calculate L2 distance between new point and nearest node
    def distance(self, x1, y1, x2, y2):
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    
    # Calculate angle between new point and nearest node
    def angle(self, x1, y1, x2, y2):
        return math.atan2(y2 - y1, x2 - x1)
        
    # return the index of point among the points in node_list,
    # the one that closest to point (x,y)
    def nearest_node(self, x, y, node_list):
        distances = [self.distance(x, y, node.x, node.y) for node in node_list]
        return distances.index(min(distances))
    
    # check collision
    def collision(self, x1, y1, x2, y2, img):
        color = []
        if int(x1) == int(x2) and int(y1) == int(y2):
            return False

        line_points = np.column_stack((np.linspace(x1, x2, num=100), np.linspace(y1, y2, num=100)))

        for point in line_points:
            x, y = map(int, point)
            color.append(img[y, x])

        return 0 in color
    
    # check the collision with obstacle and the trajectory
    # if there is no collision, add the new point to the tree
    def expansion(self, random_x, random_y, nearest_x, nearest_y, offset, map):
        theta = self.angle(nearest_x, nearest_y, random_x, random_y)
        new_node_x = nearest_x + offset * np.cos(theta)
        new_node_y = nearest_y + offset * np.sin(theta)
        height, width = map.shape

        if new_node_y < 0 or new_node_y >= height or new_node_x < 0 or new_node_x >= width:
            expand_connect = False
        else:
            if self.collision(new_node_x, new_node_y, nearest_x, nearest_y, map):
                expand_connect = False
            else:
                expand_connect = True
        return (new_node_x, new_node_y, expand_connect)

    # find the nearest point to the newest point in Tree_A to Tree_B
    def extend(self, Tree_A, Tree_B, map):
        x = Tree_A[-1].x
        y = Tree_A[-1].y
        nearest_index_b = self.nearest_node(x, y, Tree_B)
        nearest_x = Tree_B[nearest_index_b].x
        nearest_y = Tree_B[nearest_index_b].y
        return not self.collision(x, y, nearest_x, nearest_y, map), nearest_index_b

    def rrt_path(self, target):
        if target != " ":
            self.set_target(target)
            print("Successfully found the target.")
        else:
            print("No object target.")

        height, width = self.dilation.shape

        # Initialize the start node with the user-defined start point
        self.start_nodes.append(Nodes(self.start[0], self.start[1]))
        self.start_nodes[0].x_parent.append(self.start[0])
        self.start_nodes[0].y_parent.append(self.start[1])

        # Initialize the goal node based on the target object
        self.goal_nodes.append(Nodes(self.goal[0], self.goal[1]))
        self.goal_nodes[0].x_parent.append(self.goal[0])
        self.goal_nodes[0].y_parent.append(self.goal[1])

        grow_tree = -1
        i = 1
        path = []  # Initialize the path variable
        while i < self.iteration:
            Tree_A = self.start_nodes if grow_tree == -1 else self.goal_nodes
            Tree_B = self.goal_nodes if grow_tree == -1 else self.start_nodes

            random_x, random_y = self.random_point(height, width)

            nearest_index = self.nearest_node(random_x, random_y, Tree_A)
            nearest_x, nearest_y = Tree_A[nearest_index].x, Tree_A[nearest_index].y
            new_node_x, new_node_y, expand_connect = self.expansion(random_x, random_y, nearest_x, nearest_y, self.offset, self.dilation)

            if expand_connect:
                new_node = Nodes(new_node_x, new_node_y)
                Tree_A.append(new_node)  # Append the new node to the tree
                Tree_A[-1].x_parent = Tree_A[nearest_index].x_parent.copy()  # Copy the parent
                Tree_A[-1].y_parent = Tree_A[nearest_index].y_parent.copy() 
                Tree_A[-1].x_parent.append(new_node_x)
                Tree_A[-1].y_parent.append(new_node_y)

                # Optional: visualize tree growth
                color = (0, 0, 255)
                cv2.circle(self.map, (int(new_node_x), int(new_node_y)), 2, color, thickness=3, lineType=8)
                cv2.line(self.map, (int(new_node_x), int(new_node_y)), (int(Tree_A[nearest_index].x), int(Tree_A[nearest_index].y)), color, thickness=1, lineType=8)
                cv2.imshow("RRT Visualization", self.map)
                cv2.waitKey(1)

                # Find the node closest to the new node in Tree_B
                nearest_index_b = self.nearest_node(new_node_x, new_node_y, Tree_B)
                nearest_x_b = Tree_B[nearest_index_b].x
                nearest_y_b = Tree_B[nearest_index_b].y
                
                # Check the connection between the new node and the nearest node in Tree_B
                if not self.collision(new_node_x, new_node_y, nearest_x_b, nearest_y_b, self.dilation):
                    print("Path is successfully formulated between the two trees.")
                    path = []
                    
                    # Constructing the path
                    for j in range(len(Tree_A[-1].x_parent)):
                        path.append((Tree_A[-1].x_parent[j], Tree_A[-1].y_parent[j]))
                    Tree_B[nearest_index_b].x_parent.reverse()
                    Tree_B[nearest_index_b].y_parent.reverse()    
                    for j in range(len(Tree_B[nearest_index_b].x_parent)):
                        path.append((Tree_B[nearest_index_b].x_parent[j], Tree_B[nearest_index_b].y_parent[j]))

                    # Visualize the final path in green
                    for j in range(len(path) - 1):
                        cv2.line(self.map, (int(path[j][0]), int(path[j][1])), (int(path[j + 1][0]), int(path[j + 1][1])), (0, 255, 0), thickness=2, lineType=8)

                    cv2.imshow("Final Path", self.map)
                    cv2.waitKey(0)  # Wait until a key is pressed
                    # Save the image with the path
                    if self.target == " ":
                        cv2.imwrite("birrt.jpg", self.map)
                    else:
                        cv2.imwrite("bi-RRT_Path/" + self.target + ".jpg", self.map)
                    
                    break
                
            grow_tree = -grow_tree
            self.start_nodes = Tree_A
            self.goal_nodes = Tree_B
            i += 1
            
if __name__ == '__main__':
    # Get the map path from the file
    current_dir = os.path.dirname(__file__)
    map_path = os.path.join(current_dir, "map.png")

    # Create an instance of the RRTree class with the map
    rrt = RRTree(map_path, offset=5, iteration=5000)

    # Load the map image for visualization
    rrt.map = cv2.imread(map_path)

    # Open the map image in a window
    cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Map", rrt.map)

    # Set the start point
    cv2.setMouseCallback("Map", lambda event, x, y, flags, param: rrt.start_nodes.append(Nodes(x, y)) if event == cv2.EVENT_LBUTTONDBLCLK else None)

    # Wait for the user to double-click to set the start point
    print("Double click to set the start point.")
    cv2.waitKey(0)

    # Store the chosen start point
    rrt.start = (rrt.start_nodes[-1].x, rrt.start_nodes[-1].y)

    # Prompt for target name
    target_name = input("Enter target name (rack, cushion, stair, cooktop, sofa): ")
    rrt.rrt_path(target_name)
    cv2.destroyAllWindows()
