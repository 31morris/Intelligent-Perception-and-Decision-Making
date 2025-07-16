import numpy as np
import argparse
import os
import random
import math
import cv2



class RRTree:
    def __init__(self, map, offset, iteration):
        # create black obstacle map
        img = cv2.imread(map, cv2.IMREAD_GRAYSCALE) 
        img[np.where((img[:, :] != 255))] = 0
        dilation = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=12)
        self.dilation = dilation 
        self.map = cv2.imread(map) 
        self.iteration = iteration
        self.start = [0, 0]
        self.goal = [0, 0]
        self.offset = offset
        self.start_nodes = [()]
        self.target = " "

        # 顯示並保存 Dilation map
        cv2.imshow("Dilation Map", self.dilation)
        cv2.imwrite("dilation_map.jpg", self.dilation)  # 將膨脹圖保存為 jpg 圖像
        print('dilation_map')
        cv2.waitKey(0)  # 暫停，等候按鍵後繼續
        cv2.destroyWindow("Dilation Map")  # 關閉顯示的膨脹圖窗口

    def set_target(self, target):
        object_positions = {
            "rack": (748, 284),
            "cushion": (1039, 454),
            "stair": (1075, 80),
            "cooktop": (380, 545),
            "sofa":(1074, 471)
        }
        if target in object_positions:
            x, y = object_positions[target]
            self.target = target
            self.goal = (x, y)
        else:
            print(f"Target '{target}' not found. Please use one of the following: {list(object_positions.keys())}")

    def random_point(self, height, width):
        random_x = random.randint(0, width)
        random_y = random.randint(0, height)
        return (random_x, random_y)

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def angle(self, x1, y1, x2, y2):
        return math.atan2(y2 - y1, x2 - x1)

    def nearest_node(self, x, y, node_list):
        distances = [self.distance(x, y, node.x, node.y) for node in node_list]
        return distances.index(min(distances))

    def collision(self, x1, y1, x2, y2, img):
        color = []
        if int(x1) == int(x2) and int(y1) == int(y2):
            return False

        line_points = np.column_stack((np.linspace(x1, x2, num=100), np.linspace(y1, y2, num=100)))

        for point in line_points:
            x, y = map(int, point)
            color.append(img[y, x])

        return 0 in color  # 如果有障礙物，返回 True

    def expansion(self, random_x, random_y, nearest_x, nearest_y, offset, map):
        theta = self.angle(nearest_x, nearest_y, random_x, random_y)
        new_node_x = nearest_x + offset * np.cos(theta)
        new_node_y = nearest_y + offset * np.sin(theta)
        width, height = map.shape

        if new_node_y < 0 or new_node_y > width or new_node_x < 0 or new_node_x > height:
            expand_connect = False
        else:
            expand_connect = not self.collision(new_node_x, new_node_y, nearest_x, nearest_y, map)
        return (new_node_x, new_node_y, expand_connect)

    def extend(self, Tree, goal_node, map):
        x = Tree[-1].x
        y = Tree[-1].y
        nearest_index_b = self.nearest_node(x, y, goal_node)
        nearest_x = goal_node[nearest_index_b].x
        nearest_y = goal_node[nearest_index_b].y

        return (not self.collision(x, y, nearest_x, nearest_y, map), nearest_index_b)

    def rrt_path(self, target):
        if target != " ":
            self.set_target(target)

        height, width = self.dilation.shape

        self.start_nodes[0] = Nodes(self.start[0], self.start[1])
        self.start_nodes[0].x_parent.append(self.start[0])
        self.start_nodes[0].y_parent.append(self.start[1])

        i = 1
        while i < self.iteration:
            Tree_A = self.start_nodes.copy()
            random_x, random_y = self.random_point(height, width)

            nearest_index = self.nearest_node(random_x, random_y, Tree_A)
            nearest_x, nearest_y = Tree_A[nearest_index].x, Tree_A[nearest_index].y
            new_node_x, new_node_y, expand_connect = self.expansion(random_x, random_y, nearest_x, nearest_y, self.offset, self.dilation)

            if expand_connect:
                Tree_A.append(Nodes(new_node_x, new_node_y))
                Tree_A[i].x_parent = Tree_A[nearest_index].x_parent.copy()
                Tree_A[i].y_parent = Tree_A[nearest_index].y_parent.copy()
                Tree_A[i].x_parent.append(new_node_x)
                Tree_A[i].y_parent.append(new_node_y)

                cv2.circle(self.map, (int(new_node_x), int(new_node_y)), 2, (0, 0, 255), thickness=3, lineType=8)
                cv2.line(self.map, (int(new_node_x), int(new_node_y)),
                         (int(Tree_A[nearest_index].x), int(Tree_A[nearest_index].y)),
                         (0, 255, 0), thickness=1, lineType=8)
                cv2.imwrite("RRT_Path/" + str(i) + ".jpg", self.map)
                cv2.imshow("image", self.map)
                cv2.waitKey(1)

                extend_connect, index = self.extend(Tree_A, [Nodes(self.goal[0], self.goal[1])], self.dilation)

                if extend_connect:
                    print("finish")
                    path = []
                    cv2.line(self.map, (int(new_node_x), int(new_node_y)),
                             (int(self.goal[0]), int(self.goal[1])), (0, 255, 0), thickness=1, lineType=8)

                    for i in range(len(Tree_A[-1].x_parent)):
                        path.append((Tree_A[-1].x_parent[i], Tree_A[-1].y_parent[i]))

                    Nodes(self.goal[0], self.goal[1]).x_parent.reverse()
                    Nodes(self.goal[0], self.goal[1]).y_parent.reverse()

                    for i in range(len(Nodes(self.goal[0], self.goal[1]).x_parent)):
                        path.append((Nodes(self.goal[0], self.goal[1]).x_parent[i],
                                     Nodes(self.goal[0], self.goal[1]).y_parent[i]))

                    cv2.line(self.map, (int(Tree_A[-1].x_parent[-1]), int(Tree_A[-1].y_parent[-1])),
                             (int(self.goal[0]), int(self.goal[1])), (255, 0, 0), thickness=2, lineType=8)

                    for i in range(len(path) - 1):
                        cv2.line(self.map, (int(path[i][0]), int(path[i][1])),
                                 (int(path[i + 1][0]), int(path[i + 1][1])), (255, 0, 0), thickness=2, lineType=8)

                    cv2.waitKey(1)
                    cv2.imwrite("RRT_Path/" + str(i) + ".jpg", self.map)
                    if self.target == " ":
                        cv2.imwrite("rrt.jpg", self.map)
                    else:
                        cv2.imwrite("RRT_Path/" + self.target + ".jpg", self.map)
                    break
            else:
                continue

            i += 1
            self.start_nodes = Tree_A.copy()

        if i == self.iteration:
            print("Failed to find the path")
        print("Number of iterations: ", i)

        path = np.asarray(path)
        # 假設智能體的座標範圍
        agent_x_range = 16  # 智能體的X軸範圍為 [-8, 8]，因此總長度為 16
        agent_y_range = 12  # 智能體的Y軸範圍為 [-6, 6]，因此總高度為 12

        # 計算縮放因子
        width_transform = width / agent_x_range
        height_transform = height / agent_y_range

        # 更新坐標轉換
        path[:, 0] = path[:, 0] / width_transform - (agent_x_range / 2)
        path[:, 1] = (agent_y_range / 2) - path[:, 1] / height_transform
        return path
    
    def smooth_path(self, path, dilation_map):
        smooth_path = [path[0]]  # 將起點放入平滑路徑
        i = 0  # 開始於路徑的起始點
        
        while i < len(path) - 1:
            j = len(path) - 1  # 嘗試直接連接到終點
            while j > i:
                if not self.collision(path[i][0], path[i][1], path[j][0], path[j][1], dilation_map):
                    # 如果兩點之間沒有障礙物，則可以將它們直接連接
                    smooth_path.append(path[j])
                    i = j  # 移動到新點
                    break
                j -= 1  # 如果有障礙物，則嘗試連接到更近的點

        return smooth_path

class Nodes:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_parent = []
        self.y_parent = []
        
# click on the picture to get the point
def draw_circle(event, u, v, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(rrt.map, (u, v), 5, (255, 0, 0), -1)
        coordinates.append(u)
        coordinates.append(v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters:')
    parser.add_argument('-p', type=str, default='map.png', metavar='ImagePath',
                        action='store', dest='imagePath',
                        help='File path of the map image')
    parser.add_argument('-o', type=int, default=40, metavar='offset',
                        action='store', dest='offset',
                        help='Step size in RRT algorithm')
    args = parser.parse_args()

    rrt = RRTree(args.imagePath, args.offset, 1000)

    coordinates = []

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle)

    # click to set starting point
    print("set the starting point")
    while True:
        cv2.imshow("image", rrt.map)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    rrt.start = (coordinates[0], coordinates[1])

    # Get target object from user input
    target_input = input("輸入目標物（rack, cushion, sofa,stair, and cooktop） : ")
    rrt.set_target(target_input)

    path = rrt.rrt_path(rrt.target)

    # 路徑平滑化
    smooth_path = rrt.smooth_path(path, rrt.dilation)

    # 顯示平滑後的路徑
    for i in range(len(smooth_path) - 1):
        cv2.line(rrt.map, (int(smooth_path[i][0]), int(smooth_path[i][1])),
                 (int(smooth_path[i + 1][0]), int(smooth_path[i + 1][1])), (255, 0, 255), thickness=2, lineType=8)

     # 保存平滑路徑為 .npy 文件
    np.save("smooth_path.npy", smooth_path)  # 保存為 .npy 文件

    

    cv2.imshow("Smoothed Path", rrt.map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()