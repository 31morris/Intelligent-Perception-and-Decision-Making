import cv2
import numpy as np

points = []

class Projection(object):
    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view (BEV) image
        """
        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, _ = self.image.shape
        self.points = points

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective (front) view image
        """
        bev_pixels = []
        new_pixels = []

        principal_point = [self.width // 2, self.height // 2]
        f = (self.width / 2) * (1 / np.tan(np.deg2rad(fov / 2)))  

        for point in self.points:
            pixel_x, pixel_y = principal_point[0] - point[0], principal_point[1] - point[1]
            bev_pixels.append([pixel_x, pixel_y, 1])

        bev_points = []
        for bev_pixel in bev_pixels:
            bev_point = [2.5 * bev_pixel[0] / f, 2.5 * bev_pixel[1] / f, 2.5]  
            bev_points.append(bev_point)

        bev_points = np.array(bev_points)

        theta_rad = np.deg2rad(theta)  # Convert theta to radians
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)

        transformation_matrix = np.array([
            [1, 0, 0, dx],
            [0, c, -s, 1.5],
            [0, s, c, dz],
            [0, 0, 0, 1]
        ])
        # Apply transformation
        front_points = np.dot(transformation_matrix, np.hstack((bev_points, np.ones((bev_points.shape[0], 1)))).T).T[:, :3]

        for front_point in front_points:
            if front_point[2] != 0:  # Prevent division by zero
                new_pixel_x = principal_point[0] - int(front_point[0] / front_point[2] * f)  
                new_pixel_y = principal_point[1] - int(front_point[1] / front_point[2] * f)  
                new_pixels.append([new_pixel_x, new_pixel_y])

        return new_pixels

    def show_image(self, new_pixels, img_name='projection_result.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective (front) view image.
        """
        new_image = cv2.fillPoly(self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        points.append([x, y])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

if __name__ == "__main__":
    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=90, dx=0, dz=0) 
    projection.show_image(new_pixels)
