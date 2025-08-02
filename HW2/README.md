# pdm-f24-hw2
NYCU Perception and Decision Making 2024 Fall

Spec: https://drive.google.com/file/d/1YdGHuwxW4AOitrrfvNO55kCfEAbjpCMl/view?usp=drive_link

## Preparation
In your original pdm-f24 directory, `git pull` to get new `hw2` directory.

## Usage

```bash
python map.py
```
- Filtered point cloud data will be saved to:
 /home/morris/pdm-f24/hw2/filtered_points.npy
 /home/morris/pdm-f24/hw2/filtered_colors.npy
 
- A scatter plot image of the filtered point cloud will be saved to:
 /home/morris/pdm-f24/hw2/map.png


```bash
python RRT.py
```
- Dilation Map: Saved as dilation_map.jpg.
- Intermediate Path Images: Saved in the RRT_Path/ folder.
- Smoothed Path: Saved as smooth_path.npy.


```bash
python navigationcolor.py -f <target_object>
```
- Replace <script_name>.py with the name of your script file and <target_object> with one of the following options:
rack,cushion,sofa,stair,cooktop

- The program generates a video file of the agent's navigation and saves it to the video/ directory. The file is named according to the target object, e.g., rack.mp4.


```bash
python bi-rrt22.py
```
