import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import math
import argparse
import os


test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
json_path = "replica_v1/apartment_0/habitat/info_semantic.json"

# Read the mapping file from instance ID to semantic ID
with open(json_path, "r") as f:
    annotations = json.load(f)

# Convert the instance IDs to a list of semantic IDs
id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])

# Generate the semantic label list; set IDs less than 0 to 0
id_to_label = [0 if label < 0 else label for label in instance_id_to_semantic_label_id]
id_to_label = np.asarray(id_to_label)

# Define simulation settings parameters
sim_settings = {
    "scene": test_scene,         # The scene to be tested
    "default_agent": 0,          # The default agent ID
    "sensor_height": 1.5,        # The height of the sensor
    "width": 512,                # The width of the sensor image
    "height": 512,               # The height of the sensor image
    "sensor_pitch": 0,           # The pitch angle of the sensor
}


def rgb_to_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata(semantic_obs.flatten().astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.01) 
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0) 
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
path = np.load('smooth_path.npy') #load the path
start = path[0]
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([start[1], 0.0, start[0]])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        # Convert RGB image to BGR for display
        RGB_img = rgb_to_bgr(observations["color_sensor"])

        # Convert the semantic image
        SEIMEN_img = transform_semantic(id_to_label[observations["semantic_sensor"]])

        # Display the semantic image
        cv2.imshow("Semantic Image", SEIMEN_img)
        cv2.waitKey(1)

        # Find the pixel positions that match the target RGB value
        index = np.where((SEIMEN_img[:,:,0]==b) & (SEIMEN_img[:,:,1]==g) & (SEIMEN_img[:,:,2]==r))

        # Print the number of matching pixels found
        print(f"Found {len(index[0])} pixels matching the target color.")

        # Apply a red mask: overlay the target object with a red transparent mask
        if len(index[0]) != 0:
            red_mask = np.zeros_like(RGB_img)
            red_mask[index] = (0, 0, 255)  # Set the red mask
            # Reduce transparency to make the effect more visible
            RGB_img = cv2.addWeighted(RGB_img, 0.4, red_mask, 0.6, 0)

        # Display the result with the red mask
        cv2.imshow("RGB with Red Mask", RGB_img)
        cv2.waitKey(1)

        # Save the image to the video file
        videowriter.write(RGB_img)

        # Get the current camera state
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print(sensor_state.position[0], sensor_state.position[1], sensor_state.position[2], sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

        return sensor_state


###############################################################################
def driver(previous_position, current_position, destination):
    # Calculate the rotation movement
    # Determine the initial vector, use a default vector if there's no previous position
    if previous_position == []:
        initial_vector = np.array([-1, 0])
    else:
        initial_vector = np.array([current_position[0] - previous_position[0], current_position[1] - previous_position[1]])

    # Calculate the vector from the current position to the destination
    target_vector = np.array([destination[0] - current_position[0], destination[1] - current_position[1]])
    print(initial_vector, target_vector)

    # Use the cross product of the vectors to determine the rotation direction
    rotation_sign = initial_vector[0] * target_vector[1] - initial_vector[1] * target_vector[0]
    dot_product_value = initial_vector @ target_vector

    # Calculate the magnitude of the vectors
    magnitude_initial = math.sqrt(initial_vector[0]**2 + initial_vector[1]**2)
    magnitude_target = math.sqrt(target_vector[0]**2 + target_vector[1]**2)

    # Calculate the rotation angle
    angle_to_rotate = int(math.acos(dot_product_value / (magnitude_initial * magnitude_target)) / math.pi * 180)
    print(angle_to_rotate)
    print("Number of rotations", int(angle_to_rotate))

    # Determine whether to turn left or right based on the rotation direction
    rotation_action = "turn_left" if rotation_sign >= 0 else "turn_right"
    for _ in range(int(abs(angle_to_rotate))):
        sensor_state = navigateAndSee(rotation_action)

    #############################################################
    # Calculate the forward movement
    move_action = "move_forward"
    distance_to_travel = math.sqrt((destination[0] - current_position[0])**2 + (destination[1] - current_position[1])**2)
    steps_to_move = int(distance_to_travel / 0.01)

    for _ in range(steps_to_move):
        sensor_state = navigateAndSee(move_action)

    # Return the new coordinates of the position
    x_coordinate = sensor_state.position[0]
    z_coordinate = sensor_state.position[2]

    return (z_coordinate, x_coordinate)

pre_node = []
start = path[0]
goal = {"rack":(0, 255, 133),
        "cushion":(255, 9, 92),
        "sofa":(10, 0, 255),
        "stair":(173,255,0),
        "cooktop":(7, 255, 224)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Below are the params:')
    parser.add_argument(
        '-f', 
        type=str, 
        default="rack", 
        metavar='END', 
        action='store', 
        dest='End',
        help='rack, cushion, sofa, stair, cooktop'
    )
    args = parser.parse_args()

    # Check if the specified goal exists in the goal dictionary
    if args.End not in goal:
        print(f"Error: The target '{args.End}' is not in the goal list. Please choose a valid target.")
        exit(1)

    # Get the RGB value for the chosen target and set the corresponding end position
    end_rgb = goal[args.End]
    r, g, b = end_rgb

    print(f"Navigating to the target: {args.End} with RGB value: ({r}, {g}, {b})")

    # Initialize video writer
    if not os.path.exists("video/"):
        os.makedirs("video/")
    
    video_path = f"video/{args.End}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(video_path, fourcc, 100, (512, 512))

    # Load the path and navigate
    for i in range(len(path) - 1):
        temp = start
        start = driver(pre_node, start, path[i + 1])
        pre_node = temp

    # Release the video writer after navigation
    videowriter.release()
    print('finish')

