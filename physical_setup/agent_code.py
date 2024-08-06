import torch
import numpy as np
import gymnasium as gym
from collections import namedtuple
from gymnasium.envs.registration import register
from skrl.envs.wrappers.torch import wrap_env
from bin_packing_agent import BinPackingAgent
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_action_coords(action):
    action_to_placement_map = {}
    rotate = False
    if action >= 1024:
        rotate = True
        action -= 1024

    ignore_rows = [1, 2, 3, 4]  # Rows to be ignored (0-indexed)
    ignore_columns = [1, 2, 3, 4]  # Columns to be ignored (0-indexed)
    action_counter = 0

    for row in range(36):
        for col in range(36):
            if row in ignore_rows or col in ignore_columns:
                continue
            action_to_placement_map[action_counter] = (col, row)
            action_counter += 1

    return action_to_placement_map[action], rotate

def calculate_bin_matrix_place_coord(item_size, target_position, bin_matrix):
    # Extracting the dimensions of the item
    width, length, height = item_size
    # Calculating the center position
    # For the x-coordinate, add half of the item's width to the target x-position
    center_x = 20 - (target_position[1] + (width / 2))

    # For the y-coordinate, add half of the item's length to the target y-position
    center_y = 20 - (target_position[0] + (length / 2))

    # The z-coordinate is half of the item's height, assuming the item sits on the ground (z=0)
    sub_area = bin_matrix[int(target_position[0]):(int(target_position[0]) + int(length)), int(target_position[1]):(int(target_position[1]) + int(width))]
    max_value = np.max(sub_area)
    center_z = max_value + (height)

    return (center_x, center_y, center_z)


def calculate_world_center(item_size, target_position, packing_square_center, bin_matrix):
    # Calculating the center position in the grid
    grid_center_x, grid_center_y, grid_center_z = calculate_bin_matrix_place_coord(item_size, target_position, bin_matrix)
    if grid_center_z > 12:
        return None
    # Translate grid coordinates to world coordinates
    # The packing square is 40x40, so its half dimensions are 20x20
    # Subtract 20 (half the size of the square) from the grid position to align it with the world coordinates
    z_margin = 0.002
    world_center_x = packing_square_center[0] - (grid_center_x/100)
    world_center_y = packing_square_center[1] + (grid_center_y/100)
    world_center_z = packing_square_center[2] + (grid_center_z/100) + z_margin  # Assuming the z-coordinate remains the same

    print("Item center in grid coordinates: ({}, {}, {})".format(grid_center_x, grid_center_y, grid_center_z))
    print("Item center in world coordinates: ({}, {}, {})".format(world_center_x, world_center_y, world_center_z))
    return (world_center_x, world_center_y, world_center_z)

class ItemPacker:
    def __init__(self, bin_size=(40, 40), packing_center=np.array([0.2, 0.495, -0.0055])):
        self.item_dimension_list = [0] * 3 * 110
        self.bin_size = bin_size
        self.packing_center = packing_center
        self.bin_matrix = np.zeros(bin_size)
        self.agent_tensor = self.create_agent_tensor().cuda()
        self.bpa = self.setup_agent()
        self.volume_used = 0
    
    def setup_agent(self):
        register(
            id="PackingEnv-v0",
            entry_point="modules:PackingEnv",
            max_episode_steps=240
        )

        env = gym.make('PackingEnv-v0')
        env = wrap_env(env)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Configuration = namedtuple('Configuration', ['learning_epochs', 'mini_batches', 'discount_factor'])
        config = Configuration(learning_epochs=10, mini_batches=100, discount_factor=0.99)
        bpa = BinPackingAgent(env, device, config, "model_reward", "21_11")
        bpa.load_agent("/home/pmkz/good_agent/agent/agent_1500000.pt")

        return bpa

    def create_agent_tensor(self):
        temp_list = []
        temp_list.extend([0] * 3)  # extend space for current item
        temp_list.extend(self.item_dimension_list)
        # Extend 1600 zeros at the end to make bin
        temp_list.extend([0] * 1600)

        return torch.Tensor([temp_list])

    def pack_item(self, item):
        rotate = False
        item_dim = (item[0] * 100, item[1] * 100, item[2] * 100)
        print("Item dimensions:")
        print(item_dim)
        self.update_agent_tensor_current_item(item_dim)
        print("Agent tensor:")
        print(self.agent_tensor)
        action = self.bpa.get_action(self.agent_tensor).item()
        coords, rotate = get_action_coords(action)
        print("Coordinates:")
        print(coords)
        if rotate:
            print("Rotate")
            item_dim = (item_dim[1], item_dim[0], item_dim[2])
        world_coords = calculate_world_center(item_dim, coords, self.packing_center, self.bin_matrix)
        if world_coords is None:
            print("Volume used: {}".format((self.volume_used/(40*40*20)*100)))
            print("Out of bounds reached maximum height")
            return True, None
        self.volume_used += (item_dim[0] * item_dim[1] * item_dim[2])
        if rotate:
            world_coords += (1,)
        else:
            world_coords += (0,)

        self.update_bin_matrix(coords, item_dim)
        self.update_agent_tensor()


        return False, world_coords

    def update_agent_tensor_current_item(self, item_dim):
        # Updating the agent tensor with new item dimensions
        start_index = 0
        self.agent_tensor[0][start_index] = item_dim[1]
        self.agent_tensor[0][start_index + 1] = item_dim[0]
        self.agent_tensor[0][start_index + 2] = item_dim[2]

    def update_agent_tensor(self):
        temp_list = []
        temp_list.extend([0] * 3)  # extend space for current item
        temp_list.extend([0] * 3 * 110)
        # Extend 1600 zeros at the end to make bin
        temp_list.extend(np.reshape(self.bin_matrix, (1600,)))

        self.agent_tensor = torch.Tensor([temp_list]).cuda()

    def update_bin_matrix(self, coords, item_dim):
        sub_area = self.bin_matrix[int(coords[0]):(int(coords[0]) + int(item_dim[1])), coords[1]:(int(coords[1]) + int(item_dim[0]))]
        print("Sub area:")
        print(sub_area)
        max_value = np.max(sub_area)
        new_height = max_value + (item_dim[2])
        print(coords)
        # Updating the bin matrix to reflect the placement of the new item
        self.bin_matrix[int(coords[0]):(int(coords[0]) + int(item_dim[1])), coords[1]:(int(coords[1]) + int(item_dim[0]))] = new_height
        print("Bin matrix:")
        print(self.bin_matrix)
