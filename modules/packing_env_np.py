import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
import random
import torch
import copy

class PackingEnvNp(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode = None, size = 40):
        self.size = size #size of the grid
        self.x = 40
        self.y = 40
        self.z = 40
        self.box_size = 17
        self.volume_used = 0
        self.illegal_actions = 0
        self.window_size = 800 #vizualizer resolution
        self.box_placed = False
        self.no_items = 110
        self.rotating = False
        self.ignore = 4
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(0, self.z - 1, shape=(self.x,self.y), dtype=int),
                "box": spaces.Box(0, self.box_size - 1, shape=(3,), dtype=int),
                "boxes": spaces.Box(0, self.box_size - 1, shape=(self.no_items, 3), dtype=int), #spaces.Tuple((spaces.Discrete(self.no_items) for _ in range(3)))
            }
        )
        self.action_space = spaces.Discrete((size-self.ignore-4) * (size-self.ignore-4) * 2) #action space is number of places in the grid
        self._action_to_placement_map = {} #mapping actions to lcoations on the grid
        self._actual_boxes = [
            (17, 9, 3),
            (17, 6, 4),
            (17, 5, 3),
            (17, 12, 4),
            (17, 8, 4),
            (16, 14, 4)
        ]

        self._boxes = [self._actual_boxes[random.randint(0, len(self._actual_boxes)-1)] for _ in range(self.no_items)]
        self.boxes_local = copy.copy(self._boxes)
        
        self.box_index = random.randint(0, len(self.boxes_local) - 1)
        self._box = self.boxes_local.pop(self.box_index)

         ####generate action dictionary ####
        ignore_rows = [1, 2, 3, 4]  # Rows to be ignored (0-indexed)
        ignore_columns = [1, 2, 3, 4]  # Columns to be ignored (0-indexed)
        action_counter = 0

        for row in range(self.x - self.ignore):
            for col in range(self.y - self.ignore):
                if row in ignore_rows or col in ignore_columns:
                    continue
                self._action_to_placement_map[action_counter] = (col, row)
                action_counter += 1
        
        # for number in range((self.x-self.ignore) * (self.y-self.ignore)):
        #     row = number // self.y
        #     col = number % self.x
        #     self._action_to_placement_map[number] = (col, row)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        if self.box_placed:
            self.box_placed = False

        box  = torch.tensor([self._box[0], self._box[1], self._box[2]], dtype=torch.int32)
        world = torch.tensor(self._observation, dtype=torch.int32)
        observation = world.numpy()
        boxes_tensor = torch.tensor(self._boxes, dtype=torch.int32)
        return {
            "observation": observation,
            "box": box.numpy(),
            "boxes": boxes_tensor.numpy(),
        }

    def _get_info(self):
        return {
        }

    def get_volume_used(self):
        bin_volume = self.x * self.y * self.z
        space_utilized = (self.volume_used / bin_volume) * 100
        return space_utilized

    #function for picking color in vizualizer
    def _get_color(self, case):
        color_dict = {
            0: (0, 0, 255),
            1: (8, 0, 247),
            2: (16, 0, 239),
            3: (24, 0, 231),
            4: (32, 0, 223),
            5: (40, 0, 215),
            6: (48, 0, 207),
            7: (56, 0, 199),
            8: (64, 0, 191),
            9: (72, 0, 183),
            10: (80, 0, 175),
            11: (88, 0, 167),
            12: (96, 0, 159),
            13: (104, 0, 151),
            14: (112, 0, 143),
            15: (120, 0, 135),
            16: (128, 0, 127),
            17: (136, 0, 119),
            18: (144, 0, 111),
            19: (152, 0, 103),
            20: (160, 0, 95),
            21: (168, 0, 87),
            22: (176, 0, 79),
            23: (184, 0, 71),
            24: (192, 0, 63),
            25: (200, 0, 55),
            26: (208, 0, 47),
            27: (216, 0, 39),
            28: (224, 0, 31),
            29: (232, 0, 23),
            30: (240, 0, 15),
            31: (248, 0, 7),
            32: (255, 0, 0),
            33: (255, 15, 0),
            34: (255, 31, 0),
            35: (255, 47, 0),
            36: (255, 63, 0),
            37: (255, 79, 0),
            38: (255, 95, 0),
            39: (255, 111, 0),
            40: (255, 180, 0)
        }
        
        return color_dict.get(case, (0,0,0))

    def _is_valid_action(self, x, y):
        return x-1 < self.x and y-1 < self.y

    # def _stability_check(self, package, bin_section):
    #     package_bottom_plane_area = package[0] * package[1] # compute package's bottom area
    #     highest_value = np.max(bin_section)
    #     supported_area = (bin_section == highest_value).sum() #find area supported by packages below
    #     supported_area_lower = (bin_section == (highest_value-1)).sum()
    #     supported_area = supported_area + supported_area_lower
    #     supported_area_percentage = supported_area / package_bottom_plane_area #compute supported percentage
        
    #     #check if corners are supported:
    #     top_left = bin_section[0][0] == highest_value
    #     bottom_left = (bin_section[(package[0]-1)][0]) == highest_value
    #     top_right = bin_section[0][package[1]-1] == highest_value
    #     bottom_right = bin_section[package[0]-1][package[1]-1] == highest_value
    #     supported_corners = top_left + bottom_left + top_right + bottom_right #count how many are supported
    
    #     #check for feasability rules:
    #     feasible = False
    #     if supported_area_percentage > 0.3 and supported_corners == 4:
    #         feasible = True
    #     elif supported_area_percentage > 0.5 and supported_corners > 3:
    #         feasible = True
    #     elif supported_area_percentage > 0.6:
    #         feasible = True
    #     return feasible

    def _support_check(self, package, bin_section):
        package_bottom_plane_area = package[0] * package[1] # compute package's bottom area
        highest_value = np.max(bin_section)
        supported_area = (bin_section == highest_value).sum() #find area supported by packages below
        supported_area_percentage = supported_area / package_bottom_plane_area #compute supported percentage
        return supported_area_percentage



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._observation = np.zeros((self.size, self.size))
        
        self._boxes = [self._actual_boxes[random.randint(0, len(self._actual_boxes)-1)] for _ in range(self.no_items)]
        self.boxes_local = copy.copy(self._boxes)
        self.box_index = random.randint(0, len(self.boxes_local) - 1)
        self._box = self.boxes_local.pop(self.box_index)
        
        observation = self._get_obs()
        info = self._get_info()
        self.volume_used = 0
        self.box_placed = False
        self.illegal_actions = 0
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        reward = 0
        terminated = False

        if action < 1024:
            placement = self._action_to_placement_map[action] #get location in grid from action
        else:
            self._box = (self._box[1], self._box[0], self._box[2])
            placement = self._action_to_placement_map[action-1024]

        end_point = (placement[0] + self._box[0], placement[1] + self._box[1]) #calculate end point for box <-> opposite corner
        
        if self._is_valid_action(end_point[0], end_point[1]):
            sub_area = self._observation[placement[0]:end_point[0], placement[1]:end_point[1]]
            max_value = np.max(sub_area) #highest point in the placement location
            new_value = max_value + self._box[2]
            scale = self._support_check(self._box, sub_area)
            if new_value <= self.z:
                self.box_placed = True
                self._boxes[self.box_index] = (0, 0, 0)
                
                if len(self.boxes_local) > 0:
                    self.box_index = random.randint(0, len(self.boxes_local) - 1)
                    self._box = self.boxes_local.pop(self.box_index)
                else:
                    terminated = True
                
                box_volume = self._box[0] * self._box[1] * self._box[2]
                self.volume_used = self.volume_used + box_volume

                reward = box_volume / 640

                top_left_corner = self._observation[placement[0], placement[1]] # value of bin where the top left corner of package is going to be placed
                    
                if placement[0] > 0:
                    left_along_x = self._observation[placement[0]-1, placement[1]]
                    if top_left_corner == left_along_x:
                        reward = reward/2

                if placement[1] > 0:
                    left_along_y = self._observation[placement[0], placement[1]-1]
                    if top_left_corner == left_along_y:
                        reward = reward/2

                for i in range(placement[0], end_point[0]): #update bin grid with the box
                    for j in range(placement[1], end_point[1]):
                        self._observation[i, j] =  new_value
            else:
                self.box_placed = False
                self.illegal_actions = self.illegal_actions + 1
                terminated = True
        else:
            self.box_placed = False
            self.illegal_actions = self.illegal_actions + 1
            terminated = True

        #terminated = np.any(self._world > 9) # termination criteria
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        info = {"lol": terminated}
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size+500, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
    
        canvas = pygame.Surface((self.window_size+500, self.window_size))
        canvas.fill((181, 148, 94))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        pygame.draw.rect(
                canvas,
                (0, 0, 255),
                pygame.Rect(
                    (int(0), int(0)),
                    (800, 800),
                ),
            )
    
        ##### Item to be packed #####
        color = (139,69,19)
        if self.rotating:
            color = (245, 66, 221)
            
        pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    (int(880), int(80)),
                    (self._box[0] * pix_square_size, self._box[1] * pix_square_size),
                ),
            )
        #########################
        
        # Draw boxes
        nonzero_indices = np.nonzero(self._observation)
        for index_tuple in zip(nonzero_indices[0], nonzero_indices[1]):
            color = self._get_color(self._observation[index_tuple[0], index_tuple[1]])
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    (int(pix_square_size) * index_tuple[0], int(pix_square_size) * index_tuple[1]),
                    (pix_square_size, pix_square_size),
                ),
            )
            font = pygame.font.Font(None, 36)  # You can adjust the font size
            text = font.render(str(int(self._observation[index_tuple[0], index_tuple[1]])), True, (255, 255, 255))  # Text color: white
        
            # Calculate the position to center the text within the rectangle
            text_rect = text.get_rect()
            text_rect.center = (int(pix_square_size) * index_tuple[0] + pix_square_size // 2, int(pix_square_size) * index_tuple[1] + pix_square_size // 2)
        
            # Draw the text on the canvas
            canvas.blit(text, text_rect)
    
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
    
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
    
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()