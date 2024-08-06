import time
import sys
import torch
import numpy as np
from collections import namedtuple
from omni.isaac.cortex.sample_behaviors.ur10.bin_packing_agent import BinPackingAgent
import gymnasium as gym
from gymnasium.envs.registration import register
from skrl.envs.wrappers.torch import wrap_env
import cv2 as cv
import math

import omni.isaac.cortex.math_util as math_util
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.math import normalized
from omni.isaac.cortex.cortex_world import CortexWorld
from omni.isaac.cortex.df import (
    DfDecider,
    DfDecision,
    DfNetwork,
    DfSetLockState,
    DfState,
    DfStateMachineDecider,
    DfStateSequence,
    DfTimedDeciderState,
    DfWaitState,
    DfWriteContextState,
)
from omni.isaac.cortex.dfb import DfDiagnosticsMonitor, DfLift, make_go_home
from omni.isaac.cortex.motion_commander import ApproachParams, MotionCommand, PosePq
from omni.isaac.cortex.obstacle_monitor_context import ObstacleMonitor, ObstacleMonitorContext

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)


def angle_to_quaternion(angle):
    qw = math.cos(angle / 2.0)
    qx = 0
    qy = 0
    qz = math.sin(angle / 2.0)
    return (qz, qx, qy, qw)


def detect_item_on_conveyor(image):
    """Detects if an item is on the conveyor belt."""
    if image is None:
        return False
    image = image[250:300, 450:850]
    image[np.isinf(image)] = 255
    return np.any(image < 1.489)


def calculate_item_center_in_camera_frame(image):
    image[np.isinf(image)] = 2
    image[image > 1.489] = 2
    image_crop = image[250:700, 450:850]
    value_image = np.full_like(image, 2)
    value_image[250:700, 450:850] = image_crop

    # remove all but item and conveyorbelt
    image = value_image

    normalized_depth = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    depth_uint8 = normalized_depth.astype(np.uint8)
    blurred = cv.GaussianBlur(depth_uint8, (5, 5), 0)

    edges = cv.Canny(blurred, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    epsilon = 0.04 * cv.arcLength(contours[0], True)
    approx = cv.approxPolyDP(contours[0], epsilon, True)

    M = cv.moments(approx)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        angle = 0.5 * np.arctan2(2 * M["mu11"], (M["mu20"] - M["mu02"]))
        angle_degrees = np.degrees(angle)
    else:
        cx, cy = 0, 0

    centroid = (cx, cy, value_image[cy][cx])

    angle = np.deg2rad(angle_degrees)

    quaternion = angle_to_quaternion(angle)
    return (centroid, quaternion)


def get_action_coords(action):
    action_to_placement_map = {}
    rotate = False
    if action >= 1024:
        rotate = True
        action -= 1024

    ignore_rows = [1, 2, 3, 4] # Rows to be ignored (0-indexed)
    ignore_columns = [1, 2, 3, 4] # Columns to be ignored (0-indexed)
    action_counter = 0

    for row in range(36):
        for col in range(36):
            if row in ignore_rows or col in ignore_columns:
                continue
            action_to_placement_map[action_counter] = (col, row)
            action_counter += 1
    
    return action_to_placement_map[action], rotate  # get location in grid from action


def make_array(items, bin_matrix=None):
    result = []
    for item in items:
        # Assuming scale is a tuple like (length, width, height)
        result.extend(tuple([x * 100 for x in item.scale]))
    # Append 1600 zeros at the end
    result.extend([0] * 3)  # current item
    if bin_matrix is not None:
        result.extend(np.reshape(bin_matrix, (1600,)))
    else:
        result.extend([0] * 1600)  # bin
    return result


def calculate_item_center(item_size, target_position, bin_matrix):
    """
    Calculate the center position of an item in world coordinates.

    :param item_size: A tuple (width, length, height) of the item in cm.
    :param target_position: The grid coordinates (x, y) where the top-left corner of the item should be.
    :return: The world coordinates (x, y, z) for the center of the item.
    """
    # Extracting the dimensions of the item
    width, length, height = item_size
    # Calculating the center position
    # For the x-coordinate, add half of the item's width to the target x-position
    center_x = 20 - (target_position[0] + ((width*100) / 2))

    # For the y-coordinate, add half of the item's length to the target y-position
    center_y = 20 - (target_position[1] + ((length*100) / 2))

    # The z-coordinate is half of the item's height, assuming the item sits on the ground (z=0)
    max_height = 0
    for x in range(int(target_position[0]), int(target_position[0]) + int(width * 100)):
        for y in range(int(target_position[1]), int(target_position[1]) + int(length * 100)):
            if bin_matrix[x][y] > max_height:
                max_height = bin_matrix[x][y]
    center_z = max_height + ((height*100) / 2)
    return (center_x, center_y, center_z)


def calculate_world_center(item_size, target_position, packing_square_center, bin_matrix):
    """
    Calculate the world coordinates for the center of an item, given its size, the target position in the grid,
    and the center of the packing square in world coordinates.

    :param item_size: A tuple (width, length, height) of the item in cm.
    :param target_position: The grid coordinates (x, y) where the top-left corner of the item should be.
    :param packing_square_center: The world coordinates (x, y, z) of the center of the packing square.
    :return: The world coordinates (x, y, z) for the center of the item.
    """
    # Calculating the center position in the grid
    grid_center_x, grid_center_y, grid_center_z = calculate_item_center(item_size, target_position, bin_matrix)
    if grid_center_z > 20:
        return None
    # Translate grid coordinates to world coordinates
    # The packing square is 40x40, so its half dimensions are 20x20
    # Subtract 20 (half the size of the square) from the grid position to align it with the world coordinates
    world_center_x = packing_square_center[0] + (grid_center_x/100)
    world_center_y = packing_square_center[1] + (grid_center_y/100)
    world_center_z = packing_square_center[2] + (grid_center_z/100)  # Assuming the z-coordinate remains the same

    print("Item center in grid coordinates: ({}, {}, {})".format(grid_center_x, grid_center_y, grid_center_z))
    print("Item center in world coordinates: ({}, {}, {})".format(world_center_x, world_center_y, world_center_z))
    return (world_center_x, world_center_y, world_center_z)


class ItemPacker:
    def __init__(self, item_set, random_item_set, bpa, bin_size=(40, 40), packing_center=np.array([0.8, 0, -0.7])):
        self.item_set = item_set
        self.item_dimension_list = self.make_item_dimension_list()
        self.random_item_set = random_item_set
        self.bin_size = bin_size
        self.packing_center = packing_center
        self.bin_matrix = np.zeros(bin_size)
        self.item_place_coordinates = []
        self.agent_tensor = self.create_agent_tensor().cuda()
        self.bpa = bpa
        self.volume_used = 0

    def make_item_dimension_list(self):
        # Function to make a list of item dimensions from the item set
        item_dimension_list = []
        for item in self.item_set:
            item_dimension_list.append(item.scale[0])
            item_dimension_list.append(item.scale[1])
            item_dimension_list.append(item.scale[2])
        return item_dimension_list

    def create_agent_tensor(self):
        temp_list = []
        temp_list.extend([0] * 3)  # extend space for current item
        temp_list.extend(self.item_dimension_list)
        # Extend 1600 zeros at the end to make bin
        temp_list.extend([0] * 1600)

        return torch.Tensor([temp_list])

    def pack_items(self):
        rotate = False
        for i in range(len(self.item_set)):
            item = self.random_item_set.popitem()
            item_index = item[1]
            item_dim = item[0].scale
            print("Item dimensions:")
            print(item_dim)
            self.update_agent_tensor_current_item(item_dim)

            action = self.bpa.get_action(self.agent_tensor).item()

            coords, rotate = get_action_coords(action)
            print("Coordinates:")
            print(coords)
            if rotate:
                item_dim = (item_dim[1], item_dim[0], item_dim[2])
            world_coords = calculate_world_center(item_dim, coords, self.packing_center, self.bin_matrix)
            if world_coords is None:
                print("Volume used: {}".format((self.volume_used/(0.4*0.4*0.20)*100)))
                print("Out of bounds reached maximum height")
                break
            self.volume_used += (item_dim[0] * item_dim[1] * item_dim[2])
            if rotate:
                world_coords += (1,)
            else:
                world_coords += (0,)
            self.item_place_coordinates.append(world_coords)
            self.update_bin_matrix(coords, item_dim)
            self.update_item_set(item_index)
            self.update_agent_tensor()
            rotate = not rotate

    def update_agent_tensor_current_item(self, item_dim):
        # Updating the agent tensor with new item dimensions
        start_index = 0
        self.agent_tensor[0][start_index] = item_dim[0] * 100
        self.agent_tensor[0][start_index + 1] = item_dim[1] * 100
        self.agent_tensor[0][start_index + 2] = item_dim[2] * 100

    def update_item_set(self, item_index):
        # Updating the item set with zeros as the item is not being used anymore
        self.item_dimension_list[(item_index*3)] = 0
        self.item_dimension_list[(item_index*3) + 1] = 0
        self.item_dimension_list[(item_index*3) + 2] = 0

    def update_agent_tensor(self):
        temp_list = []
        temp_list.extend([0] * 3)  # extend space for current item
        temp_list.extend(self.item_dimension_list)
        # Extend 1600 zeros at the end to make bin
        temp_list.extend(np.reshape(self.bin_matrix, (1600,)))

        self.agent_tensor = torch.Tensor([temp_list]).cuda()

    def get_placement_coords(self):
        # Function to get coordinates for placing the item
        return self.item_place_coordinates

    def update_bin_matrix(self, coords, item_dim):
        sub_area = self.bin_matrix[int(coords[0]):(int(coords[0]) + int(item_dim[0] * 100)), coords[1]:(int(coords[1]) + int(item_dim[1] * 100))]
        max_value = np.max(sub_area)
        new_height = max_value + (item_dim[2] * 100)
        # Updating the bin matrix to reflect the placement of the new item
        for x in range(int(coords[0]), int(coords[0]) + int(item_dim[0] * 100)):
            for y in range(int(coords[1]), int(coords[1]) + int(item_dim[1] * 100)):
                self.bin_matrix[x][y] = new_height
        print("Bin matrix:")
        for row in self.bin_matrix:
            print(row)
        #print(self.bin_matrix)


class ItemState:
    def __init__(self, item_obj):
        self.item_obj = item_obj
        self.grasp_T = None
        self.is_grasp_reached = None
        self.is_attached = None


class NavigationObstacleMonitor(ObstacleMonitor):
    def __init__(self, context):
        obstacles = [
            context.world.scene.get_object("navigation_dome_obs"),
            context.world.scene.get_object("navigation_barrier_obs"),
        ]
        super().__init__(obstacles)
        self.context = context

    def is_obstacle_required(self):
        target_p, _ = self.context.robot.arm.target_prim.get_world_pose()

        ref_p = np.array([0.6, 0.37, -0.99])
        eff_p = self.context.robot.arm.get_fk_p()

        ref_p[2] = 0.0
        eff_p[2] = 0.0
        target_p[2] = 0.0

        s_target = np.sign(np.cross(target_p, ref_p)[2])
        s_eff = np.sign(np.cross(eff_p, ref_p)[2])
        is_required = s_target * s_eff < 0.0
        return is_required


class ItemPackingDiagnostic:
    def __init__(self, item_name=None, grasp=None, grasp_reached=None, attached=None):
        self.item_name = item_name
        self.grasp = grasp
        self.grasp_reached = grasp_reached
        self.attached = attached


class ItemStackingDiagnosticsMonitor(DfDiagnosticsMonitor):
    def __init__(self, print_dt=1.0, diagnostic_fn=None):
        super().__init__(print_dt=print_dt)
        self.diagnostic_fn = diagnostic_fn

    def print_diagnostics(self, context):
        if context.has_active_item:
            diagnostic = ItemPackingDiagnostic(
                context.active_item.item_obj.name,
                context.active_item.grasp_T,
                context.active_item.is_grasp_reached,
                context.active_item.is_attached,
            )
        else:
            diagnostic = ItemPackingDiagnostic()
        if self.diagnostic_fn:
            self.diagnostic_fn(diagnostic)


class ItemPackingContext(ObstacleMonitorContext):
    def __init__(self, robot, camera, monitor_fn=None, item_set=None, random_item_set=None):
        super().__init__(robot.arm)
        self.robot = robot
        self.camera = camera
        self.item_set = item_set
        self.random_item_set = random_item_set
        self.world = CortexWorld.instance()
        self.diagnostics_monitor = ItemStackingDiagnosticsMonitor(print_dt=1.0, diagnostic_fn=monitor_fn)

        self.navigation_obs_monitor = NavigationObstacleMonitor(self)

        register(
            id="PackingEnvTest-v0",
            entry_point="modules:PackingEnvTest",
            max_episode_steps=240
        )
        env = gym.make('PackingEnvTest-v0')
        env = wrap_env(env)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        Configuration = namedtuple('Configuration', ['learning_epochs', 'mini_batches', 'discount_factor'])
        config = Configuration(learning_epochs=10, mini_batches=100, discount_factor=0.99)
        bpa = BinPackingAgent(env, device, config, "model_reward", "21_11")

        bpa.load_agent("/agent/agent_2750000.pt")

        packer = ItemPacker(self.item_set.copy(), self.random_item_set.copy(), bpa)
        packer.pack_items()
        self.item_place_coordinates = packer.get_placement_coords()

        self.add_obstacle_monitors([self.navigation_obs_monitor])

        print("Item place coordinates:")
        print(self.item_place_coordinates)
        print("Number of items to be packed: {}".format(len(self.item_place_coordinates)))

        self.items = []
        self.active_item = None
        self.packed_items = []
        self.add_monitors(
            [
                ItemPackingContext.monitor_items,
                ItemPackingContext.monitor_active_item,
                ItemPackingContext.monitor_active_item_grasp_T,
                ItemPackingContext.monitor_active_item_grasp_reached,
                self.diagnostics_monitor.monitor,
            ]
        )

    def reset(self):
        super().reset()

        self.items.clear()
        self.active_item = None
        self.packed_items.clear()

    @property
    def stack_complete(self):
        return len(self.packed_items) == len(self.item_place_coordinates)

    @property
    def elapse_time(self):
        return time.time() - self.start_time

    @property
    def has_active_item(self):
        return self.active_item is not None

    def monitor_items(self):
        if self.active_item is None:
            # Check whether there's a new item in the world.
            item_obj = RigidPrim("/World/Ur10Table/items/item_{}".format(len(self.items)), name="item_{}".format(len(self.items)))

            if item_obj is not None:
                p, w = item_obj.get_world_pose()
                x, y, z = p
                if y <= 0.9:
                    self.items.append(ItemState(item_obj))
                    self.active_item = self.items[-1]

    def monitor_active_item(self):
        if self.active_item is not None:
            p, _ = self.active_item.item_obj.get_world_pose()
            if p[2] < -2.0:
                self.active_item = None

    def monitor_active_item_grasp_T(self):
        if self.active_item is not None:
            item_T = math_util.pq2T(*self.active_item.item_obj.get_world_pose())
            item_R, item_p = math_util.unpack_T(item_T)
            print(item_R)
            item_ax, item_ay, item_az = math_util.unpack_R(item_R)
            print(item_ax)
            target_ax = -item_az

            margin = -0.02

            if item_ax[1] < 0.0:
                # x axis is pointing toward the robot
                target_ay = -item_ax
            else:
                target_ay = item_ax
            target_az = np.cross(target_ax, target_ay)
            target_p = item_p - margin * item_az

            target_T = math_util.pack_Rp(math_util.pack_R(target_ax, target_ay, target_az), target_p)
            self.active_item.grasp_T = target_T

    def monitor_active_item_grasp_reached(self):
        if self.has_active_item:
            fk_T = self.robot.arm.get_fk_T()
            self.active_item.is_grasp_reached = math_util.transforms_are_close(
                self.active_item.grasp_T, fk_T, p_thresh=0.005, R_thresh=0.01
            )
            # We can be looser with this proximity check.
            self.active_item.is_attached = (
                math_util.transforms_are_close(self.active_item.grasp_T, fk_T, p_thresh=0.1, R_thresh=1.0)
                and self.robot.suction_gripper.is_closed()
            )

    def mark_active_item_as_complete(self):
        self.packed_items.append(self.active_item)
        self.active_item = None


class Move(DfState):
    def __init__(self, p_thresh, R_thresh):
        self.p_thresh = p_thresh
        self.R_thresh = R_thresh
        self.command = None

    def update_command(self, command):
        self.command = command

    def step(self):
        self.context.robot.arm.send(self.command)

        fk_T = self.context.robot.arm.get_fk_T()
        if math_util.transforms_are_close(
            self.command.target_pose.to_T(), fk_T, p_thresh=self.p_thresh, R_thresh=self.R_thresh
        ):
            return None
        return self


class MoveWithNavObs(Move):
    def enter(self):
        super().enter()
        self.context.navigation_obs_monitor.activate_autotoggle()

    def exit(self):
        super().exit()
        self.context.navigation_obs_monitor.deactivate_autotoggle()


class ReachToPick(MoveWithNavObs):
    """Reach to pick the item. The item can be anywhere.
    """

    def __init__(self):
        super().__init__(p_thresh=0.001, R_thresh=2.0)

    def enter(self):
        super().enter()

    def step(self):
        R, p = math_util.unpack_T(self.context.active_item.grasp_T)
        ax, ay, az = math_util.unpack_R(R)

        posture_config = np.array([-1.2654234, -2.9708025, -2.219733, 0.6445836, 1.5186214, 0.30098662])

        approach_length = 0.4

        self.update_command(
            MotionCommand(
                target_pose=PosePq(p, math_util.matrix_to_quat(R)),
                approach_params=ApproachParams(direction=approach_length * ax, std_dev=0.005),
                posture_config=posture_config,
            )
        )

        return super().step()

    def exit(self):
        super().exit()


class ReachToPlace(MoveWithNavObs):
    def __init__(self):
        super().__init__(p_thresh=0.005, R_thresh=2.0)

    def enter(self):
        super().enter()
        # Here we decide where to place the item
        p = self.context.item_place_coordinates[len(self.context.packed_items)]
        self.target_p = (p[0], p[1], p[2])

        target_ax = np.array([0.0, 0.0, -1.0])
        target_az = np.array([0.0, -1.0, 0.0])
        target_ay = np.cross(target_az, target_ax)

        R_z_90 = np.array([[0, -1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])

        # Rotated axes
        if p[3] == 1:
            target_ax = np.dot(R_z_90, target_ax)
            target_ay = np.dot(R_z_90, target_ay)
            target_az = np.dot(R_z_90, target_az)

        self.target_R = math_util.pack_R(target_ax, target_ay, target_az)

    def step(self):
        target_pose = PosePq(self.target_p, math_util.matrix_to_quat(self.target_R))

        approach_params = ApproachParams(direction=0.2 * np.array([0.0, 0.0, -1.0]), std_dev=0.005)
        posture_config = self.context.robot.default_config
        self.update_command(
            MotionCommand(target_pose=target_pose, approach_params=approach_params, posture_config=posture_config)
        )

        return super().step()


class CloseSuctionGripperWithRetries(DfState):
    def enter(self):
        pass

    def step(self):
        gripper = self.context.robot.suction_gripper
        gripper.close()
        if gripper.is_closed():
            return None
        return self


class CloseSuctionGripper(DfState):
    def enter(self):
        print("<close gripper>")
        self.context.robot.suction_gripper.close()

    def step(self):
        return None


class OpenSuctionGripper(DfState):
    def enter(self):
        print("<open gripper>")
        self.context.robot.suction_gripper.open()

    def step(self):
        return None


class DoNothing(DfState):
    def enter(self):
        self.context.robot.arm.clear()

    def step(self):
        print(self.context.robot.arm.target_prim.get_world_pose())
        return self


class PickItem(DfStateMachineDecider):
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPick(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    CloseSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.3), activity_duration=0.4),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )


class PlaceItem(DfStateMachineDecider):
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPlace(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    OpenSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.1), activity_duration=0.25),
                    DfWriteContextState(lambda ctx: ctx.mark_active_item_as_complete()),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )


class Dispatch(DfDecider):
    def __init__(self):
        super().__init__()

        self.add_child("pick_item", PickItem())
        self.add_child("place_item", PlaceItem())
        self.add_child("go_home", make_go_home())
        self.add_child("do_nothing", DfStateMachineDecider(DoNothing()))

    def decide(self):
        if self.context.stack_complete:
            return DfDecision("go_home")

        if self.context.has_active_item:
            if not self.context.active_item.is_attached:
                return DfDecision("pick_item")
            else:
                return DfDecision("place_item")
        else:
            return DfDecision("go_home")


def make_decider_network(robot, camera, monitor_fn, item_set, random_item_set):
    return DfNetwork(Dispatch(), context=ItemPackingContext(robot, camera, monitor_fn, item_set, random_item_set))
