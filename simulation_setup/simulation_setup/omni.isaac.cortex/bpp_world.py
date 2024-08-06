from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import random
import copy
import numpy as np
import omni.isaac.cortex.math_util as math_util
import omni.isaac.cortex.sample_behaviors.ur10.bpp_behavior as behavior
from omni.isaac.core.objects import VisualCapsule, VisualSphere
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.cortex.cortex_utils import get_assets_root_path_or_die
from omni.isaac.cortex.cortex_world import CortexWorld
from omni.isaac.cortex.robot import CortexUr10
from omni.isaac.core.objects.cuboid import FixedCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.materials.physics_material import PhysicsMaterial

class Item:
    def __init__(self, scale, orientation, position, color):
        self.scale = scale
        self.orientation = orientation
        self.position = position
        self.color = color

    def set_scale(self, scale):
        self.scale = scale


class ItemSet:
    def __init__(self, num_items):
        self.item_choices = [
            (17, 9, 3),
            (17, 6, 4),
            (17, 5, 3),
            (17, 12, 4),
            (17, 8, 4),
            (16, 14, 4)
        ]
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255)
        ]
        self.num_items = num_items
        self.items = []

    def generate_items(self):
        for i in range(self.num_items):
            choice = random.choice(self.item_choices)
            color_choice = random.choice(self.colors)
            scale = np.array([choice[0] / 100, choice[1] / 100, choice[2] / 100])
            color = np.array([color_choice[0], color_choice[1], color_choice[2]])
            x = random.uniform(-0.15, 0.15)
            y = 1.5
            z = -0.4
            position = np.array([x, y, z])

            z = random.random() * 0.02 - 0.01
            w = random.random() * 0.02 - 0.01
            norm = np.sqrt(z**2 + w**2)
            quat = math_util.Quaternion([w / norm, 0, 0, z / norm])
            orientation = quat.vals

            self.items.append(Item(scale, orientation, position, color))
        return self.items


def is_item_on_conveyor(image):
    """Detects if an item is on the conveyor belt."""
    image = image[250:700, 450:850]
    image[np.isinf(image)] = 255
    return np.any(image < 1.47)

def is_item_close_to_UR(image):
    """Detects if an item is on the conveyor belt."""
    image = image[250:300, 450:850]
    image[np.isinf(image)] = 255
    return np.any(image < 1.47)

class Ur10Assets:
    def __init__(self):
        self.assets_root_path = get_assets_root_path_or_die()
        self.ur10_table_usd = (
            "/usd_files/bpp_world3.usd"
        )
        self.item_usd = self.assets_root_path + "/Isaac/Props/Blocks/block_instancable.usd"
        self.glass = self.assets_root_path + "/NVIDIA/Materials/2023_2_1/Base/Glass/Tinted_Glass_R75.mdl"


def print_diagnostics(diagnostic):
    print("=========== logical state ==========")
    if diagnostic.item_name:
        print("active bin info:")
        print("- bin_obj.name: {}".format(diagnostic.item_name))
        print("- grasp_T:\n{}".format(diagnostic.grasp))
        print("- is_grasp_reached: {}".format(diagnostic.grasp_reached))
        print("- is_attached:  {}".format(diagnostic.attached))
    else:
        print("<no active bin>")

    print("------------------------------------")


class BinStackingTask(BaseTask):
    def __init__(self, env_path, assets, camera, conveyor_belt, item_set):
        super().__init__("bin_stacking")
        self.assets = assets  # assets in environment
        self.camera = camera  # camera in environment
        self.conveyor_belt = conveyor_belt  # conveyor belt in environment
        self.env_path = env_path  # path to environment
        self.on_conveyor = None
        self.i = 0
        self.item_set = item_set  # generate items
        self.items = []
        self.item_detected = False

    def post_reset(self) -> None:
        # initialize camera setting specifically to detect depth
        if len(self.items) > 0:
            for item in self.items:
                self.scene.remove_object(item.name)
            self.items.clear()
        self.i = 0
        self.on_conveyor = None

    def pre_step(self, time_step_index, simulation_time) -> None:
        """Spawn a new randomly oriented bin if the previous bin has been placed."""
        spawn_new = False
        if self.on_conveyor is None:
            spawn_new = True
        else:
            # capture every 5th frame
            if self.i == 5:
                depth_capture = self.camera.get_current_frame()["distance_to_camera"]
                if is_item_close_to_UR(depth_capture):
                    self.conveyor_belt.set_linear_velocity(np.array([0, 0, 0]))
                    self.item_detected = True
                elif not is_item_on_conveyor(depth_capture) and self.item_detected:
                    spawn_new = True
                    self.item_detected = False
                self.i = 0
        if spawn_new:
            self.conveyor_belt.set_linear_velocity(np.array([0, -0.80, 0]))
            name = "item_{}".format(len(self.items))
            prim_path = self.env_path + "/items/{}".format(name)
            item = self.item_set.popitem()
            position = item[0].position
            scale = item[0].scale
            scale[0] -= 0.01
            scale[1] -= 0.01
            orientation = item[0].orientation
            self.on_conveyor = DynamicCuboid(name=name,
                                           prim_path=prim_path,
                                           position=position,
                                           scale=scale,
                                           orientation=orientation,
                                           size=1.0,
                                           color=item[0].color)
            self.items.append(self.on_conveyor)
        if self.i <= 10:
            self.i += 1


def main():
    world = CortexWorld()

    env_path = "/World/Ur10Table"
    ur10_assets = Ur10Assets()
    add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)
    robot = world.add_robot(CortexUr10(name="robot", prim_path="{}/ur10".format(env_path)))

    conveyor_belt = RigidPrim(
        "/World/Ur10Table/conveyor/Belt",
        linear_velocity=np.array([0, -0.8, 0]),
    )
    conveyor_belt.set_linear_velocity(np.array([0, -0.80, 0]))
    
    camera = Camera(
        "/World/Ur10Table/camera_mount/rsd455/RSD455/Camera_Pseudo_Depth",
        frequency=30,
        resolution=(1280, 720),
    )

    obs = world.scene.add(
        VisualSphere(
            "/World/Ur10Table/Obstacles/NavigationDome",
            name="navigation_dome_obs",
            position=[-0.031, -0.018, -1.086],
            radius=1.1,
            visible=False,
        )
    )
    robot.register_obstacle(obs)

    az = np.array([1.0, 0.0, -0.3])
    ax = np.array([0.0, 1.0, 0.0])
    ay = np.cross(az, ax)
    R = math_util.pack_R(ax, ay, az)
    quat = math_util.matrix_to_quat(R)
    obs = world.scene.add(
        VisualCapsule(
            "/World/Ur10Table/Obstacles/NavigationBarrier",
            name="navigation_barrier_obs",
            position=[0.471, 0.276, -0.463 - 0.1],
            orientation=quat,
            radius=0.5,
            height=0.9,
            visible=False,
        )
    )
    robot.register_obstacle(obs)

    bin_bottom = world.scene.add(
        FixedCuboid(
            "/World/Ur10Table/Bin/BinBottom",
            name="bin_bottom",
            position=np.array([0.8, 0.0, -0.745]),
            scale=np.array([0.4, 0.4, 0.01]),
            color=np.array([205, 159, 97]),
        )
    )
    robot.register_obstacle(bin_bottom)

    bin_left = world.scene.add(
        FixedCuboid(
            "/World/Ur10Table/Bin/BinLeft",
            name="bin_left",
            position=np.array([0.8, 0.21, -0.645]),
            scale=np.array([0.41, 0.01, 0.2]),
            color=np.array([205, 159, 97]),
            physics_material=PhysicsMaterial("/World/Ur10Table/Looks/Tinted_Glass_R75"),
        )
    )

    bin_right = world.scene.add(
        FixedCuboid(
            "/World/Ur10Table/Bin/BinRight",
            name="bin_right",
            position=np.array([0.8, -0.21, -0.645]),
            scale=np.array([0.41, 0.01, 0.2]),
            color=np.array([205, 159, 97]),

        )
    )

    bin_back = world.scene.add(
        FixedCuboid(
            "/World/Ur10Table/Bin/BinBack",
            name="bin_back",
            position=np.array([0.59, 0.0, -0.645]),
            scale=np.array([0.01, 0.41, 0.2]),
            color=np.array([205, 159, 97]),
        )
    )

    bin_front = world.scene.add(
        FixedCuboid(
            "/World/Ur10Table/Bin/BinFront",
            name="bin_front",
            position=np.array([1.01, 0.0, -0.645]),
            scale=np.array([0.01, 0.41, 0.2]),
            color=np.array([205, 159, 97]),
        )
    )

    camera.initialize()
    camera.add_distance_to_camera_to_frame()

    item_set = ItemSet(110).generate_items()
    item_dict = {item: index for index, item in enumerate(item_set)}
    dict_items = list(item_dict.items())
    print(dict_items)
    random.shuffle(dict_items)
    random_items_dict = dict(dict_items)
    dict_1 = copy.deepcopy(random_items_dict)
    dict_2 = copy.deepcopy(random_items_dict)
    world.add_task(BinStackingTask(env_path, ur10_assets, camera, conveyor_belt, dict_1))
    world.add_decider_network(behavior.make_decider_network(robot, camera, print_diagnostics, item_set.copy(), dict_2))

    world.run(simulation_app)
    simulation_app.close()


if __name__ == "__main__":
    main()
