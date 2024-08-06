import numpy as np
from robot_commands import UR5e
from computer_vision import D435
from agent_code import ItemPacker
import time


def transform_box2rob(centroid):
    cam2rob_tran = np.array([[-0.99972311, -0.02275655, 0.00598612, 0.23978807],
                             [-0.02279908, 0.99971461, -0.00713504, -0.51893375],
                             [-0.00582205, -0.00726954, -0.99995663, 0.91756092],
                             [0.0, 0.0, 0.0, 1.0]])

    return np.matmul(cam2rob_tran, np.transpose([centroid[0], centroid[1], centroid[2], 1.0]))[0:3]


def main():
    # Initialize robot and camera
    ur5e = UR5e()
    d435 = D435()

    reference_point = [0, -0.500, 0, np.pi, 0, 0]

    agent = ItemPacker()

    # Move to home pose
    print(ur5e.move_to_home_pose())

    input("Press Enter to continue...")

    # Find object pose
    while True:
        box_size, centroid, angle_degrees = d435.find_object()
        real_height = box_size[0][2]
        length = box_size[1][0]
        width = box_size[1][1]
        height = box_size[1][2]

        ur5e.move_above_pickup_pose()
        time.sleep(1)

        item_pick_position = transform_box2rob(centroid)
        rotate = False
        if angle_degrees > 45:
            angle_degrees = angle_degrees - 90
            rotate = True
        elif angle_degrees < -45:
            angle_degrees = angle_degrees + 90
            rotate = True
        # Object pose in relation to reference point
        offset = [item_pick_position[0] - reference_point[0], (reference_point[1] - item_pick_position[1]), -real_height, 0, 0, np.deg2rad(angle_degrees)]

        # calculate transformation
        target_pickup_pose = ur5e.rtde_c.poseTrans(reference_point, offset)

        # Move to object pose
        ur5e.move_to_pickup_pose(target_pickup_pose)
        time.sleep(1)
        ur5e.pickup_object()
        time.sleep(1)
        
        print(target_pickup_pose)

        is_packing_done, place_coords = agent.pack_item((width, length, height))
        if is_packing_done:
            print("Packing is done")
            break
        place_pose = None
        if place_coords[3] == 0 and not rotate:
            place_pose = [place_coords[0], place_coords[1], place_coords[2], 2.219769314908943, 2.2230234734137575, 0]
        elif place_coords[3] == 1 and not rotate:
            place_pose = [place_coords[0], place_coords[1], place_coords[2], 0, np.pi, 0]
        elif place_coords[3] == 1 and rotate:
            place_pose = [place_coords[0], place_coords[1], place_coords[2], 2.219769314908943, 2.2230234734137575, 0]
        elif place_coords[3] == 0 and rotate:
            place_pose = [place_coords[0], place_coords[1], place_coords[2], 0, np.pi, 0]
        else:
            print("Something went wrong")
            print(place_coords)
            break
        print(place_coords)
        ur5e.change_place_waypoint(place_pose)
        ur5e.move_to_home_pose()
        time.sleep(1)
        ur5e.move_above_place_pose()
        time.sleep(1)
        # Move to place pose
        ur5e.move_to_place_pose(place_pose)
        time.sleep(1)
        ur5e.place_object()
        time.sleep(1)

        # Move above place pose
        ur5e.move_above_place_pose()
        time.sleep(1)

        # Move to home pose
        ur5e.move_to_home_pose()
        time.sleep(1)
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()
