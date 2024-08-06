import rtde_control
import rtde_io
import rtde_receive
import numpy as np

class UR5e:
    def __init__(self):
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.10.21")
        self.rtde_io = rtde_io.RTDEIOInterface("192.168.10.21")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.10.21")
        self.actual_tcp = self.rtde_r.getActualTCPPose()
        self.home_waypoint = [0.4915861375204079, -0.04178948258624242, 0.3526986582795037, -1.378671134664996, -2.8228498922916674, 8.975961826819285e-05]
        self.place_waypoint = [0.1290979396127544, 0.47617680224044423, 0.35271077000432904, -0.7501739114990704, 3.0506896753001675, -8.17608073781823e-05]
        self.pickup_waypoint = [0.17286390838440968, -0.4620828742336828, 0.35267570768999496, -2.6740952616723566, -1.6486374576302854, 3.648289997432579e-05]
        self.suction_io = 0

    def turn_on_suction(self):
        self.rtde_io.setStandardDigitalOut(self.suction_io, True)

    def turn_off_suction(self):
        self.rtde_io.setStandardDigitalOut(self.suction_io, False)

    def move_above_pickup_pose(self):
        return self.rtde_c.moveL(self.pickup_waypoint)

    def move_above_place_pose(self):
        return self.rtde_c.moveL(self.place_waypoint)

    def change_place_waypoint(self, pose):
        self.place_waypoint[0] = pose[0]
        self.place_waypoint[1] = pose[1]
        self.place_waypoint[3] = pose[3]
        self.place_waypoint[4] = pose[4]
        self.place_waypoint[5] = pose[5]

    def move_to_home_pose(self):
        return self.rtde_c.moveL(self.home_waypoint)

    def move_from_pickup_to_place(self):
        velocity = 0.5
        acceleration = 0.5
        no_blend = 0
        small_blend = 0.02
        temp_pickup = self.pickup_waypoint
        temp_place = self.place_waypoint
        temp_home = self.home_waypoint

        temp_pickup.append(velocity)
        temp_pickup.append(acceleration)
        temp_pickup.append(no_blend)

        temp_place.append(velocity)
        temp_place.append(acceleration)
        temp_place.append(no_blend)

        temp_home.append(velocity)
        temp_home.append(acceleration)
        temp_home.append(small_blend)

        path = [temp_pickup, temp_home, temp_place]
        return self.rtde_c.moveL(path)

    def move_from_place_to_pickup(self):
        velocity = 0.5
        acceleration = 0.5
        no_blend = 0
        small_blend = 0.5
        temp_pickup = self.pickup_waypoint
        temp_place = self.place_waypoint
        temp_home = self.home_waypoint

        temp_pickup.append(velocity)
        temp_pickup.append(acceleration)
        temp_pickup.append(no_blend)

        temp_place.append(velocity)
        temp_place.append(acceleration)
        temp_place.append(no_blend)

        temp_home.append(velocity)
        temp_home.append(acceleration)
        temp_home.append(small_blend)

        path = [temp_place, temp_home, temp_pickup]
        return self.rtde_c.moveL(path)

    def place_object(self):
        # Move down end effector to place pose and realse object
        # TODO find end effector place pose
        self.turn_off_suction()

    def pickup_object(self):
        # Move down end effector to pickup pose and activate suction
        # TODO find end effector pickup pose
        self.turn_on_suction()

    def move_to_pickup_pose(self, ee_pose):
        # Move to pickup pose
        return self.rtde_c.moveL(ee_pose)
    
    def move_to_place_pose(self, ee_pose):
        # Move to place pose
        return self.rtde_c.moveL(ee_pose)

    def get_pose(self):
        return self.rtde_r.getActualTCPPose()
    
    def test_waypoints(self):
        while True:
            if self.rtde_r.getDigitalOutState(1):
                self.rtde_c.moveL(self.home_waypoint, 0.075, 1, asynchronous=False)
                print("Home waypoint reached")
                self.rtde_c.stopScript()
                #self.rtde_c.moveL(self.pickup_waypoint)
                #self.rtde_c.moveL(self.home_waypoint)
                #self.rtde_c.moveL(self.place_waypoint)
                break
        
