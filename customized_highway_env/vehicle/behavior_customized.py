from highway_env.vehicle.behavior import IDMVehicle
import numpy as np
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class IDMVehicle_original(IDMVehicle):
    LENGTH = 4.7
    WIDTH = 2.1
    safe_distance = 5
    initial_distance_f = 20
    initial_distance_r = 20
    @classmethod
    def create_random(cls, road,
                      speed=None,
                      lane_from=None,
                      lane_to=None,
                      lane_id=None,
                      spacing=1,
                      target_speed=None):

        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7 * lane.speed_limit, lane.speed_limit)
            else:
                speed = road.np_random.uniform(IDMVehicle_original.DEFAULT_SPEEDS[0],
                                               IDMVehicle_original.DEFAULT_SPEEDS[1])
        default_spacing = 15 + 1.2 * speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))

        if not len(road.vehicles):
            x0 = 3 * offset
            x0 += offset * road.np_random.uniform(0.9, 1.1)
        elif np.random.rand(1) > 0.2:
            x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            x0 += offset * road.np_random.uniform(0.9, 1.1)
        else:
            x0 = np.min([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            x0 -= offset * road.np_random.uniform(0.9, 1.1)

        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed, target_speed=target_speed)

        return v

    @classmethod
    def create_random_adw_TZMG(cls, road,
                      speed=None,
                      lane_from=None,
                      lane_to=None,
                      lane_id=None,
                      spacing=1,
                      target_speed=None):

        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7 * lane.speed_limit, lane.speed_limit)
            else:
                speed = road.np_random.uniform(IDMVehicle_original.DEFAULT_SPEEDS[0],
                                               IDMVehicle_original.DEFAULT_SPEEDS[1])
        default_spacing = 15 + 1.2 * speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))

        if not len(road.vehicles):
            x0 = 3 * offset
            x0 += offset * road.np_random.uniform(0.9, 1.1)
        elif np.random.rand(1) > 0.5:
            x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            x0 += offset * road.np_random.uniform(0.9, 1.1)
        else:
            x0 = np.min([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            x0 -= offset * road.np_random.uniform(0.9, 1.1)

        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed, target_speed=target_speed)

        return v

    @classmethod
    def create_f_s_car(cls, road, subject_car):
        speed = np.random.uniform(20,30)
        heading = 0
        lane = road.network.get_lane(subject_car.lane_index)
        pos_x = lane.local_coordinates(subject_car.position)[0] + np.random.uniform(cls.safe_distance, cls.initial_distance_f)
        v = cls(road, lane.position(pos_x, 0), heading, speed)
        return v

    @classmethod
    def create_r_s_car(cls, road, subject_car):
        speed = np.random.uniform(20, 30)
        heading = 0
        lane = road.network.get_lane(subject_car.lane_index)
        pos_x = lane.local_coordinates(subject_car.position)[0] - np.random.uniform(cls.safe_distance, cls.initial_distance_f)
        v = cls(road, lane.position(pos_x, 0), heading, speed)
        return v

    @classmethod
    def create_f_l_car(cls, road, subject_car):
        speed = np.random.uniform(20, 30)
        heading = 0
        if subject_car.lane_index[2] == 0:
            return None
        lane_index = ('0', '1', subject_car.lane_index[2] -1)
        lane = road.network.get_lane(lane_index)
        pos_x = lane.local_coordinates(subject_car.position)[0] + np.random.uniform(0.1, cls.initial_distance_f)
        v = cls(road, lane.position(pos_x, 0), heading, speed)
        return v

    @classmethod
    def create_r_l_car(cls, road, subject_car, l_f_car=None):
        speed = np.random.uniform(20, 30)
        heading = 0
        if subject_car.lane_index[2] == 0:
            return None
        lane_index = ('0', '1', subject_car.lane_index[2] - 1)
        lane = road.network.get_lane(lane_index)
        if l_f_car:
            pos_x = min(lane.local_coordinates(subject_car.position)[0], l_f_car.position[0] - cls.safe_distance) - np.random.uniform(0.1, cls.initial_distance_r)
        else:
            pos_x = lane.local_coordinates(subject_car.position)[0] - np.random.uniform(0.1, cls.initial_distance_r)

        v = cls(road, lane.position(pos_x, 0), heading, speed)
        return v

    @classmethod
    def create_f_r_car(cls, road, subject_car):
        speed = np.random.uniform(20, 30)
        heading = 0
        if subject_car.lane_index[2] == len(road.network.graph['0']['1'])-1:
            return None
        lane_index = ('0', '1', subject_car.lane_index[2] + 1)
        lane = road.network.get_lane(lane_index)
        pos_x = lane.local_coordinates(subject_car.position)[0] + np.random.uniform(0.1, cls.initial_distance_f)
        v = cls(road, lane.position(pos_x, 0), heading, speed)
        return v

    @classmethod
    def create_r_r_car(cls, road, subject_car, r_f_car=None):
        speed = np.random.uniform(20, 30)
        heading = 0
        if subject_car.lane_index[2] == len(road.network.graph['0']['1']) - 1:
            return None
        lane_index = ('0', '1', subject_car.lane_index[2] + 1)
        lane = road.network.get_lane(lane_index)
        if r_f_car:
            pos_x = min(lane.local_coordinates(subject_car.position)[0], r_f_car.position[0] - cls.safe_distance) - np.random.uniform(
                0.1, cls.initial_distance_r)
        else:
            pos_x = lane.local_coordinates(subject_car.position)[0] - np.random.uniform(0.1, cls.initial_distance_r)

        v = cls(road, lane.position(pos_x, 0), heading, speed)
        return v

    @classmethod
    def create_second_car(cls, road, subject_car):
        if subject_car.lane_index[2] == 0:
            lottery_zone = [1,2, 5,6]
        elif subject_car.lane_index[2] == len(road.network.graph['0']['1']) - 1:
            lottery_zone = [1,2, 3,4]
        else:
            lottery_zone = [1, 2, 3, 4, 5, 6]

        lottery_ticket = np.random.choice(lottery_zone)

        if lottery_ticket == 1:
            v = cls.create_f_s_car(road, subject_car)
        elif lottery_ticket == 2:
            v = cls.create_r_s_car(road, subject_car)
        elif lottery_ticket == 3:
            v = cls.create_f_l_car(road, subject_car)
        elif lottery_ticket == 4:
            v = cls.create_r_l_car(road, subject_car)
        elif lottery_ticket == 5:
            v = cls.create_f_r_car(road, subject_car)
        else:
            v = cls.create_r_r_car(road, subject_car)

        return v, lottery_ticket



class no_input_IDMVehicle(ControlledVehicle):
    LENGTH = 4.7
    WIDTH = 2.1

    def __init__(self,
                 road,
                 position,
                 heading: float = 0,
                 speed: float = 0):
        super().__init__(road, position, heading, speed)

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls,  vehicle: IDMVehicle) -> "no_input_vehicle":

        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed)
        return v

    @classmethod
    def clone_from(cls, road_clone, vehicle: IDMVehicle) -> "no_input_vehicle":

        v = cls(road_clone, vehicle.position, heading=vehicle.heading, speed=vehicle.speed)
        return v

    def act(self, action = None):

        action = {}
        action['steering'] = 0
        action['acceleration'] = 0
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float):

        super().step(dt)