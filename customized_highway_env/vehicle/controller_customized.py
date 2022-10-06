from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from typing import List, Union
import numpy as np
import copy


class ControlledVehicle_original(ControlledVehicle):
    LENGTH = 4.7
    WIDTH = 2.1
    safe_distance = 5
    initial_distance_f = 20
    initial_distance_r = 20

    color_schme = True

    @classmethod
    def create_random(cls, road,
                      speed=None,
                      lane_from=None,
                      lane_to=None,
                      lane_id=None,
                      spacing=1):

        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7 * lane.speed_limit, lane.speed_limit)
            else:
                speed = road.np_random.uniform(ControlledVehicle_original.DEFAULT_SPEEDS[0],
                                               ControlledVehicle_original.DEFAULT_SPEEDS[1])
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

        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)

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















    def act(self, action: Union[dict, str] = None) -> None:
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if abs(target_lane_index[2] - self.lane_index[2]) < 2:
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if abs(target_lane_index[2] - self.lane_index[2]) < 2:
                self.target_lane_index = target_lane_index

        action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.speed_control(self.target_speed)}
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        super().act(action)



class MDPVehicle_original(ControlledVehicle_original):
    LENGTH = 4.7
    WIDTH = 2.1

    SPEED_COUNT: int = 3  # []
    SPEED_MIN: float = 20  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading,
                 speed,
                 target_lane_index= None,
                 target_speed= None,
                 route=None) -> None:
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:

        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            return self.SPEED_MIN

    def speed_to_index(self, speed: float) -> int:

        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        return np.int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:

        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    @classmethod
    def get_speed_index(cls, vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:

        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


class clone_MDPVehicle(ControlledVehicle_original): # we can check how it works later

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_COUNT: int = 3  # []
    SPEED_MIN: float = 20  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    def __init__(self,
                 road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index = None,
                 target_speed: float = None,
                 route = None) -> None:
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    @classmethod
    def clone_from(cls, road_clone,  vehicle: "MDPVehicle") ->"clone_MDPVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(road_clone, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            return self.SPEED_MIN

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        return np.int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    @classmethod
    def get_speed_index(cls, vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states