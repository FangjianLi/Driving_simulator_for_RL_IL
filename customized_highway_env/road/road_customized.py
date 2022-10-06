from highway_env.road.road import Road
import numpy as np


class G_Vehicle():

    def __init__(self):
        self.position = [0, 0]
        self.velocity = [0, 0]

    def to_dict(self, origin_vehicle=None, observe_intentions=True) -> dict:
        d = {
            'presence': 0,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
        }
        return d

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()


class Road_original(Road):
    def __init__(self, network=None, vehicles=None, road_objects=None, np_random=None, record_history=False):
        super().__init__(network, vehicles, road_objects, np_random, record_history)
        self.g_car = G_Vehicle()

    def close_vehicles_to_CBF_v0(self, vehicle, distance: float, count: int = None,
                                 see_behind: bool = True) -> object:

        if vehicle.lane_index[2] == 0:
            cars_on_the_left = (None, None)
        else:
            cars_on_the_left = self.neighbour_vehicles(vehicle, ('0', '1', vehicle.lane_index[2] - 1))

        if vehicle.lane_index[2] == len(self.network.graph['0']['1'])-1: # riginal 3
            cars_on_the_right = (None, None)
        else:
            cars_on_the_right = self.neighbour_vehicles(vehicle, ('0', '1', vehicle.lane_index[2] + 1))

        CBF_vehicle_list = [self.neighbour_vehicles(vehicle, vehicle.lane_index), cars_on_the_left, cars_on_the_right]

        vehicle_list = []

        for j in CBF_vehicle_list:
            for i in j:
                if not i or np.linalg.norm(i.position - vehicle.position) > distance or (
                        not see_behind and -2 * 30 > vehicle.lane_distance_to(i)):
                    vehicle_list.append(self.g_car)
                else:
                    vehicle_list.append(i)

        if count:
            vehicle_list = vehicle_list[:count]
        return vehicle_list

    def close_vehicles_to_CBF(self, vehicle: 'kinematics.Vehicle', distance: float, count: int = None,
                              see_behind: bool = True) -> object:

        #we need to correct it

        if vehicle.lane_index[2] == 0:
            cars_on_the_left = (None, None)
        else:
            cars_on_the_left = self.neighbour_vehicles(vehicle, ('0', '1', vehicle.lane_index[2] - 1))

        if vehicle.lane_index[2] == len(self.network.graph['0']['1'])-1:
            cars_on_the_right = (None, None)
        else:
            cars_on_the_right = self.neighbour_vehicles(vehicle, ('0', '1', vehicle.lane_index[2] + 1))

        CBF_vehicle_list = [self.neighbour_vehicles(vehicle, vehicle.lane_index), cars_on_the_left, cars_on_the_right]

        vehicle_list = [i for j in CBF_vehicle_list for i in j if
                            i and np.linalg.norm(i.position - vehicle.position) < distance
                            and (see_behind or -2 * 30 < vehicle.lane_distance_to(i))]

        return vehicle_list

    def step_pred(self, dt: float) -> None:

        for vehicle in self.vehicles:
            vehicle.step(dt)

