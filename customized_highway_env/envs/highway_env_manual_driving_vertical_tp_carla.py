import carla
import numpy as np
from gym.envs.registration import register
from highway_env import utils
from highway_env.utils import near_split
from customized_highway_env.envs.common.abstract_original import AbstractEnv_original
from highway_env.road.road import RoadNetwork
from customized_highway_env.road.road_customized import Road_original
from customized_highway_env.vehicle.controller_customized import ControlledVehicle_original, MDPVehicle_original
from customized_highway_env.envs.common.graphics import EnvViewer_vertical_tp_carla
from customized_highway_env.envs.common.carla_utils import CARLA_viusalizer
import time

ACTIONS_ALL_VALUE = {
    'LANE_LEFT': 0,
    'IDLE': 1,
    'LANE_RIGHT': 2,
    'FASTER': 3,
    'SLOWER': 4
}

VALUES_ALL_ACTION = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}


class HighwayEnv_manual_vertical_tp_carla(AbstractEnv_original):
    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.6  # change from 0.4 to 0.6
    LANE_CHANGE_REWARD: float = 0

    def __init__(self):
        self.viewer = None
        self.carla_visualizer = None
        super().__init__()
        self.time_real = time.time()
        self.action_to_take = "IDLE"
        self.action_taken = None
        self.neighbor_vehicle = None
        self.ratio = None


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
            "type": "MultiAgentObservation_original",
            "observation_config": {
                "type": "Kinematics_original",
                "vehicles_count": 7,
            }},
            "action": {
                "type": "DiscreteMetaAction_original",
            },
            "lanes_count": 4,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "other_vehicles_type": "customized_highway_env.vehicle.behavior_customized.IDMVehicle_original",
            "screen_width": 300,  # [px]
            "screen_height": 200,  # this is originally 150
            "initial_lane_id": None,
            "duration": 100,  # we can double check if it is second
            "ego_spacing": 2,
            "vehicles_density": 1.5,  ## this is something I changed
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "simulation_frequency": 10,
            "policy_frequency": 2,
            "ratio": 1,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        print("reset")

    def reset_carla(self):
        self.ratio = self.config["ratio"]
        if not self.viewer:
            self.viewer = EnvViewer_vertical_tp_carla(self, self.ratio)
        if not self.carla_visualizer:
            self.carla_visualizer = CARLA_viusalizer(image_x=1800 * self.ratio, image_y=900 * self.ratio,
                                                     animation=False, two_player=True)
        self.update_visual_carla()


    def _create_road(self) -> None:

        self.road = Road_original(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                                  np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = MDPVehicle_original.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )


            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            self.neighbor_vehicle = MDPVehicle_original.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]/10
            )
            self.neighbor_vehicle.color_schme = False
            self.controlled_vehicles.append(self.neighbor_vehicle)
            self.road.vehicles.append(self.neighbor_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, speed=np.random.uniform(low=20, high=25),
                                                            target_speed=np.random.uniform(low=20, high=25),
                                                            spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)  ## this is something I changed

    def update_info_carla(self):
        self.carla_x_list = []
        self.carla_y_list = []
        self.carla_heading_list = []

        for car_highway in self.road.vehicles:
            self.carla_x_list.append(car_highway.position[0])
            self.carla_y_list.append(car_highway.position[1])
            self.carla_heading_list.append(car_highway.heading)

    def update_visual_carla(self):
        self.update_info_carla()
        corrected_carla_x_list = np.array(self.carla_x_list) - 1000
        corrected_carla_y_list = np.array(self.carla_y_list) - 12
        corrected_carla_heading_list = np.array(self.carla_heading_list)*180/np.pi

        for car_carla, car_x, car_y, car_yaw in zip(self.carla_visualizer.car_list, corrected_carla_x_list, corrected_carla_y_list, corrected_carla_heading_list):
            car_transform = car_carla.get_transform()
            car_transform.location.x = car_x
            car_transform.location.y = car_y
            car_transform.rotation.yaw = car_yaw
            car_carla.set_transform(car_transform)

        # setup the spectator also

        car_s_transform = self.carla_visualizer.controlled_car.get_transform()
        spectator_transform = carla.Transform(car_s_transform.location + carla.Location(z=50),
                                              carla.Rotation(pitch=-90))
        self.carla_visualizer.spectator.set_transform(spectator_transform)





    def _reward(self, action) -> float:

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle_original) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
            + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                            [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or self.neighbor_vehicle.crashed or \
               self.steps >= self.config["duration"] or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

    def render(self, mode: str = 'human'):

        self.rendering_mode = mode

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
        self.should_update_rendering = False

    def _info(self, obs, action) -> dict:

        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action_subject": action,
            "action_neighbor": self.action_taken
        }

        return info

    def _simulate(self, action=None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            # Forward action to the vehicle

            if action is not None and self.config["manual_control"] and \
                    self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                # print(action)
                self.vehicle.act(VALUES_ALL_ACTION[action])

                self.neighbor_vehicle.act(self.action_to_take)
                self.action_taken = ACTIONS_ALL_VALUE[self.action_to_take]
                self.action_to_take = "IDLE"

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()
            self.update_visual_carla()

        self.enable_auto_render = False


register(
    id='highway_manual_vertical_tp_carla-v0',
    entry_point='customized_highway_env.envs:HighwayEnv_manual_vertical_tp_carla',
)
