import numpy as np
from gym.envs.registration import register
from highway_env import utils
from highway_env.utils import near_split
from customized_highway_env.envs.common.abstract_original import AbstractEnv_original
from highway_env.road.road import RoadNetwork
from customized_highway_env.road.road_customized import Road_original
from customized_highway_env.vehicle.controller_customized import ControlledVehicle_original
from customized_highway_env.envs.common.graphics import EnvViewer_vertical
import time

ACTIONS_ALL_VALUE = {
        'LANE_LEFT': 0,
         'IDLE': 1,
        'LANE_RIGHT': 2,
         'FASTER': 3,
         'SLOWER': 4
    }


class HighwayEnv_manual_vertical(AbstractEnv_original):

    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.6  # change from 0.4 to 0.6
    LANE_CHANGE_REWARD: float = 0

    def __init__(self):
        super().__init__()
        self.time_real = time.time()
        self.action_to_take = "IDLE"
        self.action_taken = None


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics_original",
                "vehicles_count": 7, # specific environment
            },
            "action": {
                "type": "DiscreteMetaAction_original",
            },
            "lanes_count": 4,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "other_vehicles_type": "customized_highway_env.vehicle.behavior_customized.IDMVehicle_original",
            "screen_width": 300,  # [px]
            "screen_height": 200, # this is originally 150
            "initial_lane_id": None,
            "duration": 100,  # we can double check if it is second
            "ego_spacing": 2,
            "vehicles_density": 1.5, ## this is something I changed
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
        self.ratio = self.config["ratio"]

    def _create_road(self) -> None:

        self.road = Road_original(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, speed=np.random.uniform(low=20, high=25), target_speed=np.random.uniform(low=20, high=25),  spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)  ## this is something I changed

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
        return self.vehicle.crashed or \
               self.steps >= self.config["duration"] or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


    def render(self, mode: str = 'human'):

        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer_vertical(self, self.ratio)

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
            "action": self.action_taken,
        }

        return info




    def _simulate(self, action = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            # Forward action to the vehicle


            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            elif self.config["manual_control"] and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.vehicle.act(self.action_to_take)
                self.action_taken = ACTIONS_ALL_VALUE[self.action_to_take]
                self.action_to_take = "IDLE"




            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

        self.enable_auto_render = False


register(
    id='highway_manual_vertical-v0',
    entry_point='customized_highway_env.envs:HighwayEnv_manual_vertical',
)
