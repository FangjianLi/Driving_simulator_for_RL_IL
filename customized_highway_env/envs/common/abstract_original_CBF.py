from highway_env.envs.common.abstract import AbstractEnv
from customized_highway_env.envs.common.observation_customized import observation_factory_original
from customized_highway_env.envs.common.action_customized import DiscreteMetaAction_original, action_factory_original
from customized_highway_env.vehicle.controller_customized import MDPVehicle_original
import os

class AbstractEnv_original_CBF(AbstractEnv):

    PERCEPTION_DISTANCE = 6.0 * MDPVehicle_original.SPEED_MAX

    def step(self, action):

        """
        The change is mainly to change the observation of surrounding vehicles
        """

        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        self._simulate(action)

        pred_info = self._do_the_prediction()  # will predict the motion of the vehicles

        obs = self.observation_type.observe_CBF()
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action, pred_info)

        return obs, reward, terminal, info

    def reset(self):

        """
        The main change is the return of the states
        """

        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self.should_update_rendering = True
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        pred_info = self._do_the_prediction()
        info = self._info(self.observation_type.observe_CBF(), 1, pred_info)
        return self.observation_type.observe_CBF(), info

    def _info(self, obs, action, pred_info):

        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "pred_info": pred_info,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass
        return info

    def _do_the_prediction(self):

        raise NotImplementedError


    def define_spaces(self) -> None:

        """
        changes because of the observation factory and action factory are changed
        """

        self.observation_type = observation_factory_original(self, self.config["observation"])
        self.action_type = action_factory_original(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()


    def get_available_actions(self):

        """
        The MDP car model is changed
        """

        if not isinstance(self.action_type, DiscreteMetaAction_original):
            raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = [self.action_type.actions_indexes['IDLE']]
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_RIGHT'])
        if self.vehicle.speed_index < self.vehicle.SPEED_COUNT - 1 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['FASTER'])
        if self.vehicle.speed_index > 0 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['SLOWER'])
        return actions

