from highway_env.envs.common.action import *
from customized_highway_env.vehicle.controller_customized import MDPVehicle_original

class DiscreteMetaAction_original(DiscreteMetaAction):

    @property
    def vehicle_class(self) -> Callable:
        return MDPVehicle_original


class MultiAgentAction_original(ActionType):
    def __init__(self,
                 env: 'AbstractEnv',
                 action_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.action_config = action_config
        self.agents_action_types = []
        for vehicle in self.env.controlled_vehicles:
            action_type = action_factory_original(self.env, self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([action_type.space() for action_type in self.agents_action_types])

    @property
    def vehicle_class(self) -> Callable:
        return action_factory_original(self.env, self.action_config).vehicle_class

    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)


def action_factory_original(env: 'AbstractEnv', config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "DiscreteMetaAction_original":
        return DiscreteMetaAction_original(env, **config)
    elif config["type"] == "MultiAgentAction_original":
        return MultiAgentAction_original(env, **config)
    else:
        raise ValueError("Unknown action type")