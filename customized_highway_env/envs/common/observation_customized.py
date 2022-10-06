from highway_env.envs.common.observation import *


class KinematicObservation_original(KinematicObservation):

    def observe_CBF(self) -> np.ndarray:  ## this is the new observe
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        # sort = self.order == "sorted"
        close_vehicles = self.env.road.close_vehicles_to_CBF_v0(self.observer_vehicle,
                                                                self.env.PERCEPTION_DISTANCE,
                                                                count=self.vehicles_count - 1,
                                                                see_behind=self.see_behind)
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs

    def inject_vehicles(self, f_s_vehicle, f_r_vehicle, r_r_vehicle):
        self.f_s_vehicle = f_s_vehicle
        self.f_r_vehicle = f_r_vehicle
        self.r_r_vehicle = r_r_vehicle
        self.second_player_vehicle_list = [self.f_s_vehicle, self.f_r_vehicle, self.r_r_vehicle]

    def observe_second_player(self):
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        close_vehicles_subject = self.env.road.close_vehicles_to_CBF_v0(self.observer_vehicle,
                                                                        self.env.PERCEPTION_DISTANCE,
                                                                        count=self.vehicles_count - 1,
                                                                        see_behind=self.see_behind)

        additional_vehicle = None
        for vehicle in close_vehicles_subject:
            if vehicle not in self.second_player_vehicle_list:
                additional_vehicle = vehicle

        origin = self.observer_vehicle if not self.absolute else None
        df = df.append(pd.DataFrame.from_records([v.to_dict(origin, observe_intentions=self.observe_intentions) for v in
                                                  self.second_player_vehicle_list[-self.vehicles_count + 1:]])[self.features],
                       ignore_index=True)

        if additional_vehicle:
            rows = np.zeros((1, len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        else:
            df = df.append(pd.DataFrame.from_records(
                [additional_vehicle.to_dict(origin, observe_intentions=self.observe_intentions)])[self.features],
                           ignore_index=True)
        if self.normalize:
            df = self.normalize_obs(df)
        df = df[self.features]
        obs = df.values.copy()
        return obs

    def observe_CBF_clone(self) -> np.ndarray:  ## this is for the cloned environment

        observer_vehicle_clone = self.env.controlled_vehicle_clone

        # Add ego-vehicle
        df = pd.DataFrame.from_records([observer_vehicle_clone.to_dict()])[self.features]
        close_vehicles = self.env.road_clone.close_vehicles_to_CBF_v0(observer_vehicle_clone,
                                                                      self.env.PERCEPTION_DISTANCE,
                                                                      count=self.vehicles_count - 1,
                                                                      see_behind=self.see_behind)
        if close_vehicles:
            origin = observer_vehicle_clone if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)  ## we should be able to use the self.normalize_obs here
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs


class MultiAgentObservation_original(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory_original(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])

    def observe_CBF(self) -> tuple:
        return tuple(obs_type.observe_CBF() for obs_type in self.agents_observation_types)


def observation_factory_original(env: 'AbstractEnv', config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics_original":
        return KinematicObservation_original(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation_original":
        return MultiAgentObservation_original(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
