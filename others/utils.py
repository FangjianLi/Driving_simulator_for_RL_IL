import gym
import numpy as np




class StructEnv_Highway(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space_shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = np.concatenate(self.env.reset())
        self.rew_episode = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.obs_a = np.concatenate(self.env.reset(**kwargs))
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def reset_0(self, **kwargs):
        self.env.reset(**kwargs)
        self.observation_space_shape = (self.env.observation_space.shape[0] * self.env.observation_space.shape[1],)
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = np.concatenate(self.env.reset(**kwargs))
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return np.concatenate(ob), reward, done, info

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_length(self):
        return self.len_episode

class StructEnv_Highway_multiagent(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space_shape = (self.observation_space[0].shape[-2] * self.observation_space[0].shape[-1],)
        self.action_space_modified = self.action_space[0]
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = [np.concatenate(state) for state in self.env.reset()]
        self.rew_episode = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.obs_a = [np.concatenate(state) for state in self.env.reset(**kwargs)]
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def reset_0(self, **kwargs):
        self.env.reset(**kwargs)
        self.observation_space_shape = (self.env.observation_space[0].shape[-2] * self.env.observation_space[0].shape[-1],)
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = [np.concatenate(state) for state in self.env.reset(**kwargs)]
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return [np.concatenate(state) for state in ob], reward, done, info

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_length(self):
        return self.len_episode


class StructEnv_Highway_multiagent_adv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space_shape = (self.observation_space.shape[-2] * self.observation_space.shape[-1],)
        self.action_space_modified = self.action_space[0]
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = np.concatenate(self.env.reset())
        self.rew_episode = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.obs_a = np.concatenate(self.env.reset())
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def reset_0(self, **kwargs):
        self.env.reset(**kwargs)
        self.observation_space_shape = (self.env.observation_space.shape[-2] * self.env.observation_space.shape[-1],)
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = np.concatenate(self.env.reset())
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return np.concatenate(ob), reward, done, info

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_length(self):
        return self.len_episode


class StructEnv_Highway_multiagent_tp(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space_shape = (self.observation_space[0].shape[-2] * self.observation_space[0].shape[-1],)
        self.action_space_modified = self.action_space
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = [np.concatenate(state) for state in self.env.reset()]
        self.rew_episode = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.obs_a = [np.concatenate(state) for state in self.env.reset(**kwargs)]
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def reset_0(self, **kwargs):
        self.env.reset(**kwargs)
        self.observation_space_shape = (self.env.observation_space[0].shape[-2] * self.env.observation_space[0].shape[-1],)
        # self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = [np.concatenate(state) for state in self.env.reset(**kwargs)]
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return [np.concatenate(state) for state in ob], reward, done, info

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_length(self):
        return self.len_episode



class StructEnv_Highway_Q(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space_shape = (self.observation_space.shape[-2] * self.observation_space.shape[-1],)
        self.obs_a = np.concatenate(self.env.reset()[0])
        self.rew_episode = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        states, info = self.env.reset(**kwargs)
        self.obs_a = np.concatenate(states)
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy(), info

    def reset_0(self, **kwargs):
        states, info = self.env.reset(**kwargs)
        self.observation_space_shape = (self.observation_space.shape[-2] * self.observation_space.shape[-1],)
        self.obs_a = np.concatenate(states)
        self.rew_episode = 0
        self.len_episode = 0
        return self.obs_a.copy(), info

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return np.concatenate(ob), reward, done, info

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_length(self):
        return self.len_episode


class StructEnv_AIRL_Highway(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space.shape = (self.observation_space.shape[0] * self.observation_space.shape[1],)
        self.obs_a = np.concatenate(self.env.reset())
        self.rew_episode = 0
        self.len_episode = 0
        self.rew_episode_airl = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.obs_a = np.concatenate(self.env.reset(**kwargs))
        self.rew_episode = 0
        self.rew_episode_airl = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def reset_0(self, **kwargs):
        self.env.reset(**kwargs)
        self.observation_space.shape = (self.env.observation_space.shape[0] * self.env.observation_space.shape[1],)
        self.obs_a = np.concatenate(self.env.reset(**kwargs))
        self.rew_episode = 0
        self.rew_episode_airl = 0
        self.len_episode = 0
        return self.obs_a.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.rew_episode += reward
        self.len_episode += 1
        return np.concatenate(ob), reward, done, info

    def step_airl(self, reward_airl):
        self.rew_episode_airl += reward_airl

    def get_episode_reward(self):
        return self.rew_episode

    def get_episode_reward_airl(self):
        return self.rew_episode_airl

    def get_episode_length(self):
        return self.len_episode


def test_DQN_reward(env_test, q_net, num_episodes=20):
    reward_episode_total = []

    for _ in range(num_episodes):
        d = False
        reward_episode = 0
        o = env_test.reset()

        while not d:
            a = eps_greedy(np.squeeze(q_net.get_act_q_values([o])), eps=0.05)
            o, r, d, _ = env_test.step(a)

            reward_episode += r

        reward_episode_total.append(reward_episode)

    return reward_episode_total


def eps_greedy(action_values, eps=0.1):
    if np.random.uniform(0, 1) < eps:
        return np.random.randint(len(action_values))
    else:
        return np.argmax(action_values)


def q_target_values(mini_batch_rw, mini_batch_done, av, discounted_value):
    '''
    Calculate the target value y for each transition
    '''
    max_av = np.max(av, axis=1)

    ys = []
    for r, d, av in zip(mini_batch_rw, mini_batch_done, max_av):
        if d:
            ys.append(r)
        else:
            q_step = r + discounted_value * av
            ys.append(q_step)

    assert len(ys) == len(mini_batch_rw)
    return np.array(ys)


# add several functions for CBF_sampling


def check_CBF_actions(vehicle_data):
    # some constants
    h_c = 0.8  # 1
    h_t = 0.8  # 1
    h_rc = 0.4  # 0.6
    a_l = 7
    alpha = 1
    l_c = 5

    # 0a. distill the motion information of the subject car
    subject_car_x = np.array(vehicle_data["controlled_vehicle"])[:, 0]
    subject_car_speed = np.array(vehicle_data["controlled_vehicle"])[:, 2]
    subject_car_heading = np.array(vehicle_data["controlled_vehicle"])[:, 3]
    subject_car_acceleration = np.array(vehicle_data["controlled_vehicle"])[:, 4]
    subject_car_beta = np.array(vehicle_data["controlled_vehicle"])[:, 5]
    subject_car_status_mask = np.array(np.array(vehicle_data["controlled_vehicle"])[:, 6] != 0, dtype=float)

    # 0b. distill the motion information of the current front car
    fc_car_x = np.array(vehicle_data["front_current"])[:, 0]
    fc_car_vx = np.array(vehicle_data["front_current"])[:, 1]

    # 0c. distill the motion information of the current rear car
    rc_car_x = np.array(vehicle_data["rear_current"])[:, 0]
    rc_car_vx = np.array(vehicle_data["rear_current"])[:, 1]

    # 0d. distill the motion information of the target front car
    ft_car_x = np.array(vehicle_data["front_target"])[:, 0]
    ft_car_vx = np.array(vehicle_data["front_target"])[:, 1]

    # 0e. distill the motion information of the target rear car
    rt_car_x = np.array(vehicle_data["rear_target"])[:, 0]
    rt_car_vx = np.array(vehicle_data["rear_target"])[:, 1]

    delta_h_fc = []
    delta_h_rc = []
    delta_h_ft = []
    delta_h_rt = []

    for index in range(len(subject_car_speed)):
        # 1. check the CBF with respect to front car on the current lane
        if subject_car_speed[index] > fc_car_vx[index]:
            delta_h = fc_car_vx[index] - subject_car_speed[index] * np.cos(subject_car_heading[index]) + \
                      (-h_c + (fc_car_vx[index] - subject_car_speed[index]) / a_l) * subject_car_acceleration[index] + \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (fc_car_x[index] - subject_car_x[index] - l_c - h_c * subject_car_speed[index] - (
                    fc_car_vx[index] - subject_car_speed[index]) ** 2 / 2 / a_l)
        else:
            delta_h = fc_car_vx[index] - subject_car_speed[index] * np.cos(subject_car_heading[index]) - \
                      h_c * subject_car_acceleration[index] + \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (fc_car_x[index] - subject_car_x[index] - l_c - h_c * subject_car_speed[index])
        delta_h_fc.append(delta_h)

        # 2. check the CBF with respect to rear car on the current lane
        if subject_car_speed[index] < rc_car_vx[index]:
            delta_h = -rc_car_vx[index] + subject_car_speed[index] * np.cos(subject_car_heading[index]) + \
                      (rc_car_vx[index] - subject_car_speed[index]) / a_l * subject_car_acceleration[index] - \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (subject_car_x[index] - rc_car_x[index] - l_c - h_rc * rc_car_vx[index] - (
                    rc_car_vx[index] - subject_car_speed[index]) ** 2 / 2 / a_l)
        else:

            delta_h = -rc_car_vx[index] + subject_car_speed[index] * np.cos(subject_car_heading[index]) - \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (subject_car_x[index] - rc_car_x[index] - l_c - h_rc * rc_car_vx[index])

        delta_h_rc.append(delta_h)

        # 3. check the CBF with respect to front car on the target lane
        if subject_car_speed[index] > ft_car_vx[index]:
            delta_h = ft_car_vx[index] - subject_car_speed[index] * np.cos(subject_car_heading[index]) + \
                      (-h_c + (ft_car_vx[index] - subject_car_speed[index]) / a_l) * subject_car_acceleration[index] + \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (ft_car_x[index] - subject_car_x[index] - l_c - h_c * subject_car_speed[index] - (
                    ft_car_vx[index] - subject_car_speed[index]) ** 2 / 2 / a_l)
        else:
            delta_h = ft_car_vx[index] - subject_car_speed[index] * np.cos(subject_car_heading[index]) - \
                      h_c * subject_car_acceleration[index] + \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (ft_car_x[index] - subject_car_x[index] - l_c - h_c * subject_car_speed[index])
        delta_h_ft.append(delta_h)

        # 4. check the CBF with respect to rear car on the target lane

        if subject_car_speed[index] < rt_car_vx[index]:
            delta_h = -rt_car_vx[index] + subject_car_speed[index] * np.cos(subject_car_heading[index]) + \
                      (rt_car_vx[index] - subject_car_speed[index]) / a_l * subject_car_acceleration[index] - \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (subject_car_x[index] - rt_car_x[index] - l_c - h_t * rt_car_vx[index] - (
                    rt_car_vx[index] - subject_car_speed[index]) ** 2 / 2 / a_l)
        else:

            delta_h = -rt_car_vx[index] + subject_car_speed[index] * np.cos(subject_car_heading[index]) - \
                      subject_car_speed[index] * np.sin(subject_car_heading[index]) * subject_car_beta[index] + \
                      alpha * (subject_car_x[index] - rt_car_x[index] - l_c - h_t * rt_car_vx[index])

        delta_h_rt.append(delta_h)

    # 5. return the delta CBF value
    delta_h_ft = np.multiply(subject_car_status_mask, delta_h_ft)
    delta_h_rt = np.multiply(subject_car_status_mask, delta_h_rt)

    return min(delta_h_fc), min(delta_h_rc), min(delta_h_ft), min(delta_h_rt)


def prob_h_calculator(h, epsilon):
    prob_h = np.zeros_like(h)

    safe_indices = np.argwhere(h >= 0)
    unsafe_indices = np.argwhere(h < 0)

    if len(safe_indices):
        for index in safe_indices:
            prob_h[index] = (1 - epsilon) / len(safe_indices)

    if len(unsafe_indices):
        unsafe_sum_prob = np.sum([1 / h[i] for i in unsafe_indices])
        for index in unsafe_indices:
            prob_h[index] = (epsilon) * 1 / h[index] / unsafe_sum_prob

    # normalize the the prob_h

    prob_h /= sum(prob_h)
    return prob_h


def CBF_sample_action(info, epsilon):
    ACTION_modes = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']
    pred_info = info["pred_info"]
    h_value = []
    for mode in ACTION_modes:
        h_value.append(min(check_CBF_actions(pred_info[mode])))
    # check purpose
    # print(h_value)
    prob_h = prob_h_calculator(np.array(h_value), epsilon)

    # check purpose
    # print(prob_h)
    experi_h = np.random.multinomial(1, prob_h)
    index = np.argwhere(experi_h > 0)
    return np.asscalar(index)
