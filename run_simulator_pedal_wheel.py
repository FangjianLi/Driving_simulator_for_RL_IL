import argparse
import gym
import numpy as np
from others.utils import StructEnv_Highway
import os
import customized_highway_env
import time

np.random.seed(256)


def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='legacy_trained_models/')
    parser.add_argument('--traj_savedir', default='trajectory/manual_driving_continuous/')
    parser.add_argument('--alg', default='SAIRL/')
    parser.add_argument('--index', default='model_3/')
    parser.add_argument('--model_index', help='save model name', default='115model.ckpt')
    parser.add_argument("--envs_k", default="highway_manual_continuous-v0")
    parser.add_argument("--envs_p", default="highway_manual_continuous_tp_carla-v0")
    parser.add_argument("--carla", default=True, help="if to trigger Carla rendering")
    parser.add_argument('--iteration', default=1, type=int)
    parser.add_argument('--min_length', default=80000, type=int)
    parser.add_argument('--duration', default=50, type=int)

    parser.add_argument('--ratio', default=1, type=int, help="The display size ratio")

    return parser.parse_args()


def highway_animation(args):
    trajectory_save_dir = args.traj_savedir

    check_and_create_dir(trajectory_save_dir)

    if args.carla:
        env = StructEnv_Highway(gym.make(args.envs_p))
    else:
        env = StructEnv_Highway(gym.make(args.envs_k))

    env.config["real_time_rendering"] = True
    env.config["manual_control"] = True
    env.config["ratio"] = args.ratio
    env.seed(256)
    env.reset_0()

    if args.carla:
        env.reset_carla()

    observation_save = []
    action_save = []
    reward_episode_counter = []
    episode_length = 0
    while True:
        time_1 = time.time()
        episode_length += 1
        next_obs, reward, done, info = env.step(np.random.rand(2))
        env.render()
        observation_save.append(env.obs_a)
        action_save.append(info['action'])

        if done:
            reward_episode_counter.append(env.get_episode_reward())
            env.reset()
            np.save(trajectory_save_dir + "observation.npy", observation_save)
            np.save(trajectory_save_dir + "action.npy", action_save)
        else:
            env.obs_a = next_obs.copy()

        if episode_length >= args.min_length:
            env.reset()
            break
        print(time.time() - time_1)

    print("The average episode reward is: {}".format(np.mean(reward_episode_counter)))
    print(np.shape(observation_save))
    print(np.shape(action_save))
    np.save(trajectory_save_dir + "observation.npy", observation_save)
    np.save(trajectory_save_dir + "action.npy", action_save)


if __name__ == '__main__':
    args = argparser()
    highway_animation(args)
