import argparse
import gym
import numpy as np
from network_models.policy_net_continuous_discrete import Policy_net
from others.utils import StructEnv_Highway_multiagent_tp
import tensorflow as tf
import os
import customized_highway_env
import time

np.random.seed(256)
tf.compat.v1.set_random_seed(256)


def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='legacy_trained_models/')
    parser.add_argument('--traj_savedir', default='trajectory/manual_driving_vertical_tp/')
    parser.add_argument('--alg', default='SAIRL/')
    parser.add_argument('--index', default='model_2/')
    parser.add_argument('--model_index', help='save model name', default='115model.ckpt')
    parser.add_argument("--carla", default=True, help="if to trigger Carla rendering")
    parser.add_argument("--envs_k", default="highway_manual_vertical_tp-v0")
    parser.add_argument("--envs_p", default="highway_manual_vertical_tp_carla-v0")
    parser.add_argument('--iteration', default=1, type=int)
    parser.add_argument('--min_length', default=80000, type=int)
    parser.add_argument('--duration', default=50, type=int)

    parser.add_argument('--ratio', default=0.75, type=int, help="The display size ratio")

    return parser.parse_args()


def highway_animation(args):
    model_save_dir = args.savedir + args.alg + args.index
    trajectory_save_dir = args.traj_savedir

    check_and_create_dir(trajectory_save_dir)

    if args.carla:
        env = gym.make(args.envs_p)
    else:
        env = gym.make(args.envs_k)
    env.config["real_time_rendering"] = True
    env.config["duration"] = args.duration
    env.config["manual_control"] = True
    env.config["ratio"] = args.ratio
    env.seed(256)

    env = StructEnv_Highway_multiagent_tp(env)
    env.reset()

    if args.carla:
        env.reset_carla()

    Policy = Policy_net('policy', env, [96, 96, 96], [128, 128, 128])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_save_dir + args.model_index)
        observation_save = []
        action_subject_save = []
        action_neighbor_save = []
        reward_episode_counter = []
        episode_length = 0
        while True:
            time_1 = time.time()
            episode_length += 1

            act = np.asscalar(Policy.act(obs=[env.obs_a[0]])[0])

            next_obs, reward, done, info = env.step(act)
            env.render()
            observation_save.append(env.obs_a)
            action_subject_save.append(info['action_subject'])
            action_neighbor_save.append(info['action_neighbor'])

            if done:
                reward_episode_counter.append(env.get_episode_reward())
                env.reset()
                np.save(trajectory_save_dir + "observation.npy", observation_save)
                np.save(trajectory_save_dir + "action_subject.npy", action_subject_save)
                np.save(trajectory_save_dir + "action_neighbor.npy", action_neighbor_save)
            else:
                env.obs_a = next_obs.copy()

            if episode_length >= args.min_length:
                env.reset()
                break
            print(time.time() - time_1)

        print("The average episode reward is: {}".format(np.mean(reward_episode_counter)))
        print(np.shape(observation_save))
        print(np.shape(action_subject_save))
        np.save(trajectory_save_dir + "observation.npy", observation_save)
        np.save(trajectory_save_dir + "action_subject.npy", action_subject_save)
        np.save(trajectory_save_dir + "action_neighbor.npy", action_neighbor_save)


if __name__ == '__main__':
    args = argparser()
    highway_animation(args)
