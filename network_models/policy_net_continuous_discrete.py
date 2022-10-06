import tensorflow as tf
import numpy as np
import gym
np.random.seed(0)
tf.compat.v1.set_random_seed(0)

class Policy_net:
    def __init__(self, name: str, env, units_p, units_v, activation_p=tf.nn.relu,
                 activation_p_last=tf.tanh, activation_p_last_d = tf.nn.softmax, activation_v=tf.nn.relu, train_stddev=True):
        """
        :param name: string
        :param env: gym env
        """
        try:
            discrete_env_check = isinstance(env.action_space_modified, gym.spaces.discrete.Discrete)
        except AttributeError:
            discrete_env_check = isinstance( env.action_space, gym.spaces.discrete.Discrete)

        #ob_space = env.observation_space

        try:
            ob_space_shape = env.observation_space_shape
        except AttributeError:
            ob_space_shape = env.observation_space.shape

        if discrete_env_check:
            try:
                act_dim = env.action_space_modified.n
            except AttributeError:
                act_dim = env.action_space.n
        else:
            act_dim = env.action_space.shape[0]
            action_high = env.action_space.high
            action_low = env.action_space.low


        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space_shape), name='obs')

            if discrete_env_check:
                self.acts = tf.placeholder(dtype=tf.int32, shape=[None], name='acts')
            else:
                self.acts = tf.placeholder(dtype=tf.float32, shape=[None, act_dim], name='acts')

            #initializer = tf.random_normal_initializer()
            initializer = None



            with tf.variable_scope('value_net'):
                layer_v = self.obs
                for l_v in units_v:
                    layer_v = tf.layers.dense(inputs=layer_v, units=l_v, activation=activation_v)
                self.v_preds = tf.layers.dense(inputs=layer_v, units=1, activation=None)

                self.scope_2 = tf.get_variable_scope().name



            if discrete_env_check:
                with tf.variable_scope('policy_net'):
                    layer_p = self.obs
                    for l_p in units_p:
                        layer_p = tf.layers.dense(inputs=layer_p, units=l_p, activation=activation_p,
                                                  kernel_initializer=initializer)

                    self.act_probs = tf.layers.dense(inputs=layer_p, units=act_dim, activation=activation_p_last_d,
                                                     kernel_initializer=initializer)
                    self.scope_1 = tf.get_variable_scope().name

                self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
                self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

                self.act_deterministic = tf.argmax(self.act_probs, axis=1)

                self.act_probs = self.act_probs * tf.one_hot(indices=self.acts, depth=act_dim)

                self.act_probs = tf.reduce_sum(self.act_probs, axis=1)


            else:
                with tf.variable_scope('policy_net'):
                    layer_p = self.obs
                    for l_p in units_p:
                        layer_p = tf.layers.dense(inputs=layer_p, units=l_p, activation=activation_p,
                                                  kernel_initializer=initializer)
                    self.p_means = tf.layers.dense(inputs=layer_p, units=act_dim, activation=activation_p_last,
                                                  kernel_initializer=initializer) #2 is here

                    if train_stddev:
                        self.log_std = tf.get_variable(name='log_std', initializer=np.zeros(act_dim, dtype=np.float32) )
                    else:
                        self.log_std = tf.constant(np.zeros(act_dim, dtype=np.float32))
                    self.scope_1 = tf.get_variable_scope().name

                self.p_noisy = self.p_means + tf.random_normal(tf.shape(self.p_means), 0, 1) * tf.exp(self.log_std)
                self.act_stochastic = tf.clip_by_value(self.p_noisy, action_low, action_high)

                def gaussian_log_likelihood(x, mean, log_std):
                    log_p = -0.5 * ((x - mean) ** 2 / (tf.exp(log_std) ** 2 + 1e-9) + 2 * log_std + np.log(2 * np.pi))
                    return tf.reduce_sum(log_p, axis=-1)
                self.act_probs = tf.exp(gaussian_log_likelihood(self.acts, self.p_means, self.log_std))




            self.scope = tf.get_variable_scope().name


    def act(self, obs):
        return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})

    def get_action(self, obs):
        return tf.get_default_session().run(self.act_stochastic, feed_dict={self.obs: obs})

    def get_value(self, obs):
        return tf.get_default_session().run(self.v_preds, feed_dict={self.obs: obs})



    def get_action_prob(self, obs, acts):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs, self.acts: acts})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_trainable_variables_policy(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_1)

    def get_trainable_variables_value(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope_2)

