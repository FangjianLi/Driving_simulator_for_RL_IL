import tensorflow as tf
import copy
import numpy as np
import gym


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.99, lambda_1=0.97, clip_value=0.1,  lr_policy=1e-4,
                 ep_policy=1e-8, lr_value=1e-4, ep_value=2e-8):

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma
        self.lambda_1 = lambda_1

        pi_trainable = self.Policy.get_trainable_variables()
        pi_policy_trainable = self.Policy.get_trainable_variables_policy() #modified
        pi_value_trainable = self.Policy.get_trainable_variables_value()   #modified
        old_pi_trainable = self.Old_Policy.get_trainable_variables()
        old_pi_policy_trainable = self.Old_Policy.get_trainable_variables_policy()  # modified




        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))



        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
            self.rtg_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='rtg_ph')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')


        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        self.test_1 = act_probs/act_probs_old






        with tf.variable_scope('loss'):
            # construct computation graph for loss_clip
            loss_policy = clipped_surrogate_obj(act_probs, act_probs_old, self.gaes, clip_value)
            tf.summary.scalar('total', loss_policy)


            entropy =0#tf.reduce_mean(-self.Policy.act_probs) #loss
            tf.summary.scalar('entropy', entropy)


            v_preds = self.Policy.v_preds
            loss_value = tf.reduce_mean((self.rtg_ph - tf.squeeze(self.Policy.v_preds)) ** 2)
            loss_value_v = tf.squared_difference(self.rewards + self.gamma * tf.squeeze(self.v_preds_next), tf.squeeze(v_preds))

            #tf.summary.scalar('value_difference', loss_value)

        self.test_1 = act_probs/act_probs_old




        self.merged = tf.summary.merge_all()
        optimizer_policy = tf.train.AdamOptimizer(learning_rate=lr_policy, epsilon=ep_policy) #learning_rate=1e-4, epsilon=1e-5
        optimizer_value = tf.train.AdamOptimizer(learning_rate=lr_value, epsilon=ep_value)
        optimizer_value_v=tf.train.AdamOptimizer(learning_rate=lr_value, epsilon=ep_value)



        self.train_op_policy = optimizer_policy.minimize(loss_policy, var_list=pi_trainable)
        self.train_op_value = optimizer_value.minimize(loss_value, var_list=pi_trainable)
        self.train_op_value_v = optimizer_value_v.minimize(loss_value_v, var_list=pi_trainable)




    def train_policy(self, obs, actions, gaes):
        tf.get_default_session().run([self.train_op_policy], feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.Policy.acts: actions,
                                                               self.Old_Policy.acts: actions,
                                                               self.gaes: gaes})


    def test_1_get(self, obs, actions, gaes):
        return tf.get_default_session().run([self.test_1], feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.Policy.acts: actions,
                                                               self.Old_Policy.acts: actions,
                                                               self.gaes: gaes})

    def train_value_v(self, obs, rewards, v_preds_next):
        tf.get_default_session().run([self.train_op_value_v], feed_dict={self.Policy.obs: obs,
                                                                        self.Old_Policy.obs: obs,
                                                                        self.rewards: rewards,
                                                                        self.v_preds_next: v_preds_next
                                                                       })

    def test_it(self, obs, actions, gaes):
        test_1,test_2 = tf.get_default_session().run([self.test_1, self.Policy.act_probs], feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.Policy.acts: actions,
                                                               self.Old_Policy.acts: actions,
                                                               self.gaes: gaes})
        return np.shape(test_1), np.shape(test_2)

    def train_value(self, obs, rtg):
        tf.get_default_session().run([self.train_op_value], feed_dict={self.Policy.obs: obs,
                                                                        self.Old_Policy.obs: obs,
                                                                        self.rtg_ph: rtg})




    def get_summary(self, obs, actions, gaes, rtg):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.Policy.acts: actions,
                                                                    self.Old_Policy.acts: actions,
                                                                    self.gaes: gaes,
                                                                    self.rtg_ph: rtg})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)




def clipped_surrogate_obj(new_p, old_p, adv, eps):
    '''
    Clipped surrogate objective function
    '''
    rt = new_p/old_p  # i.e. pi / old_pi
    return -tf.reduce_mean(tf.minimum(rt * adv, tf.clip_by_value(rt, 1 - eps, 1 + eps) * adv))
