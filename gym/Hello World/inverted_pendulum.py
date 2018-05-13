import gym
import tensorflow as tf
import random
import numpy as np
env = gym.make("CartPole-v0")
hidden_dim = 1000
lr = 1e-3
with tf.variable_scope("DQN-Model"):
    net_input = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    net_output = tf.layers.dense(inputs=net_input, units=hidden_dim, activation=tf.nn.relu, use_bias=True)
    net_output = tf.layers.dense(inputs=net_output, units=2, activation=tf.nn.relu, use_bias=True)
with tf.variable_scope("training-config"):
    target_output = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    loss = tf.reduce_mean((target_output - net_output)**2)
    learning_rate = tf.get_variable(dtype=tf.float32, initializer=lr, name="learning_rate")
    global_step = tf.get_variable(dtype=tf.int32, initializer=0, name="global_step")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimize_op = optimizer.minimize(loss, global_step=global_step)

num_of_episode = 1000
exploitation_rate = 0.1
exploration_rate_decay = 1 - 1e-3
gamma = 0.8  # Discount Factor
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(learning_rate, lr))
    for i_episode in range(num_of_episode):
        observation = prev_observation = env.reset()
        num_of_steps = 0
        print("Episode {} begins with exploitation rate: {}.".format(i_episode, exploitation_rate))
        while True:
            env.render()
            q_vals = sess.run(net_output, feed_dict={net_input: observation[np.newaxis, :]})
            if random.random() < exploitation_rate:
                action = np.argmax(q_vals[0])
            else:
                action = env.action_space.sample()
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            if done:
                q_vals[0][action] = reward
            else:
                q_vals[0][action] = reward + max(sess.run(net_output, feed_dict={net_input: observation[np.newaxis, :]})[0])
            sess.run(optimize_op, feed_dict={net_input: prev_observation[np.newaxis, :],
                                             target_output: q_vals})
            num_of_steps += 1
            if done:
                print("Episode {} ended after {} step(s).".format(i_episode, num_of_steps))
                exploitation_rate = 1 - (1 - exploitation_rate) * exploration_rate_decay
                break