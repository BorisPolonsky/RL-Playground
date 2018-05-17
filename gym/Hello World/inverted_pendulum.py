import gym
import tensorflow as tf
import random
import numpy as np

cache_path = "./cache"

env = gym.make("CartPole-v1")
hidden_dim = 100
lr = 1e-4
n_hidden_layer = 1
# This network takes the 4-dimensional observation of the environment as state input, and outputs a 2-dimensional vector
# representing the Q-value for taking action 0 or 1, respectively.
with tf.variable_scope("DQN-Model"):
    net_input = net_output = tf.placeholder(shape=[None, 4], dtype=tf.float32)
    for layer_i in range(n_hidden_layer):
        net_output = tf.layers.dense(inputs=net_output, units=hidden_dim, activation=tf.nn.relu, use_bias=True)
    net_output = tf.layers.dense(inputs=net_output, units=2, use_bias=True)
with tf.variable_scope("training-config"):
    target_output = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    loss = tf.reduce_mean((target_output - net_output)**2)
    learning_rate = tf.get_variable(dtype=tf.float32, initializer=lr, name="learning_rate")
    global_step = tf.get_variable(dtype=tf.int32, initializer=0, name="global_step")
    total_reward = tf.get_variable(dtype=tf.float32, initializer=0.0, name="total_reward")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimize_op = optimizer.minimize(loss, global_step=global_step)
    training_loss_summary = tf.summary.scalar(name="training_loss", tensor=loss)
    reward_summary = tf.summary.scalar(name="training_loss", tensor=total_reward)

num_of_episode = 3000
starting_exploitation_rate = 0.3
maximum_exploitation_rate = 0.9
exploration_rate_decay = 1 - 1e-3
gamma = 0.8  # Discount Factor
batch_size = 50
max_memory_size = 10000

with tf.Session() as sess:
    writer = tf.summary.FileWriter(cache_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(learning_rate, lr))
    memory = []  # (state, action, reward, next_state)
    exploitation_rate = starting_exploitation_rate
    for i_episode in range(num_of_episode):
        observation = prev_observation = env.reset()
        num_of_steps = 0
        total_reward_value = 0
        print("Episode {} begins with exploitation rate: {}.".format(i_episode, exploitation_rate))
        while True:
            env.render()
            if random.random() < exploitation_rate:
                q_vals = sess.run(net_output, feed_dict={net_input: observation[np.newaxis, :]})
                action = np.argmax(q_vals[0])  # Exploitation
            else:
                action = env.action_space.sample()  # Exploration
            prev_observation = observation
            observation, reward, done, info = env.step(action)
            memory.append((prev_observation, action, reward, (observation if not done else None)))
            if len(memory) > max_memory_size:
                memory.pop(0)

            batch = random.sample(memory, batch_size if batch_size <= len(memory) else len(memory))
            states = np.array([mem[0] for mem in batch])
            next_states = np.array([mem[3] if mem[3] is not None else np.zeros(env.observation_space.shape[0]
)for mem in batch])
            max_q_val_next = np.max(sess.run(net_output, feed_dict={net_input: next_states}), axis=1)
            target_q_val = sess.run(net_output, feed_dict={net_input: states})
            for i_sample, (_, current_action, current_reward, next_state) in enumerate(batch):
                if next_state is None:
                    target_q_val[i_sample][current_action] = current_reward
                else:
                    target_q_val[i_sample][current_action] = current_reward + gamma * max_q_val_next[i_sample]
            _, global_step_val, summary = sess.run([optimize_op, global_step, training_loss_summary],
                                                   feed_dict={net_input: states, target_output: target_q_val})
            writer.add_summary(summary=summary, global_step=global_step_val)
            num_of_steps += 1
            total_reward_value += reward
            if done:
                print("Episode {} ended after {} step(s).".format(i_episode, num_of_steps))
                sess.run(tf.assign(total_reward, total_reward_value))
                writer.add_summary(summary=summary, global_step=i_episode+1)
                exploitation_rate = 1 - (1 - exploitation_rate) * exploration_rate_decay
                if maximum_exploitation_rate is not None and exploitation_rate > maximum_exploitation_rate:
                    exploitation_rate = maximum_exploitation_rate
                break
