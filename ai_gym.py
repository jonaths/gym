# Import the gym module

import gym
import skimage as skimage
from skimage import transform, color, exposure
import numpy as np
from lib.c51 import C51Agent
from lib.networks import Networks
import sys

def preprocessImg(img, size):
    # Cambia el orden. Al final hay 3 porque es RGB
    # It becomes (640, 480, 3)
    img = np.rollaxis(img, 0, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)

    return img

if __name__ == "__main__":

    # Create a breakout environment -------------------------------------------
    env = gym.make('BreakoutDeterministic-v4')

    # configure agent ---------------------------------------------------------
    action_size = env.action_space.n
    print("=action_size")
    print(action_size)

    img_rows, img_cols = 64, 64
    # We stack 4 frames
    img_channels = 4

    # C51
    num_atoms = 51

    # (64, 64, 4)
    state_size = (img_rows, img_cols, img_channels)

    # 480 x 640 x 3 canales
    x_t = env.reset()
    print("x_t 1")
    print(x_t.shape)

    # Convierte en blanco y negro y reduce dimensiones
    x_t = preprocessImg(x_t, size=(img_rows, img_cols))
    print("x_t 2")
    print(x_t.shape)

    # para que el 4?
    s_t = np.stack(([x_t] * 4), axis=2)  # It becomes 64x64x4
    print("s_t 1")
    print(s_t.shape)
    s_t = np.expand_dims(s_t, axis=0)  # 1x64x64x4
    print("s_t 2")
    print(s_t.shape)

    print("configuring agent... ")
    agent = C51Agent(state_size, action_size, num_atoms)

    misc = {'ale.lives': 5}
    prev_misc = misc

    # configure networks ------------------------------------------------------
    print("configuring networks... ")
    agent.model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)
    # agent.load_model("models/c51_ddqn.h5")
    agent.target_model = Networks.value_distribution_network(state_size, num_atoms, action_size, agent.learning_rate)

    # Reset it, returns the starting frame
    is_done = False

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)
    life = 0

    # Buffer to compute rolling statistics
    life_buffer = []

    while not is_done:

        loss = 0
        r_t = 0
        # a_t = np.zeros([action_size])

        # Epsilon Greedy
        action_idx = agent.get_action(s_t)
        # a_t[action_idx] = 1

        # Perform a random action, returns the new frame, reward and whether the game is over
        x_t1, reward, is_done, misc = env.step(action_idx)
        is_terminated = is_done

        # Render
        # env.render()

        # print(t, reward, is_terminated, misc)

        if is_terminated:
            if life > max_life:
                max_life = life
            GAME += 1
            life_buffer.append(life)
            print("Episode Finish: ", t, misc)
            x_t1 = env.reset()
            misc = {'ale.lives': 5}
            is_done = False

        # calcula el estado
        x_t1 = preprocessImg(x_t1, size=(img_rows, img_cols))
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # recupera la recompensa
        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        # update the cache
        prev_misc = misc

        # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, r_t, s_t1, is_done, t)

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            print("train_replay...")
            loss = agent.train_replay()

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            #agent.model.save_weights("models/c51_ddqn.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif agent.observe < t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if is_terminated:
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))

                # Reset rolling stats buffer
                life_buffer = []

                # Write Rolling Statistics to file
                with open("statistics/c51_ddqn_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')




