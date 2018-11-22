import numpy as np
from collections import deque
import signal
import time
import pickle
# custom files
from action_chooser import action_chooser
from double_Q import DQPID 
from memory import memory_comparator
from functions import sigint_handler
from algorithm_4 import algorithm_4
from algorithm_3 import algorithm_3
from algorithm_2 import algorithm_2
from robot_dict import robot_dict as robot_dict
from long_term_memory import update_long_term_memory

# ROBOT 
# select the platform from the robot dictionary    
platform = 'pioneer_pi'
# the pioneer and other robots have different ros topic names
# with this you can select the correct one
simulation = True
# this sets the mode for the memory
mode = 'long_term_memory'

# Constants
E_GREED = 1.
EXECUTION_TIME = 400
Ts = 10.
Teval = 1. 
N_mariano = 8
memory_repetition = 8
np.random.seed(1234)

# tracking and velocity control
def main_DQPID(load):
    global platform
    signal.signal(signal.SIGINT, sigint_handler) # to execute the signal interrupt
    

    # QPID parameters
    Q_index = 0
    Q_arrange = deque()
    action_discretization_n = 3
    maximum_depth = 8

    # robot parameters
    current_robot = robot_dict[platform]
    state = current_robot['initial_state']
    set_point = current_robot['set_point']
    initial_action_centroid = current_robot['action_centroid']
    robot = current_robot['class'](set_point, dt = 1./Ts, Teval=Teval, simulation=simulation)
    K_step = current_robot['K_step']
    print('k step', K_step)
    time.sleep(1)
    # Classes instantiations
    action_selector = action_chooser(E_GREED, EXECUTION_TIME)
    memory = memory_comparator(memory_repetition)
    Q_arrange.append(DQPID(state, None,1.,initial_action_centroid, action_discretization_n, maximum_depth, 0., 0., K_step=K_step  ))
    ltm = deque(maxlen = 10000) # long term memory max 
    minibatch_size = 32
    # others
    time.sleep(1)
    state = state[0]
    if load:
        n_to_load = np.load('len_Q.npy')[0] 
        for _ in range(n_to_load):
            file = open('Q_arrange' + str(_) + '.txt','r') 
            Q_arrange[_] = pickle.load(file)

    # I propose the generation of a new class, that will be responsible for saving everything and plotting
    
    for x in range(EXECUTION_TIME):

        start = time.time()
        
        flag_ab, action, action_index, e_greed, state_index = action_selector.get(Q_arrange[Q_index], state)
        memory.update(state_index, action_index, Q_index, action)
        
        next_state = robot.update(action, Q_arrange[Q_index].depth)

        reward = robot.get_gaussian_reward(next_state,set_point)
       
        if x < N_mariano:
             Q_arrange[Q_index] = algorithm_2(Q_arrange[Q_index],state_index,next_state,reward,action_index, flag_ab)
             next_Q_index = Q_index
             memory.counter = 0

        else:
            memory.compare()
            if memory.flag_no_variation == True:
                #print('algorithm 4')
                Q_arrangement, Q_index, next_Q_index, action_selector.e_greed_counter = algorithm_4(Q_arrange,memory.Mt,next_state,reward,maximum_depth,action_discretization_n,action_selector.e_greed_counter,set_point, flag_ab, K_step)
                memory.flag_no_variation = False
            else:
                #print('algorithm 3')
                Q_arrange, Q_index, next_Q_index = algorithm_3(Q_arrange,memory.Mt,next_state,reward,flag_ab)
            
        end = time.time()
        
        ltm.append([state, next_state, action, Q_index, next_Q_index, reward, flag_ab, action_index])
        if mode=='long_term_memory':
            Q_arrange = update_long_term_memory(ltm, Q_arrange, minibatch_size)

        Q_index = next_Q_index
        state = next_state

        print(x, 'R',  round(reward,3),'s', next_state, 'depth', Q_arrange[Q_index].depth, 't', round(end-start,2) )



    #saving Q tables
    for _ in range(len(Q_arrange)):
        file = open('Q_arrange' + str(_) + '.txt','w') 
        pickle.dump(Q_arrange[_], file)

    np.save('len_Q.npy', np.array([len(Q_arrange)]))
    
    # ploting and printing performance
    print('actions', action)
    robot.plotter.plot(savefig=True)
    robot.plotter.save_values()
    mse = robot.plotter.mean_squared_error(set_point)
    print('mse', mse, 'mean mse', np.mean(mse))
    euclidean_distance = robot.plotter.euclidean_distance(set_point)
    print('euclidean_distance', euclidean_distance)
    mahalanobis = robot.plotter.mahalanobis(set_point)
    print('mahalanobis', mahalanobis)
    
    
    robot.stop()
    


if __name__ == '__main__':
    main_DQPID(load = False)
