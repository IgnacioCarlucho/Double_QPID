import numpy as np
from double_Q import DQPID 
from collections import deque
import random
def update_long_term_memory(ltm, Q_arrange, batch_size):
    
    # ltm.append([state, next_state, action, Q_index, next_Q_index, reward, flag_ab, action_index])
    if len(ltm)> batch_size:
        mini_batch = random.sample(ltm, batch_size)
        for _ in range(len(mini_batch)):

            state = mini_batch[_][0]
            next_state = mini_batch[_][1]
            action = mini_batch[_][2]
            Q_index = mini_batch[_][3]
            next_Q_index = mini_batch[_][4]
            reward = mini_batch[_][5]
            flag_ab = mini_batch[_][6]
            action_index = mini_batch[_][7]

            state_index, distance = Q_arrange[Q_index].identify_nearest_centroid(state)
            next_state_index, distance = Q_arrange[next_Q_index].identify_nearest_centroid(next_state)

            if flag_ab=='A':
                Q_B_max_next_value = np.max(Q_arrange[next_Q_index].Q_B[next_state_index])
                Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
            else:  
                Q_A_max_next_value = np.max(Q_arrange[next_Q_index].Q_A[next_state_index])
                Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)

    return Q_arrange