import signal
import numpy as np
from collections import deque

def sigint_handler(signum, frame):
        print('exiting')
        exit()


def identify_nearest_centroid_for_multiple_tables(Q_arrangment,l_depth,state):
    
    deque_buff = deque()
    for _ in range(len(Q_arrangment)):
        if l_depth == Q_arrangment[_].depth:    
            state_index, min_distance_to_centroid = Q_arrangment[_].identify_nearest_centroid(state)
            deque_buff.append([state_index, min_distance_to_centroid, _])
            
    
    # if there is only one object I do not need to look
    if len(deque_buff) == 1:
        temp_arg = 0
    else: # if there is more than one object I need to see which centroid is closer to the state
        temp_distance = np.zeros(len(deque_buff))
        for _ in range(len(deque_buff)):
            temp_distance[_] = deque_buff[_][1]
        temp_arg = np.argmin(temp_distance)

    state_index = deque_buff[temp_arg][0]
    min_distance_to_centroid = deque_buff[temp_arg][1]
    Q_index = deque_buff[temp_arg][2]


    return min_distance_to_centroid, state_index, Q_index