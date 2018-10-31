import numpy as np 
from functions import identify_nearest_centroid_for_multiple_tables
from algorithm_2 import algorithm_2

'''
for testing

import numpy as np
import numpy
numpy.set_printoptions(threshold=numpy.nan)
from double_Q import DQPID 
from algorithm_2 import algorithm_2
from algorithm_4 import algorithm_4
from algorithm_3 import algorithm_3
Q_index = 0
from collections import deque
Q_arrange = deque()

Q_arrange.append(DQPID(np.array([[0., 0.]]),0.,1.,[0.05, 0.06, 0.05, 0.05, 0.06, 0.05],3., 7., 0., 0. ))

Q_arrange, Q_index = algorithm_3(Q_arrange,np.array([0, 0, 0, 0.05, 0.06, 0.05, 0.05, 0.06, 0.05]),[0.1, 0.2],1.,'A')

Q_arrange, Q_index = algorithm_3(Q_arrange,np.array([0, 0, 0, 0.05, 0.06, 0.05, 0.05, 0.06, 0.05]),[0.05, 0.05],1.,'A')


'''

def algorithm_3(Q_arrange,Mt,next_state,reward,flag_ab):

    # get infor from memory
    state_index = Mt[0].astype(int)
    action_index = Mt[1].astype(int)
    Q_index = Mt[2].astype(int)
    action = Mt[3:,]

    # I look for close centroids
    next_Q_index = Q_index
    next_state_index, min_distance_to_centroid = Q_arrange[next_Q_index].identify_nearest_centroid(next_state)
    delta = Q_arrange[next_Q_index].max_state
    
   
    if (np.abs(min_distance_to_centroid) <= np.abs(delta)): # if current state is inside the centroid
        l_depth = Q_arrange[next_Q_index].depth
        h_new = Q_arrange[next_Q_index].h

        # I get the maximum depth achieved so far
        L_depth_vector = np.zeros(len(Q_arrange))
        for _ in range(len(Q_arrange)):
            L_depth_vector[_] = Q_arrange[_].depth
        L_max = np.max(L_depth_vector)
        
        if (l_depth < L_max) and len(Q_arrange)>1:
            
            # if I have several objects and I am not currently on the highest achieved depth, I will go and look for an object
            # in a higher depth
            stop_flag = False

            while stop_flag == False and l_depth<L_max:
                # increase depth and check the distance 
                l_depth = l_depth + 1
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)
                
                if (np.abs(temp_min_distance_to_centroid) <= np.abs(Q_arrange[temp_Q_index].max_state)):
                    # if the current state fist inside the centroid of Higher depth, I save temporarlily
                    # But I do not stop the look, I will look in a higher depth still  
                    next_state_index = temp_state_index
                    next_Q_index = temp_Q_index
                    #fixme
                    #state_index = temp_state_index
                    #temp_l_depth = Q_arrange[Q_index].depth
                    h_new = Q_arrange[temp_Q_index].h
                    stop_flag = False
                else: 
                    # if when I looked in a higher depth the state is not in that centroid, I go  back one level and
                    # I just keep the centroid I found before
                    l_depth = l_depth - 1 
                    h_new = h_new
                    next_Q_index = next_Q_index
                    next_state_index = next_state_index
                    stop_flag = True


        else: 
            # I update what I have 
            next_Q_index = next_Q_index
            next_state_index = next_state_index

        # update Q     
        if flag_ab=='A':
            Q_B_max_next_value = np.max(Q_arrange[next_Q_index].Q_B[next_state_index])
            Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
        else:  
            Q_A_max_next_value = np.max(Q_arrange[next_Q_index].Q_A[next_state_index])
            Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)

        
    else:   
        # if the current state is outside the centroid  
        # I will have to go backwards until I find one
        l_depth = Q_arrange[Q_index].depth
        h_new = Q_arrange[Q_index].h
        stop_flag = False
        while stop_flag == False: 
            l_depth = l_depth -1 
            # I look for centroids in lower dephts
            if l_depth > 0:    
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)
            
            else: 
                l_depth = 1 # bug catcher, means deapth cant be smaller than 1 
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)


            if (np.abs(temp_min_distance_to_centroid) <= Q_arrange[temp_Q_index].max_state) and (l_depth > 1):
                # if the state is inside the lower centroid, that's good, I just keep this one
                stop_flag = True
                next_state_index = temp_state_index
                next_Q_index = temp_Q_index
                # update Q     
                if flag_ab=='A':
                    Q_B_max_next_value = np.max(Q_arrange[next_Q_index].Q_B[next_state_index])
                    Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
                else:  
                    Q_A_max_next_value = np.max(Q_arrange[next_Q_index].Q_A[next_state_index])
                    Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)



            elif l_depth == 1: 
                # if the depth is the lowest one, I have no option but to use algorithm 2 to see if there is a centroid for that state and to update it 
                l_new = 1
                Q_index = 0
                next_Q_index = 0
                h_new = Q_arrange[Q_index].h
                Q_arrange[Q_index] = algorithm_2(Q_arrange[Q_index], state_index,next_state,reward,action_index, flag_ab)
                next_Q_index = Q_index
                stop_flag = True


    return Q_arrange, Q_index, next_Q_index