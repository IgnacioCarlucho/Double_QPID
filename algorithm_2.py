# algorithm 2

import numpy as np


def algorithm_2 (objeto, state_index, next_state, reward, action_index, flag_ab):


    centroid_num, distance = objeto.identify_nearest_centroid(next_state)
    # Get s' closer to actual state 
    probably_next_state_index, distance = objeto.identify_nearest_centroid(next_state)

    if (np.abs(distance) < objeto.max_state): 

        # if it is inside the vecinity s' is the state
        next_state_index = probably_next_state_index
       
    else:
        # if it is not in the vecinity I add a new state
        print('Algoritmo 2 agrega un nuevo centroide en el nivel l= , h=') 
        next_state_index = objeto.get_new_centroid(next_state)

    #print(centroid_actual)
    if flag_ab=='A':
            Q_B_max_next_value = np.max(objeto.Q_B[next_state_index])
            objeto.update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
    else:  
            Q_A_max_next_value = np.max(objeto.Q_A[next_state_index])
            objeto.update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)
    


    return objeto


