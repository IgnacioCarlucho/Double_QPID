import numpy as np


#TODO keep in mind that Q_object.Q[state_index] and Q_object.K_table[action_index] has to work! 


class action_chooser():


    def __init__ (self, e_greed, total_time ):

        self.e_greed = e_greed
        self.e_greed_counter = 0.
        self.e_greed_tot = np.zeros(total_time + 2 )
        self.flag_ab = 'C' # start with neither just to test the algorithm

    def get(self, Q_object, state):


        # choose A or B
        p_a_b = np.random.rand(1)
        if p_a_b > 0.5:
           self.flag_ab = 'A'
        else: 
           self.flag_ab = 'B'
        
        # e greedy decaying rule 
        self.e_greed_counter = self.e_greed_counter + 1
        self.e_greed = 0.02 + 0.3*np.exp(-self.e_greed_counter/60.*(1./1.))
        #self.e_greed = 0.02 + 0.28*np.exp(-self.e_greed_counter/60*(1/1))
        #self.e_greed_tot[self.e_greed_counter] = self.e_greed
        # choose action w/e-greedy
        probabilty = np.random.rand(1)
        if probabilty < self.e_greed:
            #TODO check if this equation makes sense in python
            action_index = 1 + np.floor((Q_object.number_of_actions - 1)* np.random.rand(1))
            action_index = action_index[0].astype(int)
            action = Q_object.k_table[action_index]
            state_index, min_distance_to_centroid = Q_object.identify_nearest_centroid(state)
        else: 
            state_index, min_distance_to_centroid = Q_object.identify_nearest_centroid(state)
            Q_C = (Q_object.Q_A[state_index] + Q_object.Q_B[state_index] )/2.
            action_index = np.argmax(Q_C).astype(int)
            action = Q_object.k_table[action_index]

        return self.flag_ab, action, action_index, self.e_greed, state_index

       

