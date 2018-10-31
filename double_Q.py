import numpy as np
from generate_k_table_pioneer_6 import generate_k_table_pioneer_6, generate_k_table_pioneer_4, generate_k_table_pioneer_3, generate_k_table_pioneer_2
import time

np.random.seed(1234)

# Q cheat and get_action_index are the bottle necks. 
# Finsh everything and then optimize it. 

# For testing
'''
import numpy as np
import numpy
numpy.set_printoptions(threshold=numpy.nan)
from double_Q import DQPID 
Q_arrange = np.empty(shape=(20,), dtype=object)
Q_index = 0
Q_arrange[Q_index] = DQPID(np.array([[0., 0.]]),0.,1.,[0.05, 0.06, 0.05, 0.05, 0.06, 0.05],3., 7., 0., 0. )
a.identify_nearest_centroid([1.,1.])


b = DQPID(np.array([[0., 0.],[1.,2.]]),0.,1.,[0.05, 0.06, 0.05, 0.05, 0.06, 0.05],3., 7., 0., 0. )
a.get_new_centroid(np.array([1.,2.]))
a.identify_nearest_centroid([1.,1.])
a.get_action_index(np.array([0.05, 0.06, 0.05, 0.05, 0.06, 0.05]))


b = DQPID(np.array([[0., 1.]]),1.,2.,[0.05, 0.06, 0.05, 0.05, 0.06, 0.05],3., 7., 0., 0. )
c = DQPID(np.array([[0., 2.]]),1.,2.,[0.0, 0.0, 0.0, 0.01, 0.01, 0.01],3., 7., 0., 0. )
a.identify_nearest_centroid([1.,1.])
'''



class DQPID():
         
    
    
    # the constructor takes 0.006s. it is quite fast. 
    def __init__ (self, centroids, ascendence, depth, k_centroids, k_size, maximum_depth, action_index = 0., Q_cheat = 0. , K_step = 0.3 ):

        start = time.time()
        # constants
        self.K_STEP_DEFAULT = K_step
        self.DELTA_STATE = 0.1
        self.ALPHA = 0.2
        self.GAMMA = 0.95
        # variables 
        self.descendence = 0 # descendence is added later
        self.descendence_index = []
        self.ascendence = ascendence
        self.states_size = 1. # TODO the size of the centroid vector???
        self.centroids = centroids
        self.number_of_centroids = self.centroids.shape[0] # it is the first dimesion that i need
        self.depth = depth   
        self.k_centroids = k_centroids  
        self.k_centroids_original = k_centroids
        self.control_variables = len(self.k_centroids) # TODO check that this is correct
        self.maximum_depth = maximum_depth
        self.action_index = action_index

        self.k_max = np.zeros(self.control_variables)  
        self.k_min = np.zeros(self.control_variables)  
        self.k_step = np.zeros(self.control_variables)  
        self.k_size = k_size  
  
        self.number_of_actions = np.power(self.k_size, self.control_variables).astype(int)
        
        #end  = time.time()
        #print(end -start)
        if self.depth == 1: 
            #this is the first object
            self.h = np.array([self.centroids, 0.])
            self.max_state = +self.DELTA_STATE
            self.min_state = -self.DELTA_STATE
            for _ in range(self.control_variables):
                self.k_step[_] = self.K_STEP_DEFAULT
                self.k_max[_] = self.k_centroids[_] + self.K_STEP_DEFAULT*((self.k_size-1.)/2.)
                self.k_min[_] = self.k_centroids[_] - self.K_STEP_DEFAULT*((self.k_size-1.)/2.)

        else:
            # this is not the first object 
            self.h = np.array([self.centroids, self.action_index])
            radio_min = 0.005
            radio_max = 0.1
            b = radio_max
            a = (radio_min- radio_max)/self.maximum_depth
            y_radio = (a*self.depth) + b 
            self.max_state = y_radio
            self.min_state =-y_radio
            # create the spacing of the action spaces 
            for _ in range(self.control_variables):
                # a table of higher depth, the actions have to be calculated for each depth 
                correction_factor = 0.75
                if (self.k_centroids[_] != 0.):
                    #print('building k table for higher depth')
                    self.k_step[_] = self.K_STEP_DEFAULT/(correction_factor*np.power(self.depth,2.))
                    self.k_max[_] = self.k_centroids[_] + self.k_step[_]*((self.k_size-1.)/2.)
                    self.k_min[_] = self.k_centroids[_] - self.k_step[_]*((self.k_size-1.)/2.)
                    # if one of the actions is less than zero, I make the actions smallers
                    while self.k_min[_] < 0.: 
                        self.k_step[_] = 0.9*self.k_step[_]
                        self.k_max[_] = self.k_centroids[_] + self.k_step[_]*((self.k_size-1.)/2.)
                        self.k_min[_] = self.k_centroids[_] - self.k_step[_]*((self.k_size-1.)/2.)
                        #print('Minimum value a K = ', _, ' is zero, table is being adjusted')
                else:
                        #print('k centroid is zero, la k_table has to be generated differently')
                        # I calculate the actions normally
                        self.k_step[_] = self.K_STEP_DEFAULT/(correction_factor*np.power(self.depth,2.))
                        self.k_min[_] = 0. # min value is zero of course
                        self.k_max[_] = self.k_centroids[_] + self.k_step[_]*((self.k_size-1.)/2.)
                        # And then I recalculate the step
                        self.k_step[_] = self.k_max[_] - (self.k_min[_]/(self.k_size-1.))
                        self.k_max[_] = 0. + self.k_step[_]*((self.k_size-1.))

        #end2 = time.time()
        #print(end2-start)

        # create k_table
        if self.control_variables == 6:
            self.k_table = generate_k_table_pioneer_6(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        if self.control_variables == 4:
            self.k_table = generate_k_table_pioneer_4(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        if self.control_variables == 3:
            self.k_table = generate_k_table_pioneer_3(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        if self.control_variables == 2:
            self.k_table = generate_k_table_pioneer_2(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        
        # create Q_table
        # optimized method
        self.Q_A = -0.5 + np.multiply(-.5, np.random.rand(self.number_of_centroids, self.number_of_actions)) #before 0.1
        self.Q_B = -0.5 + np.multiply(-.5, np.random.rand(self.number_of_centroids, self.number_of_actions))
         # old method
        #self.Q_A = np.zeros((self.number_of_centroids, self.number_of_actions))
        #self.Q_B = np.zeros((self.number_of_centroids, self.number_of_actions))
        #for i in range(self.number_of_centroids):
        #    for j in range(self.number_of_actions):
        #        self.Q_A[i][j] = -0.5*np.random.rand(1) - 0.5       
        #        self.Q_B[i][j] = -0.5*np.random.rand(1) - 0.5

        # Cheat
        '''
        if self.depth > 1.: 
            print('getting q cheat')
            try:
                for _ in range(self.number_of_actions):
                    value = np.array_equal(self.k_table[_], self.k_centroids)
                    if value == True:
                        self.Q_A[0][_] = Q_cheat
                        self.Q_B[0][_] = Q_cheat
            except: 
                print('was not able to get q_cheat')    
        '''      
         

        end3 = time.time()
        #print('el tiempo de ejecucione interno',end3 -start)   

       
    def identify_nearest_centroid(self, state):


        
        distance_to_centroid = np.zeros(self.number_of_centroids)
        for _ in range(self.number_of_centroids):
            distance_to_centroid[_]=  np.linalg.norm(np.subtract(state, self.centroids[_])  )

        index_of_near_centroid = np.argmin(distance_to_centroid)
        min_distance_to_centroid = distance_to_centroid[index_of_near_centroid]
        

        return index_of_near_centroid, min_distance_to_centroid
        
    
    
    def get_new_centroid(self,new_centroid):

        #TODO add a checking mechanism to see if this new centroid it is really a new centroid
        self.number_of_centroids = self.number_of_centroids + 1 
        print('self.centroids',self.centroids,'new_centroid',new_centroid)
        self.centroids = np.vstack((self.centroids, new_centroid))
        print(self.centroids)
        new_Q_A_row = np.zeros(self.number_of_actions)
        new_Q_B_row = np.zeros(self.number_of_actions)
        for _ in range(self.number_of_actions):
            new_Q_A_row[_] = -0.5*np.random.rand(1) - 0.5  
            new_Q_B_row[_] = -0.5*np.random.rand(1) - 0.5  

        self.Q_A = np.vstack((self.Q_A, new_Q_A_row))   
        self.Q_B = np.vstack((self.Q_B, new_Q_B_row))
        # returns the index of the new centroid
        new_centroid_index = self.number_of_centroids-1
        return new_centroid_index
    
    def update_Q(self, centroid_index, action_index, reward, Q_max_next_value,flag_ab):
        
        #print('centroid_index',centroid_index)
        #print('action_index',action_index)
        if flag_ab == 'A':
            self.Q_A[centroid_index][action_index] = self.Q_A[centroid_index][action_index] + self.ALPHA*(reward + self.GAMMA*Q_max_next_value -  self.Q_A[centroid_index][action_index]) 
        else:
            self.Q_B[centroid_index][action_index] = self.Q_B[centroid_index][action_index] + self.ALPHA*(reward + self.GAMMA*Q_max_next_value -  self.Q_B[centroid_index][action_index]) 

    

    # it works but it takes 0.05 s
    def get_action_index(self, action):
        #start = time.time()
        for _ in range(self.number_of_actions):
            value = np.allclose(self.k_table[_], action)
            if value == True:
                action_index = _
        #end = time.time()
        #print(end-start)
        return action_index


