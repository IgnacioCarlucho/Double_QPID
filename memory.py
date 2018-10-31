import numpy as np


class memory_comparator():

    def __init__ (self, memory_repetition):
        self.counter = 0
        self.memory_repetition = memory_repetition
        self.flag_no_variation = False
        self.forget_counter = 0
        # I randomize the memory so that I dont have initial issues with comparissons
        self.Mt =  np.random.rand(9)
        self.previous_Mt =  np.random.rand(9)

    def update(self, state_index, action_index, Q_index, action):
        self.previous_Mt = self.Mt
        self.Mt = np.hstack([state_index, action_index, Q_index, action])

    def compare(self):

        if np.array_equal(self.previous_Mt, self.Mt): # memory is equal, i remember
            self.counter += 1
            self.forget_counter = 0
        elif np.allclose(self.previous_Mt, self.Mt, rtol = 1e-04): # I check again now comparing with more care
            self.counter += 1
            self.forget_counter = 0
        else:
            # memory has changed I forget
            self.counter = 0
            self.forget_counter +=1

        if self.counter == self.memory_repetition:
            print('memory has been equal for n periods')
            self.flag_no_variation = True
            self.counter = 0
        
        if self.forget_counter > 1000:
            print('you should forget this')

