import rospy
import numpy as np
import tf.transformations
from plotter import plotter
import gym
import time



class gym_pendulum(object):

    def __init__(self, set_point, dt = 0.1, Teval = 1., simulation = True, render = not True):

        self.render = render
        #self.MY_ENV_NAME = 'Pendulum-v0'
        gym.envs.register(id='Pendulum-longer-v0', entry_point='gym.envs.classic_control:PendulumEnv',max_episode_steps= 5000 )
        self.env = gym.make('Pendulum-longer-v0')
        self.action_dim = 1
        self.state_dim = 3
        self.state = self.env.reset()
        self.info = []
        self.action = []
        self.done = False
        self.step = 0
        self.env.unwrapped.state = np.array([0.17, 0.])
        self.env.unwrapped.dt = dt
        #self.tita = np.arctan2(self.state[1], self.state[0])
        self.tita = np.arctan2(0.,0.)
        # 
        self.set_point = set_point
        self.dt = dt
        self.Teval = Teval
        self.execution = np.divide(self.Teval,self.dt).astype(int)
        self.action = np.zeros(3)
        self.reward = -1.
        self.error = np.zeros(3)
        self.u = 0.
        self.u0 = 0.
        # to plot
        self.plotter = plotter('tita', 'u', 'state', 'action' , '')
        self.time = 0. 

    def update(self, action, depth):

        if not self.done:
            
            for _ in range(self.execution):
                self.action = action[0:3]
                # update errors        
                self.error[2] = self.error[1]
                self.error[1] = self.error[0]  
                self.error[0] = self.set_point - self.tita
                # get controller commands
                self.u = self.controller_pi(self.error[0], self.error[1], self.error[2], self.action, self.u0) 
                self.u = np.clip(self.u, -2., 2.)
                self.u0 = self.u
                # do action
                #print('u',self.u, self.state, self.tita )
                self.state, self.reward, self.done, self.info = self.env.step([self.u])
                if self.render: self.env.render()
                self.tita = np.arctan2(self.state[1], self.state[0])
                # to plot
                self.time = self.time + self.dt
                self.plotter.update(self.tita, self.u, self.state, self.time, depth, self.action, [])

        else:
            print('done episode')
            self.state = False 
            self.reward = False
            time.sleep(1)

        return self.tita

    def reset(self):
        self.state = self.env.reset()
        self.reward = -1.
        self.info = []
        self.action = []
        self.done = False
        self.step = 0
        self.tita = 0.
        self.env.unwrapped.state = np.array([0.0, 0.0])
        return self.state, self.done, self.step 

    def controller_pid(self, et, et1, et2, action, u0):
        Kp = action[0]
        Ti = action[1]
        Td = action[2]

        k1 = Kp*(1+Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt-self.dt/Ti)
        k3 = Kp*(Td)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u
    def controller_pi(self, et, et1, et2, action, u0):
        Kp = action[0]
        Ti = action[1]

        k1 = Kp
        k2 =-Kp*(1 -self.dt/Ti)
        k3 = 0.

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def controller_pd(self, et, et1, et2, action, u0):
        Kp = action[0]
        Td = action[1]

        k1 = Kp*(1+Td/self.dt)
        k2 = 0.
        k3 = Kp*(Td)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def get_gaussian_reward(self, state, set_point):
        self.reward =  self.reward/1. + 1.
        return self.reward

    def stop(self):
    	print('do nothing')