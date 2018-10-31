import numpy as np
#import pylab as py
import matplotlib.pyplot as py
from collections import deque

class plotter(object):

    def __init__(self, velocities, u, positions, action_1, action_2, mode = 'normal'):

        self.actions = deque()
        self.velocity = deque()
        self.position = deque()
        self.reward = deque()
        self.depth = deque()
        self.time = deque()
        self.action_vx = deque()
        self.action_wz = deque()

        # names
        self.n_velocities = velocities
        self.n_u = u
        self.n_positions = positions
        self.n_action_1 = action_1
        self.n_action_2 = action_2

        # mode
        self.mode = mode

    def update(self, velocities,action,position,time,depth,action_vx,action_wz):
        self.actions.append(action)
        self.velocity.append(velocities)
        self.position.append(position)
        self.time.append(time)
        self.depth.append(depth)
        self.action_vx.append(action_vx)
        self.action_wz.append(action_wz)


    # the reward and the rest of the variables have different updating cicles    
    def update_reward(self, reward):
        self.reward.append(reward)

    def plot(self, savefig=False):

        
        py.plot(self.position)
        py.xlabel('Time (time steps)')
        py.ylabel(self.n_positions)
        py.title('Double QPID')
        py.legend(('x','y','z','roll', 'pitch', 'yaw' ))
        #py.axis([0, simulation_lenght, -0.5, 0.5])
        if savefig == True: py.savefig('Positions.png')
        py.show()

        py.plot(self.velocity)
        py.xlabel('Time (time steps)')
        py.ylabel(self.n_velocities)
        py.title('Double QPID')
        py.legend(('Vx', 'Wz'))
        #py.axis([0, simulation_lenght, -1., 1.])
        if savefig == True: py.savefig('velocities.png')
        py.show()

        py.plot(self.actions)
        py.xlabel('Time (time steps)')
        py.ylabel(self.n_u)
        py.title('Double QPID')
        py.legend(('u1', 'u2','3', '4' ,'5','6'  ))
        #py.axis([0, simulation_lenght, -1., 1.])
        if savefig == True: py.savefig('U.png')
        py.show()

        try:
            x0 = np.array([_[0] for _ in self.position])
            y0 = np.array([_[1] for _ in self.position])
            py.plot(x0,y0)
            py.xlabel('x')
            py.ylabel('y')
            py.title('Double QPID')
            #py.legend(('x','y','z' ))
            #py.axis([0, simulation_lenght, -0.5, 0.5])
            if savefig == True: py.savefig('pose.png')
            py.show()
        except IndexError:
            print(' x vs y could not be plotted')

        py.plot(self.depth)
        py.xlabel('Time (time steps)')
        py.ylabel('depth')
        py.title('Double QPID')
        py.legend(('1', '2','3', '4' ,'5','6'  ))
        #py.axis([0, simulation_lenght, -1., 1.])
        if savefig == True: py.savefig('depth.png')
        py.show()
        
        kp = np.array([_[0] for _ in self.action_vx])
        ki = np.array([_[1] for _ in self.action_vx])
        py.plot(self.time, kp,'b.' ,self.time, ki,'r.', linewidth=2.5)
        py.xlabel('Time (time steps)')
        py.ylabel(self.n_action_1)
        py.title('Double QPID')
        py.legend(('Kp', 'Ki','3', '4' ,'5','6'  ))
        #py.axis([0, simulation_lenght, -1., 1.])
        if savefig == True: py.savefig('actions.png')
        py.show()


    def mean_squared_error(self, set_point):
        
        try:
            if self.mode == 'ictiobot':
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[5] for _ in self.velocity])
            elif self.mode == 'ictiobot_3D':
                print('what is my purpose?')
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[1] for _ in self.velocity])
            else:
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[1] for _ in self.velocity])
            
            mse_vx = np.sqrt(np.mean((vx - set_point[0])**2))
            mse_wz = np.sqrt(np.mean((wz - set_point[1])**2))
            mse = np.array([mse_vx, mse_wz])
        except IndexError: 
            print('index error calculating for only one variable')
            mse = np.sqrt(np.mean(( np.subtract(self.velocity,set_point[0]))**2))
        
        return mse


    def mahalanobis(self, set_point):

        try: 
            if self.mode == 'ictiobot':
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[5] for _ in self.velocity])
            elif self.mode == 'ictiobot_3D':
                print('what is a 3d ictiobot?')
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[1] for _ in self.velocity])
            else:
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[1] for _ in self.velocity])

            # x centered        
            vxm = vx - set_point[0]
            wzm = wz - set_point[1]
            # Generate matrix Xc
            xc = np.transpose(np.matrix([vxm,wzm]))
            # covariance matrix
            cx = np.matmul( np.transpose(xc), xc)
            #print('cx', cx)
            cx = cx/(len(vx)-1)
            #print('cx2', cx)
            # inverse of the covariate
            cx_inv = np.linalg.inv(cx)

            # initialize MD matrix
            MD = np.zeros(len(vx))
            # calculate each coefficient
            for i in range(len(vx)):
                #diff = np.array([x1[i] - np.mean(x1),x2[i] - np.mean(x2)])
                diff = np.array([vx[i] - set_point[0] ,wz[i] - set_point[1]])
                diff_t = np.transpose(diff)
                temp = np.matmul(diff,np.array(cx_inv))
                MD[i] = np.sqrt( np.matmul(temp,diff_t))
        except IndexError: 
            print('could not calculate mahalanobis only one dimension')
            MD = 0.

        return np.mean(MD) 

    def euclidean_distance(self, set_point):

        try:

            if self.mode == 'ictiobot':
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[5] for _ in self.velocity])
            elif self.mode == 'ictiobot_3D':
                print('you pass 3d ictiobot')
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[1] for _ in self.velocity])
            else:
                vx = np.array([_[0] for _ in self.velocity])
                wz = np.array([_[1] for _ in self.velocity])

            # x centered
            vxm = vx - set_point[0]
            wzm = wz - set_point[1]

            ED = np.zeros(len(vx))
            for i in range(len(vx)):
                ED[i]= np.sqrt(vxm[i]**2 + wzm[i]**2)
        except IndexError:
            print('index error calculating euclidean_distance for only one distance variable')
            ED = np.zeros(len(self.velocity))
            for i in range(len(self.velocity)):
                ED[i] = np.sqrt(np.mean(( np.subtract(self.velocity[i],set_point[0]))**2))

        return np.mean(ED)
    

    def save_values(self):
        np.save('actions', self.actions)
        np.save('velocity', self.velocity)
        np.save('position', self.position)
        np.save('reward', self.reward)
        np.save('time', self.time)
        np.save('depth', self.depth)
        np.save('action_vx', self.action_vx)
        np.save('action_wz', self.action_wz)

        
    def reset(self):
        self.actions = deque()
        self.velocity = deque()
        self.position = deque()
        self.reward = deque()

    def load(self):
        self.actions = np.load('actions.npy')
        self.velocity = np.load('velocity.npy')
        self.position = np.load('position.npy' )
        self.reward = np.load('reward.npy')
        self.time = np.load('time.npy')
        self.depth = np.load('depth.npy')
        self.action_vx = np.load('action_vx.npy')
        self.action_wz = np.load('action_wz.npy')



if __name__ == '__main__':
    plotter = plotter('Velocities', 'u', 'positions', 'action_vx' , 'action_wz')
    plotter.load()
    print(plotter.euclidean_distance(np.array([0.21,-0.1])))
    print(plotter.mahalanobis(np.array([0.21,-0.1])))
    print(plotter.mean_squared_error(np.array([0.21,-0.1])))
    print(plotter.velocity)