import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import tf.transformations
from plotter import plotter
#TODO nothing is stored here yet. So no plots will be abailable to the user. 


ALPHA = 0.1

class pioneer():

    def __init__(self, set_point, dt = 0.1, Teval = 1., simulation = True):


        self.position = np.zeros(3)
        self.vel_v = np.zeros(3)
        self.vel_w = np.zeros(3)
        self.velocity = np.zeros(2)
        self.euler = np.zeros(3)
        self.Quater = np.zeros(4)
        self.dt = dt
        self.Teval = Teval
        self.error = np.zeros((3,2))
        self.u0 = np.zeros(2)
        self.u = np.zeros(2)
        self.set_point = set_point
        self.node = rospy.init_node('DQPID', anonymous=False)
        if simulation: 
            self.Publisher =  rospy.Publisher("/sim_p3at/cmd_vel", Twist, queue_size=1)
            self.Subscriber = rospy.Subscriber("/sim_p3at/odom", Odometry, self.callback_pose, queue_size=1)
        else: 
            self.Publisher =  rospy.Publisher("/RosAria/cmd_vel", Twist, queue_size=1)
            self.Subscriber = rospy.Subscriber("/RosAria/pose", Odometry, self.callback_pose, queue_size=1)    
        self.rate = rospy.Rate(10.) # 10hz
        self.msg = Twist()
        self.action_vx = np.zeros(3)
        self.action_wz = np.zeros(3)
        self.reward = -1.
        self.execution = np.divide(self.Teval,self.dt).astype(int)
        self.temporal_state = np.zeros((2,self.execution))
        # to plot
        self.plotter = plotter('Velocities', 'u', 'positions', 'action_vx' , 'action_wz')
        self.time = 0. 

    def update(self, action, depth):

        for _ in range(self.execution):
            self.action_vx = action[0:3]
            self.action_wz = action[3:6]
            #I save all the velocities during execution, I can use it later to calculate the reward differently
            #TODO
            #self.temporal_state[_] = self.velocity 
            # update errors        
            self.error[2] = self.error[1]
            self.error[1] = self.error[0]  
            self.error[0][0] = self.set_point[0] - self.velocity[0]
            self.error[0][1] = self.set_point[1] - self.velocity[1]
            # get controller commands
            self.u[0] = self.controller_pid(self.error[0][0], self.error[1][0], self.error[2][0], self.action_vx, self.u0[0]) 
            self.u[1] = self.controller_pid(self.error[0][1], self.error[1][1], self.error[2][1], self.action_wz, self.u0[1]) 
            self.u = np.clip(self.u, -0.8, 0.8)
            self.u0 = self.u
            self.msg.linear.x = self.u[0]
            self.msg.angular.z = self.u[1]
            self.Publisher.publish(self.msg)
            # to plot
            self.time = self.time + self.dt
            self.plotter.update(self.velocity, self.u, self.position, self.time, depth, self.action_vx, self.action_wz)
            self.rate.sleep()
        
        return self.velocity


    def controller_pid(self, et, et1, et2, action, u0):
        Kp = action[0]
        Ti = action[1]
        Td = action[2]

        k1 = Kp*(1+Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt-self.dt/Ti)
        k3 = Kp*(Td/Ti)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def get_gaussian_reward(self, state, set_point):

        a_gauss = np.power(0.035,2.) #0.017
        exponent = np.zeros(len(set_point))
        for _ in range(len(set_point)):
            exponent[_] = np.power((state[_] - set_point[_]), 2.)
            
        exponent_total = np.sum(exponent)
        self.reward = -1. + 2*np.exp(-0.5*(exponent_total/a_gauss))
    
        # save reward to plot it
        self.plotter.update_reward(self.reward)
        

        return self.reward



    def wrapToPi(self, angles):
        
        if angles > np.pi:
            angles = angles - 2*np.pi
        elif angles < -np.pi:
            angles = angles + 2*np.pi
        return angles 





    def callback_pose(self, msg_odometry):
        global position, vel_v, vel_w, Quater, euler, euler_original
        x = msg_odometry.pose.pose.position.x
        y = msg_odometry.pose.pose.position.y
        z = msg_odometry.pose.pose.position.z
        self.position = np.array([x, y, z])

        vx = msg_odometry.twist.twist.linear.x
        vy = msg_odometry.twist.twist.linear.y
        vz = msg_odometry.twist.twist.linear.z
        self.vel_v = np.array([vx, vy, vz])
        wx = msg_odometry.twist.twist.angular.x
        wy = msg_odometry.twist.twist.angular.y
        wz = msg_odometry.twist.twist.angular.z
        self.vel_w = np.array([wx, wy, wz])

        Qx = msg_odometry.pose.pose.orientation.x
        Qy = msg_odometry.pose.pose.orientation.y
        Qz = msg_odometry.pose.pose.orientation.z
        Qw = msg_odometry.pose.pose.orientation.w
        #z y x representation
        #Quater=[Qz,Qy,Qx,Qw];
        #Quater = np.array([Qw,Qx,Qy,Qz]) # este es el que eestaba usando
        self.Quater = np.array([Qx,Qy,Qz, Qw])
        #z y x representation of quaternions
        euler_original = tf.transformations.euler_from_quaternion(self.Quater) #[rad]
        
        self.euler = [ self.wrapToPi(_) for _ in euler_original] 
        
        self.velocity[0] = 0.9*self.velocity[0] + 0.1*vx
        self.velocity[1] = 0.9*self.velocity[1] + 0.1*wz

    def stop(self):
        self.msg.linear.x = 0.
        self.msg.angular.z = 0.
        self.Publisher.publish(self.msg)



class pioneer_pi():

    def __init__(self, set_point, dt = 0.1, Teval = 1., simulation = True):


        self.position = np.zeros(3)
        self.vel_v = np.zeros(3)
        self.vel_w = np.zeros(3)
        self.velocity = np.zeros(2)
        self.euler = np.zeros(3)
        self.Quater = np.zeros(4)
        self.dt = dt
        self.Teval = Teval
        self.error = np.zeros((3,2))
        self.u0 = np.zeros(2)
        self.u = np.zeros(2)
        self.set_point = set_point
        self.node = rospy.init_node('DQPID', anonymous=False)
        if simulation: 
            self.Publisher =  rospy.Publisher("/sim_p3at/cmd_vel", Twist, queue_size=1)
            self.Subscriber = rospy.Subscriber("/sim_p3at/odom", Odometry, self.callback_pose, queue_size=1)
        else: 
            self.Publisher =  rospy.Publisher("/RosAria/cmd_vel", Twist, queue_size=1)
            self.Subscriber = rospy.Subscriber("/RosAria/pose", Odometry, self.callback_pose, queue_size=1)    

        self.rate = rospy.Rate(10.) # 10hz
        self.msg = Twist()
        self.action_vx = np.zeros(2)
        self.action_wz = np.zeros(2)
        self.reward = -1.
        self.execution = np.divide(self.Teval,self.dt).astype(int)
        self.temporal_vx = np.zeros(self.execution)
        self.temporal_wz = np.zeros(self.execution)
        # to plot
        self.plotter = plotter('Velocities', 'u', 'positions', 'action_vx' , 'action_wz')
        self.time = 0.        


    def update(self, action, depth):

        for _ in range(self.execution):
            self.action_vx = action[0:2]
            self.action_wz = action[2:4]
            #I save all the velocities during execution, I can use it later to calculate the reward differently
            #TODO
            self.temporal_vx = self.velocity[0]
            self.temporal_wz = self.velocity[1]
            # update errors        
            self.error[2] = self.error[1]
            self.error[1] = self.error[0] 
            self.error[0][0] = self.set_point[0] - self.velocity[0]
            self.error[0][1] = self.set_point[1] - self.velocity[1]
            # get controller commands
            self.u[0] = self.controller_pid(self.error[0][0], self.error[1][0], self.error[2][0], self.action_vx, self.u0[0]) 
            self.u[1] = self.controller_pid(self.error[0][1], self.error[1][1], self.error[2][1], self.action_wz, self.u0[1]) 
            self.u = np.clip(self.u, -0.8, 0.8)
            self.u0 = self.u
            # to publish
            self.msg.linear.x = self.u[0]
            self.msg.angular.z = self.u[1]
            self.Publisher.publish(self.msg)
            # to plot
            self.time = self.time + self.dt
            self.plotter.update(self.velocity, self.u, self.position, self.time, depth, self.action_vx, self.action_wz)
            # to keep sampling rate
            self.rate.sleep()

        #print('temporal_state', np.mean(self.temporal_vx),  np.mean(self.temporal_wz), 'vel', self.velocity )
        #mean_state = np.array([np.mean(self.temporal_vx), np.mean(self.temporal_wz)])

        return self.velocity


    def controller_pid(self, et, et1, et2, action, u0):
        
        Kp = action[0]
        Ti = action[1]
        Td = 0.

        k1 = Kp*(1+Td/self.dt)
        k2 =-Kp*(1+2*Td/self.dt-self.dt/Ti)
        k3 = Kp*(Td/Ti)

        u = u0 + k1*et + k2*et1 + k3*et2

        return u

    def get_gaussian_reward(self, state, set_point):

        a_gauss = np.power(0.035,2.) #0.017
        #a_gauss = np.power(0.017,2.) #0.017
        exponent = np.zeros(len(set_point))
        for _ in range(len(set_point)):
            exponent[_] = np.power((state[_] - set_point[_]), 2.)
            
        exponent_total = np.sum(exponent)
        self.reward = -1. + 2*np.exp(-0.5*(exponent_total/a_gauss))
        # save reward to plot it
        self.plotter.update_reward(self.reward)

        return self.reward




    def wrapToPi(self, angles):
        
        if angles > np.pi:
            angles = angles - 2*np.pi
        elif angles < -np.pi:
            angles = angles + 2*np.pi
        return angles 


    def callback_pose(self, msg_odometry):
        x = msg_odometry.pose.pose.position.x
        y = msg_odometry.pose.pose.position.y
        z = msg_odometry.pose.pose.position.z
        self.position = np.array([x, y, z])

        vx = msg_odometry.twist.twist.linear.x
        vy = msg_odometry.twist.twist.linear.y
        vz = msg_odometry.twist.twist.linear.z
        self.vel_v = np.array([vx, vy, vz])
        wx = msg_odometry.twist.twist.angular.x
        wy = msg_odometry.twist.twist.angular.y
        wz = msg_odometry.twist.twist.angular.z
        self.vel_w = np.array([wx, wy, wz])

        Qx = msg_odometry.pose.pose.orientation.x
        Qy = msg_odometry.pose.pose.orientation.y
        Qz = msg_odometry.pose.pose.orientation.z
        Qw = msg_odometry.pose.pose.orientation.w
        #z y x representation
        #Quater=[Qz,Qy,Qx,Qw];
        #Quater = np.array([Qw,Qx,Qy,Qz]) # este es el que eestaba usando
        self.Quater = np.array([Qx,Qy,Qz, Qw])
        #z y x representation of quaternions
        euler_original = tf.transformations.euler_from_quaternion(self.Quater) #[rad]
        
        self.euler = [ self.wrapToPi(_) for _ in euler_original] 
        
        #self.velocity = np.array([vx, wz])
        a = 0.9*self.velocity[0] + 0.1*vx
        b = 0.9*self.velocity[1] + 0.1*wz
        self.velocity = np.array([a, b])

    def stop(self):
        self.msg.linear.x = 0.
        self.msg.angular.z = 0.
        self.Publisher.publish(self.msg)