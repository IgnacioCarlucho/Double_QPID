import numpy as np
# custom files
from robots import pioneer_pi
from gym_robot import gym_pendulum
from husky import husky_pi
from drone import drone_pi



robot_dict = {

    'GYM_original' : {'class': gym_pendulum, 'set_point': np.zeros(1), 'action_centroid': np.array([35., 35.]), 'initial_state': np.array([[0.]]),'K_step': 5.,'comentarios': 'para el paper'},
    
    'GYM' : {'class': gym_pendulum, 'set_point': np.zeros(1), 'action_centroid': np.array([50., 50.]), 'initial_state': np.array([[0.]]),'K_step': 5.,'comentarios': 'para el paper'},
    
    'pioneer_pi' : {'class': pioneer_pi, 'set_point': np.array([0.21,0.21]), 'action_centroid': np.array([0.5, 0.51, 0.5, 0.51]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'para el paper'},

    'pioneer_pi_random' : {'class': pioneer_pi, 'set_point': np.array([0.21, 0.11]), 'action_centroid': np.array([np.maximum(2.*np.random.rand(),0.31), np.maximum(2.*np.random.rand(),0.31),np.maximum(2.*np.random.rand(),0.31),np.maximum(2.*np.random.rand(),0.31)]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'para el paper'},

    'pioneer_pi_random_high' : {'class': pioneer_pi, 'set_point': np.array([0.21, 0.11]), 'action_centroid': np.array([np.maximum(20.*np.random.rand(),7.), np.maximum(20.*np.random.rand(),7.), np.maximum(20.*np.random.rand(),7.), np.maximum(20.*np.random.rand(),7.)]), 'initial_state': np.array([[0., 0.]]),'K_step': 7., 'comentarios': 'para el paper'},
    
    'husky_pi_random' : {'class': husky_pi, 'set_point': np.array([0.31, -0.19]), 'action_centroid': np.array([np.maximum(2.*np.random.rand(),0.31), np.maximum(2.*np.random.rand(),0.31),np.maximum(2.*np.random.rand(),0.31),np.maximum(2.*np.random.rand(),0.31)]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch husky_gazebo husky_playpen.launch'},

    'husky_pi' : {'class': husky_pi, 'set_point': np.array([0.31,-0.19]), 'action_centroid': np.array([0.5, 0.51, 0.5, 0.51]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch husky_gazebo husky_playpen.launch'},

    'drone_pi' : {'class': drone_pi, 'set_point': np.array([0.21,0.21]), 'action_centroid': np.array([0.5, 0.51, 0.5, 0.51]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch '},

    'drone_2' : {'class': drone_pi, 'set_point': np.array([0.21,0.21]), 'action_centroid': np.array([0.5, 0.51, 0.5, 0.51]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch '},
    
    }


