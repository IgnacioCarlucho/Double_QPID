# Duble QPID

The algorithm is also known as DQPID. From the article:  

[**"Double Q-learning algorithm for mobile robot control"**](https://www.sciencedirect.com/science/article/pii/S0957417419304749)

As appears in Expert Systems with Applications.

Authors: Ignacio Carlucho - Mariano De Paula - Gerardo Acosta 

![alt text](https://github.com/IgnacioCarlucho/Double_QPID/blob/master/Fig2.jpg)

## Requirements: 

- Ros indigo
- numpy 
- python 2.7 

## The implemented robots are: 

- Pionner 3at
- Husky from clear path
- Ictiobot 
- drone (using hector_quadrotor)
- gym inverted pendulum 

## How to run: 

```
python main.py
```

In main.py there is a variable called platform. By assigning to this variable the available robots in the robot dictionary, the algorithm will run it accordingly, 
with the parameters configured in the dictionary.   
By default it is set in 'pioneer_pi' wich is configured for running the pioneer robot in a simulated environment using gazebo. 



## How to install needed simulators: 


### Pioneer: 

Into the catkin workspace clone 

```
git clone https://github.com/IgnacioCarlucho/amr-ros-config
```
then you can launch an example scenery as: 
```
roslaunch amr_robots_gazebo empty_world.launch 

```


Once the gazebo simulation is running you can then execute the algorithm by running: 

```
python main.py
```

This simulation is speed up for doing faster trials, they can be slowed down using gazebo. 


### Husky 


follow tutorial on: 
http://www.clearpathrobotics.com/assets/guides/husky/HuskyMove.html

and then launch with 
```
roslaunch husky_gazebo husky_empty_world.launch
```

set 
```
platform = 'husky_pi_random'
```
on main.py and then you can run the algorithm by doing:   

```
python main.py
```



### Quadrotor: 

Clone to your catkin workspace the hector quadrotor 

```
git clone https://github.com/tu-darmstadt-ros-pkg/hector_quadrotor

```
then you can launch an example: 
```
roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch 
```

### Gym 

https://gym.openai.com/docs/

### Ictiobot

Sorry this model is not available for the general domain. 



## References: 

- **Incremental Q-learning strategy for adaptive PID control of mobile robots** Carlucho et al. [Link](https://www.sciencedirect.com/science/article/pii/S0957417417301513)