# Duble QPID

also known as DQPID

This branch does velocity control of the mobile.

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

In main.py there is a variable called platform. By assigning to this variable the available robots in the robot dictionary, the algorithm will run it accordingly, with the parameters configured in the dictionary. 



## How to install needed simulators: 


### Pioneer: 

Into the catkin workspace clone 

```
git clone https://github.com/MobileRobots/amr-ros-config
```
then you can launch an example scenery as: 
```
roslaunch amr_robots_gazebo example-pioneer3at-terrainworld.launch
```
Source: https://github.com/MobileRobots/amr-ros-config

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
on main.py



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


## Operating on the pioneer3at platform: 

In your coputer, connect to pioneer: 
```
ssh pioneer@192.168.0.27
```
then launch everything: 

```
cd bin
./enPioneer_Pausado

```
Wait. and then you can check that everything is working by running some ros command, ie:

```
rostopic list
``` 

then open another terminal in your computer, and export the ROS variables, to connect with the master running on pioneer

```
export ROS_MASTER_URI=http://192.168.0.27:11311
export ROS_IP=your_computer_ip
```

After that you are ready to run the algorithm

```
python main.py
```

if during operations you press the button, you can reactive motors by using
```
rosservice set /RosAria/enable_motors
```
