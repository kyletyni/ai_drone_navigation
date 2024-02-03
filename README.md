# ai_drone_navigation

This project contains code that was used to train a neural network to output velocity setpoints
in order for a drone to achieve a target position. Training was done in the Gazebo robotics
simulator utilizing stabebaselines3, and the drone is the Coex Clover.

Screenshot of Gazebo simulator:

GIF of drone navigating to position (0, 0, 5):



Here is a video of the drone interfacing with VICON camera stream in real time, the drone fuses
sensor data it receives from a live UDP stream with the onboard gyroscope.


Image of pathfinding algorithm at work, creating points along a path for the drone to navigate
while avoiding known obstacle locations.