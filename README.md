# ai_drone_navigation

This project contains code that was used to train a neural network to output velocity setpoints
in order for a drone to achieve a target position. Training was done in the Gazebo robotics
simulator utilizing stabebaselines3, and the drone is the Coex Clover.


Screenshot of Gazebo simulator:
<div style="display:flex; justify-content:space-between;">
  <img src="https://github.com/kyletyni/ai_drone_navigation/blob/main/images/gazebo.png" width="500">
</div>



Video of drone navigating to set position from random spawns:
<a href="https://youtu.be/EwoKA0WvnwU" target="_blank">Video Link</a>
![drone demo](https://github.com/kyletyni/ai_drone_navigation/blob/main/images/drone.gif)



Here is a video of the drone interfacing with VICON camera stream in real time, the drone fuses
sensor data it receives from a live UDP stream with the onboard gyroscope.
<a href="https://youtu.be/_Cnf-fW7EQI" target="_blank">Video Link</a>
[![vicon demo](https://img.youtube.com/vi/_Cnf-fW7EQI/0.jpg)](https://youtu.be/_Cnf-fW7EQI)




Image of pathfinding algorithm at work, creating points along a path for the drone to navigate
while avoiding known obstacle locations.
<div style="display:flex; justify-content:space-between;">
  <img src="https://github.com/kyletyni/ai_drone_navigation/blob/main/images/finder.png" width="500">
</div>