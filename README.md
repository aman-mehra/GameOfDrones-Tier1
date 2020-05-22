# GameOfDrones-Tier1
Reinforcement Learning Agent for Game of Drones challenge (Neurips 2019)

[A demo of a partially trained experimental run](https://drive.google.com/file/d/1vn4h3XdiQedJbgtqQw70FhF7_zTgfd0K/view)

# Method

## Architecture
Our method employs the Twin Delayed DDPG algorithm for continuous action spaces. The actor architecture is comprises an LSTM followd by an MLP. The two Q function estimators are 3 layered MLPs.

![alt text](https://github.com/amehra-github/GameOfDrones-Tier1/blob/master/Architectures.png)

## Action Space
The action space is a 4 dimensional vector comprising of a 3 dimensional unit velocity vector estimate and the velocity magnitude.

## State Space
The state space is a 52 dimensional vector composed of the drone's pose, velocity and accleration, coupled with the estimated relative position of the next gate, the three nearest obstacles and the adversary drone.

## Reward
The reward system is designed to make the agent finish the course in a minimum amount of time, while ensuring minimal collisions and emphasizing the need to stay ahead of the adversary.
We allocate a time penalty to minimize course completion time. In addition to this there is a penalty for collision with other obstacles and a much larger penalty for initiating collision with an adversary (since this disqualifies the aggressor). A bonus reward is awarded at the time of completion as well as when the drone passes a checkpoint gate. Again this is to emphasise course completion.
To incorporate blocking the path of an adversary, we reward the agent if it successfully manages to make the adversary an aggressor in a collision. To facilitate overtaking the time penalty is scaled using factor depending on the position the agent occupies relative to the adversary. A harsher time penalt is employed when the agent lags behind the adversary.

## Navigation Framework
Our navigation framework generates a base velocity vector from the trajectory obtained from the current pose estimate and the subsequent waypoint. The TD3 network learns a policy for the velocity and perturbation vector which is used to agment the base velocity vector generated from the trajectory. Actions are generated periodically at an interal of 50 ms. Our approach ensures that the drone realizes the shortest path (straight line) in the absence of adverary or obstacle, while suitably altering direction and speed when necessary to avoid obstacles or ivertake adveraries.

# Results
We obtained a lap time of 68 secs on competition Tier 1 which corresponds to a leaderboard rank of 8.

[Best LAp](https://github.com/amehra-github/GameOfDrones-Tier1/blob/master/Best%20Run.jpg)

# Code
[two_racer.py](two_racer.py) - This contains the environment intervace and the trajectory planner and updater.

[lstm-racer.py](lstm-racer.py) - This contains the TD3 implementation.

