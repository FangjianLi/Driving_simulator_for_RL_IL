# Driving Simulator for reinforcement learning and imitation learning


This repository provide a driving simulator specifically for reinfrocement learning or imitation learning. The secenario is generated based on highway-env, where MDP model of the driving simulation environment is naturally embedded. The 3-D rendering of the driving simulator is based Carla. (Without the installtion of Carla, you can still run the diriving simulator on 2-D view). The you can control one car either with keyboard or steering wheel & pedals. Moreover, you can share the road together with your trained autonomous driving algorithm. 

## What you can do with the codes
* As a driving simulator, it can collect your driving data used for imitation learning training purpose. States and actions in MDP are recorded.
* You can load your trained autonomous driving poicy into one neighbor car and share the road with it. As a result, something interesting can be done. For example, you can play as a crazy driver and try to crash with the auonomous driving car to test its behavior in corner case. 


