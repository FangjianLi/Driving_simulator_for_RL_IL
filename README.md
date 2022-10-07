# Driving Simulator for reinforcement learning and imitation learning


This repository provide a driving simulator specifically for reinfrocement learning or imitation learning. The secenario is generated based on highway-env, where MDP model of the driving simulation environment is naturally embedded. The 3-D rendering of the driving simulator is based Carla. (Without the installtion of Carla, you can still run the diriving simulator on 2-D view). The you can control one car either with keyboard or steering wheel & pedals. Moreover, you can share the road together with your trained autonomous driving algorithm. 

<p>
    <img src="map_roadrunner.png" width="600" alt>
    <em>The customized map in Carla</em>
</p>

## What you can do with the codes
* As a driving simulator, it can collect your driving data used for imitation learning training purpose. States and actions in MDP are recorded.
* You can load your trained autonomous driving poicy into one neighbor car, and share the road with it. As a result, something interesting can be done. For example, you can play as a crazy driver and try to crash with the auonomous driving car to test its behavior in corner case. 


## How to run the codes
In order to use the 3-D animation feature, you need to install the Carla. I have pre-built a Carla repository with the customized map on it (via Roadrunner). You can download my customized version via the google drive [link]()

### Use the driving simulator to collect driving data 
* use the keyboard to control (arrow keys): ```python run_simulator_keyboard.py --carla False``` (2D) or ```python run_simulator_keyboard.py --carla True``` (3D)
* use the steering wheel and pedal to control: ```python run_simulator_pedal.py --carla False``` (2D) or ```python run_simulator_pedal.py --carla True``` (3D)

### Use the driving simultaor to play together with your trained autonomous vehicle
* use the keyboard to control (arrow keys): ```python run_simulator_keyboard_tp.py --carla False``` (2D) or ```python run_simulator_keyboard_tp.py --carla True``` (3D)
* use the steering wheel and pedal to control: ```python run_simulator_pedal_tp.py --carla False``` (2D) or ```python run_simulator_pedal_tp.py --carla True``` (3D)

### Tips
* You can adjust the size of simulation screen via the arg ```ratio```, for example, ```python run_simulator_keyboard.py --ratio 0.5```
* To test your own autonmous driving algorithm, you can integrate your own trained policy network in this simulator
* The Logitech G29 steering wheels and pedals are used for this repository. It should also work with other logitech steering wheels also

### Some screen captures

<p>
    <img src="vertical_view.gif" width="300"/>
    <em>2D simulator (green car is controlled by human with keyboard, and yellow car is controlled by trained driving policy)</em>
</p>

<p>
    <img src="carla_simulator.gif" width="600"/>
    <em>3D simulator (green car is controlled by human with keyboard, and yellow car is controlled by trained driving policy)</em>
</p>
