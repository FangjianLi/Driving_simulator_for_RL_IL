# highway_env


This repository includes the simulator codes with human-in-the-loop. The simulators are built on the highway-env and Carla. In total, there are 4 different simulator scenarios. In addition, the tutorials on the highway-env and Carla are also attached as ```highway-env tutorial.ipynb``` and ```carla_tutorial.ipynb```, respectively. The respository is already in titan computer, ```~/Fangjian/highway_env```.

## How to run the codes

### Tip: how to turn on the feeback force of the steering wheel
    * In an separate terminal
        * Activate the environment: ```conda activate FL_jubo_1```
        * Run the setup codes: ```python test_joystick.py```
        * It is okay to close the terminal now. 


### Scenario 1: One human driver with highway-env (discrete action space)
* **Description:** This is the most basic scenario. The human will control a vehicle in the highway-env. Discrete action space is used here. 
* **Run the codes**:
    * Without Carla rendering:
        * activate the environment: ```conda activate FL_jubo_1```
        * run the simulator: ```python run_animation_manual_vertical.py```
        * control the vehicle with arrow keys. 
    * With Carla rendering:
        * In terminal-1: 
            * come to the carla foler in titan computer ```~/Fangjian/Carla_1```, and run ```./CarlaUE4.sh```
        * In terminal-2:
            * activate the environment: ```conda activate FL_jubo_1```
            * run the simulator: ```python run_animation_manual_vertical_carla.py```
            * control the vehicle with arrow keys. 
* **Output**: saved observation ```observation.npy``` and action ```action.npy``` data  in ```trajectory/manual_driving_vertical/``` (without Carla rendering) or ```trajectory/manual_driving_vertical_carla/``` (with Carla rendering)
* **Time takes**: Real time simulation, depends on the hyperparamter ```min_length```, i.e., how many steps in total. 1 step takes 0.5 second. 


### Scenario 2: One human driver with highway-env (continuous action space)
* **Description:** This is the most basic scenario. The human will control a vehicle in the highway-env. Continuous action space is used here. 
* **Run the codes**:
    * Without Carla rendering:
        * connect the driving simulator, i.e., pedal and steering wheel,  to the computer
        * activate the environment: ```conda activate FL_jubo_1```
        * run the simulator: ```python run_animation_manual_continuous.py```
        * control the vehicle with pedal and steering wheel. 
    * With Carla rendering:
        * connect the driving simulator, i.e., pedal and steering wheel,  to the computer
        * In terminal-1: 
            * come to the carla foler in titan computer ```~/Fangjian/Carla_1```, and run ```./CarlaUE4.sh```
        * In terminal-2:
            * activate the environment: ```conda activate FL_jubo_1```
            * run the simulator: ```python run_animation_manual_continuous_carla.py```
            * control the vehicle with pedal and steering wheel. 
* **Output**: saved observation ```observation.npy``` and action ```action.npy``` data  in ``trajectory/manual_driving_continuous/``` (without Carla rendering) or ```trajectory/manual_driving_continuous_carla/``` (with Carla rendering)
* **Time takes**: Real time simulation, depends on the hyperparamter ```min_length```, i.e., how many steps in total. 1 step takes 0.5 second. 

### Scenario 3: One human driver, One autonomous car, with highway-env (discrete action space)
* **Description:**  The human will control a vehicle in the highway-env. There is one more vehicle controlled by the autonomous driving algorithm (trained driving policy). Discrete action space is used here. 
* **Run the codes**:
    * Without Carla rendering:
        * activate the environment: ```conda activate FL_jubo_1```
        * run the simulator: ```python run_animation_manual_vertical_tp.py```
        * control the vehicle with arrow keys. 
    * With Carla rendering:
        * In terminal-1: 
            * come to the carla foler in titan computer ```~/Fangjian/Carla_1```, and run ```./CarlaUE4.sh```
        * In terminal-2:
            * activate the environment: ```conda activate FL_jubo_1```
            * run the simulator: ```python run_animation_manual_vertical_tp_carla.py```
            * control the vehicle with arrow keys.
* **Output**: saved observation ```observation.npy``` and action of the vehicle controlled by human  ```action_neighbor.npy```, and action of the vehicle controlled by trained polict ```action_subject.npy```  in ```trajectory/manual_driving_vertical_tp/``` (without Carla rendering) or ```trajectory/manual_driving_vertical_tp_carla``` (with Carla rendering)
* **Time takes**: Real time simulation, depends on the hyperparamter ```min_length```, i.e., how many steps in total. 1 step takes 0.5 second. 

### Scenario 4: One human driver, One autonomous car, with highway-env (continuous action space)
* **Description:** The human will control a vehicle in the highway-env. There is one more vehicle controlled by the autonomous driving algorithm (trained driving policy).  Continuous action space is used here. 
* **Run the codes**:
    * Without Carla rendering:
        * connect the driving simulator, i.e., pedal and steering wheel,  to the computer
        * activate the environment: ```conda activate FL_jubo_1```
        * run the simulator: ```python run_animation_manual_continuous_tp.py```
        * control the vehicle with pedal and steering wheel. 
    * With Carla rendering:
        * In terminal-1: 
            * come to the carla foler in titan computer ```~/Fangjian/Carla_1```, and run ```./CarlaUE4.sh```
        * In terminal-2:
            * activate the environment: ```conda activate FL_jubo_1```
            * run the simulator: ```python run_animation_manual_continuous_tp_carla.py```
            * control the vehicle with pedal and steering wheel. 
* **Output**: saved observation ```observation.npy``` and action of the vehicle controlled by human ```action_neighbor.npy``` data, and action of the vehicle controlled by trained policy ```action_subject```  in ``trajectory/manual_driving_continuous_tp/``` (without Carla rendering) or ```trajectory/manual_driving_continuous_tp_carla/``` (with CArla rendering)
* **Time takes**: Real time simulation, depends on the hyperparamter ```min_length```, i.e., how many steps in total. 1 step takes 0.5 second. 





