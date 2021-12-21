# Adaptive-Systems-2021 Double Deep Q-Learning in OpenAI-gym

### Installation of packages
To install the package run the following command: 
```bash 
pip install .
```

If you are a developer you need to run this command:
```bash 
pip install .[dev]
```

# Lunar Lander

to run the script run the following file

```bash
gym/lunar-lander/main.py
```
![Alt Text](https://github.com/RichardDev01/Gym-Double-Deep-Q-learning/blob/main/gym/lunar-lander/worked_out_model_ll_dqn.gif?raw=true)

### reflectie

Onze agents lijkt het goed te doen. Hij landt 100% van de tijd op de grond met 2 pootjes. Als hij met 1 pootje net naast de vlaggetjes land probeert hij te corrigeren door de juiste booster aan te zetten. Dit is echter alleen een specifiek probleem wat wel op te lossen is met een creatieve werkwijzen. We zouden het model extra kunnen trainen met human exempels om uit deze scenario te komen maar dat is voor nu te lastig om goed te implementeren.

Hieronder is een grafiek met wat het verschil is tussen leren met epsilon decay vergeleken met een vaste epsilon.
Bij epsilon decay is hij sneller bij de zo goed mogelijke reward van het netwerk maar uiteindelijk komen ze wel bij dezelfde reward.

![Alt Text](https://github.com/RichardDev01/Gym-Double-Deep-Q-learning/blob/main/gym/lunar-lander/graph.png?raw=true)

# Tensorboard
To show the tensorboard
```
tensorboard --logdir gym/lunar-lander/runs

```