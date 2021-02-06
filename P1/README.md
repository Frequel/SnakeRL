## Introduction
The goal of this project is to develop an AI Bot able to learn how to play the popular game Snake from scratch. 
The project will show the difference between Monte Carlo, Q-Learning e SARSA(lambda) methods.

## Folder structure
In this folder there is all the material used in the creation of this part of the project. From the image used in the Jupyter notebook report to the python files used as base to develop the project.
The folder Entrega_Snake contains all the main files to understand and run all the project in Jupyter notebook.


## Dependencies
This project requires Python 3.6 with the pygame library installed, as well as Pytorch. \
The full list of requirements is in `requirements.txt`. 


## Run
To run and show the game, executes in the snake-ga folder:

```python
python main.py --display=True --speed=50
```
Arguments description:

- --display - Type bool, default True, display or not game view
- --speed - Type integer, default 50, game speed

The default configuration loads the file *weights/weights.hdf5* and runs a test.
The parameters of the Deep neural network can be changed in *snakeClass.py* by modifying the dictionary `params` in the function `define_parameters()`

To train the agent, set in the file snakeClass.py:
- params['load_weights'] = False
- params['train'] = True

In snakeClass.py you can set argument *--display*=False and *--speed*=0, if you do not want to see the game running. This speeds up the training phase.

## For Mac users
It seems there is a OSX specific problem, since many users cannot see the game running.
To fix this problem, in update_screen(), add this line.

```                              
def update_screen():
    pygame.display.update() <br>
    pygame.event.get() # <--- Add this line ###
```
