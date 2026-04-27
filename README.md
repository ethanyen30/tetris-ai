# tetris-ai

## Introduction

Project originally between Caleb Chang and Ethan Yen October - December 2024

Initial Progress:
- We use a DQN model trained on gtx 1070 to play tetris.
- Models showed improvement by lasting longer in game.

- Project is on hold, but we are almost done with hole detection to see if scores will improve

- I believe on hold because of winter break and no time to work on it.

Ethan continued by himself (and made new repo) from December 2024 - January 2025:

## Installs
If you don't have a virtual environment, create one with ```python -m venv .venv```, because this is what's in the gitignore file

Activate with ```.venv\Scripts\activate```

Then install the requirements with ```pip install -r requirements.txt```

## Runs
First, to run everything, I already have the trained models in the checkpoint.pth files. Each was a checkpoint in the training process so checkpoint4 is the latest/best one. Install all the requirements. Run the current config in the test.py file:
```python test.py```

If you want explanation of the process, I'd be happy to explain.

### Info:
- The model is initialized with the agent.py file which contains an TetrisAgent class. Here is the SDK documentation:
    - init(): initializes a default neural network for the agent
    - train(): trains the dqn model by playing tetris and getting reinforcement learning rewards
    - save_checkpoint(): saves the internal dqn model to a checkpoint.pth file to load up later
    - load_checkpoint(): loads an internal dqn model from a checkpoint.pth file to use for testing or further training
    - run(): runs the model on the local tetris game
    - play(): runs the model on the online tetris game

- The model was trained with help from this tutorial: https://max-we.github.io/Tetris-Gymnasium/
- I added a selenium wrapper so the model can play on this website: https://jstris.jezevec10.com/?play=1&mode=2
