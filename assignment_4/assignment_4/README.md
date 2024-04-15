# CS780 - Assignment 4

### Points to remember
- All code changes (not hyperparameter changes) are to be made in `assignment4.ipynb`. Then use `jupytext --to py assignment4.ipynb` to convert the notebook to python file

### Steps for doing hyperparameter tuning
1. Check for all the tested hyperparameters in `hyperparameters_ddpg_td3.csv` and their images in `images/<env_name>/<experiment_version>`
2. Make a copy of `assignment4.py` which you created from the "points to remember" to create another python file namely `assignment4_<env_name>_<experiment_version>.py`
3. Do the changes of hyperparameters in `assignment4_<env_name>_<version>.py` only, and run it.
