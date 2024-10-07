# Summary
In this folder, we provide necessary codes for training UNREST uncertainty and decision models, and a tiny dataset subsampled from our whole dataset, for demonstration purposes. Note that complex testing codes have been omitted in this folder, as they require complex environments and a number of configuration files to set up the Carla environment. All codes will be public when published.

# Environment setup
The training dependencies are relatively simple, simply use:
```
pip install -r requirements.txt
```
for setting up the training environment.

# File Structure
1. config/: This folder includes config files for training return transformers (for uncertainty evaluation), segmenting offline trajectories, and training UNREST decision models.
2. data/: This folder includes the mini-ver. dataset subsampled from the whole Carla offline dataset.
3. dataset/: This folder contains the pytorch trajectory datasets for training UNREST models.
4. model/: This folder contains the base self-attention, GPT modules and return/decision transformers of UNREST.
5. train/: The folder contains the training scripts for return/decision transformers, and the trajectory segmentation code with measured uncertainty.
6. utils/: The utils files to facilitate training, such as the pytorch trainers and tools.

# Instruction Demos
1. First train two return transformers and save the corresponding checkpoints:
```
python -m train.train_return use_value=false # train with the latest action
python -m train.train_return use_value=true # train without the latest action
```
2. Second segment the trajectories and save the index file:
```
python -m train.segment ckpt_return={PATH_TO_RETURN_CKPT} ckpt_value={PATH_TO_VALUE_CKPT}
```
3. Finally train the decision model based on the segmented indice file:
```
python -m train.train_unrest indice_path={PATH_TO_INDEX_FILE}
```
The trained decision model can then be used for planning in long-horizon and uncertain Carla driving environments (omitted in this dem due to complexity).