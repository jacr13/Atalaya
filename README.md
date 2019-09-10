# Atalaya

This [framework](https://pypi.org/project/atalaya/) provides a logger for pytorch models, it allows you to save the parameters, the state of the network, the state of the optimizer and allows also to save and visualize your data using tensorboardX or visdom.

- [Install](#install)
- [Examples](#Examples)
- [Usage](#usage)
  - [Init](#init)
  - [Log Information](#Log-Information)
  - [Store your Parameters](#Store-your-Parameters)
  - [Store and Restore (models and optimizers)](<#Store-and-Restore-(models-and-optimizers)>)
  - [Grapher](#Grapher)

## Install

```bash
$ pip install atalaya
```

## Example

An example is provided [here](https://github.com/jacr13/Atalaya/tree/master/example).

Launch the example doing :

```bash
$ ./run.sh
```

An example of logs produced by the logger are given in the [logs](https://github.com/jacr13/Atalaya/tree/master/example/logs) folder of the example.

## Usage

### Init

```python
from atalaya import Logger

logger = Logger(
    name="exp",         # name of the logger
    path="logs",        # path to logs
    verbose=True,       # logger in verbose mode
)

# by default Logger uses no grapher
# you can setup it by specifying if you want visdom or tensorboardX
logger = Logger(
    name="exp",         # name of the logger
    path="logs",        # path to logs
    verbose=True,       # logger in verbose mode
    grapher="visdom",
    server="http://localhost",
    port=8097,
    username="user",    # if needed for visdom
    password="pwd",     # if needed for visdom
)

# your code here
...

# to close the logger
logger.close()
```

### Log Information

```python
# logs information in console and in log file.
logger.info("your", "information", "here", "like", "a", "print")

# same as logger.info but for warning messages
logger.warning("your warning")
```

### Store your Parameters

```python
# save your parameters into a json file
logger.add_parameters(args)

# load the parameters froma previous experiment
logger.restore_parameters(path_to_folder)
```

### Store and Restore (models and optimizers)

1. Add the model (or optimizer or whatever that has a state_dict in pytorch)

   Before starting storing or restoring objects you need to add them to the logger:

   ```python
       logger.add("model", model)
       logger.add("optimizer", optimizer)
   ```

2. Store the model

   In training loop we can add this method, it allows us to save checkpoints,
   and the best model.

   - The parameter valid_loss is simply the parameter to know when to save the best. It looks if
     the new valid_loss is less than the value keep by the logger as lower if it's the case save as best and
     update the value keep in memory.
   - The parameter save_every specify how often to save a checkpoint during training.
   - overwrite specify if we want to overwrite the last checkpoint to keep only the last one
     or if we want to keep them all (saves a model per epoch --> DANGEROUS)

   ```python
       logger.store(valid_loss, save_every=1, overwrite=True)

   ```

3. Restore the model
   To restore the best after taining simply do

   ```python
       logger.restore(best=True)
   ```

   To restore a checkpoint from another exp :

   ```python
       logger.restore(folder=path_to_folder)
   ```

   To restore the best from another exp :

   ```python
       logger.restore(folder=path_to_folder, best=True)
   ```

### Grapher

Some examples of grapher methods.

```python
logger.add_scalar("train_mse", scalar_value, global_step=None, save_csv=True)

logger.add_text("tag", "your text here")

# values for each batch size at a given epoch
values = {
    "mse": [10, 9, 8, 7],
    "acc": [0.3, 0.5, 0.55, 0.6]
}
logger.register_plots(values, epoch, "train", apply_mean=True, save_csv=True, info=True)
```
