# Atalaya

This framework provides a logger for pytorch models, it allows you to save the parameters, the state of the network, the state of the optimizer and allows also to visualize your data using tensorboardX or visdom.

/!\ this readme file is WIP

* [Install](#install)
* [Examples](#Examples)
* [Usage](#usage)
    * [Init](#init)
    * [Log Information](#Log-Information)
    * [Store your Parameters](#Store-your-Parameters)
    * [Store and Restore (models and optimizers)](#Store-and-Restore-(models-and-optimizers))
    * [Grapher](#Grapher)

## Install

```bash
$ pip install atalaya
```

## Examples

Examples are provided in [examples](https://bitbucket.org/dmmlgeneva/frameworks/src/master/atalaya/examples/) directory, where we simply add the logger to an example of a pytorch implemetation ([source](https://github.com/pytorch/examples/blob/master/mnist/main.py)) in [example_1](https://bitbucket.org/dmmlgeneva/frameworks/src/master/atalaya/examples/example_1). In each [directory](https://bitbucket.org/dmmlgeneva/frameworks/src/master/atalaya/examples/) you have also the files created by the logger. There is a directory named logs and one named vizualize. The first one contains the logs of each experiment and the second one the files needed to visualize e.g. in tensorboard.

## Usage

### Init

```python
from atalaya import Logger

logger = Logger()

# by default Logger uses no grapher
# you can setup it by specifying if you want visdom or tensorboardX
logger = Logger(grapher='visdom')

    close(self)
        """Close the grapher."""




    save(self)
        """Saves the grapher."""
```

### Log Information

```python
    info(self, *argv)
        """Adds an info to the logging file."""
```

```python
    warning(self, *argv)
        """Adds a warning to the logging file."""
```

### Store your Parameters

```python
    add_parameters(self, params)
        """Adds parameters."""
```

```python
    restore_parameters(self, path)
        """Loads the parameters of a previous experience given by path"""
```

### Store and Restore (models and optimizers)

1. Add the model (or optimizer or whatever that has a state_dict in pytorch)

    ```python
        add(self, name, obj, overwrite=False)
            """Adds an object to the state (dictionary)."""
    ```

2. Store the model

    ```python
        store(self, loss, save_every=1, overwrite=True)
            """Checks if we have to store or if the current model is the best. 
            If it is the case save the best and return True."""
    ```

3. Restore the model

    ```python
        restore(self, folder=None, best=False)
            """Loads a state using torch.load()"""
    ```

### Grapher

```python
    add_scalar(self, tag, scalar_value, global_step=None, save_csv=True)
        """Adds a scalar to the grapher."""

    add_scalars(self, main_tag, tag_scalar_dict, global_step=None)
        """Adds scalars to the grapher."""

    export_scalars_to_json(self, path)
        """Exports scalars to json"""

    add_histogram(self, tag, values, global_step=None, bins='tensorflow')
        """Add histogram to summary."""

    add_image(self, tag, img_tensor, global_step=None, caption=None)
        """Add image data to summary."""

    add_figure(self, tag, figure, global_step=None, close=True)
        """Render matplotlib figure into an image and add it to summary."""

    add_video(self, tag, vid_tensor, global_step=None, fps=4)
        """Add video data to summary."""

    add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100)
        """Add audio data to summary."""

    add_text(self, tag, text_string, global_step=None)
        """Add text data to summary."""

    add_graph_onnx(self, prototxt)
        self.grapher.add_graph_onnx(prototxt)

    add_graph(self, model, input_to_model=None, verbose=False, **kwargs)
        """Adds a graph to the grapher."""

    add_embedding(self, mat, metadata=None, label_img=None,
                      global_step=None, tag='default', metadata_header=None)
        """Adds an embedding to the grapher."""

    add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None)
        """Adds precision recall curve."""

    add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall, 
                         global_step=None, num_thresholds=127, weights=None)
        """Adds precision recall curve with raw data."""

    register_plots(self, values, epoch, prefix, apply_mean=True, 
                       save_csv=True, info=True)
        """Helper to register a  dictionary with multiple list of scalars.
```
