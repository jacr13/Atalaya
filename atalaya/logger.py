# WIP

import sys
import os
from os.path import join as pjoin
import logging
import warnings
import json
from datetime import datetime

import numpy as np
import torch

from  atalaya.grapher.grapher import NoGrapher, Grapher

class _Params():
    """Class that loads hyperparameters from a json file.

    From :
    - https://cs230-stanford.github.io/logging-hyperparams.html
    - https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
    
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, params=None, path=None):
        if params is not None:
            self.__dict__.update(params)
        elif path is not None:
            self.update(path)
        else:
            raise Exception('params and path at None ! One of them must be not None.')

    def save(self, path):
        """Saves parameters to a json file"""
        with open(pjoin(path, 'params.json'), 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, path):
        """Loads parameters from json file"""
        with open(pjoin(path, 'params.json')) as f:
            params = json.load(f)
            params[list(self.__dict__.keys())[list(self.__dict__.values()).index(path)]] = path
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__



class Logger:
    def __init__(self, name='exp', path='./logs', add_time=True, verbose=True,
                 grapher='visdom', server='http://localhost', port=8097,
                 username=None, password=None):
        """Logs models, optimizers, and all you want. Creates checkpoints at 
        a given frequency and save directly the best model.
        Can also be used as a grapher to visualize graphs or images.
        """
        if add_time:
            name = '{}_{}'.format(name, 
                                  datetime.now().strftime('%Y%m%d_%H%M%S')) 
        self.name = name 
        
        self.path = path
        if not self.path:
            warnings.warn('You don\'t have specify any path for the logger, '
                           'nothing will be saved and it will be impossible '
                           'to restore any model (even the best).',
                           Warning)
        
        self.logs_dir = ''
        if self.path:
            self.logs_dir = pjoin(path, self.name)

        self.checkpoints_dir = pjoin(self.logs_dir, 'checkpoints')
        self._logs_file = pjoin(self.logs_dir, '.actions_of_logger.txt')
        self.grapher_data_folder = pjoin(self.logs_dir, 'grapher_data')
        
        self.params = None
        self.state = dict()

        if self.path:
            self._makedirs()
        self._epoch = 0
        self.add('epoch', self._epoch)
        self._epoch_prints = True
        self._loss = sys.maxsize


        # set up logging to file
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s:%(levelname)s: %(message)s',
                            filename=pjoin(self.logs_dir, 'train.log'),
                            filemode='w')

        if verbose:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)

        self._logger = logging.getLogger('atalaya')
        if self.path:
            for handler in self._logger.handlers[:]:
                print(handler)
                self._logger.removeHandler(handler)

        # grapher
        if not grapher:
            self.grapher = NoGrapher()
        elif grapher == 'visdom':
            logging.getLogger('').setLevel(logging.ERROR)
            self.grapher = Grapher('visdom', env=self.name,
                                    server=server,
                                    port=port,
                                    username=username,
                                    password=password,
                                    use_incoming_socket=False)
            logging.getLogger('').setLevel(logging.INFO)
        elif grapher == 'tensorboard':
            self.grapher = Grapher('tensorboard', 
                                   log_dir=pjoin('visualize', self.name))
        else:
            raise Exception('Grapher {} is not defined.'.format(grapher) +
                            'Choose between "visdom" and "tensorboard"')

    def _makedirs(self):
        """Creates the directories where logs will be stored."""
        content = 'This file contains the history of actions of logger:\n'
        if os.path.exists(self.logs_dir):
            raise Exception('{} directory already exists !'.format(self.logs_dir)
                            + 'Put add_date to True or give another name.')
        os.makedirs(self.logs_dir, exist_ok=True)
        content = '{}- creating {} directory\n'.format(content, self.logs_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        content = '{}- creating {} directory\n'.format(content, 
                                                       self.checkpoints_dir)
        os.makedirs(self.grapher_data_folder, exist_ok=True)
        content = '{}- creating {} directory\n'.format(content, 
                                                       self.grapher_data_folder)

        content = '{}- creating {} file\n'.format(content, self._logs_file)
        self._writefile(self._logs_file, content, mode='w')

    def _save(self, path):
        """Saves a state (dictionary) using torch.save()"""
        if self.path:
            state_to_save = dict()
            self.state['epoch'] = self._epoch
            for key in self.state.keys():
                if 'state_dict' in dir(self.state[key]):
                    state_to_save[key] = self.state[key].state_dict()
                else:
                    state_to_save[key] = self.state[key]
            torch.save(state_to_save, path)

    def _save_parameters(self):
        """Saves parameters of the experience."""
        if self.path:
            self.params.save(self.logs_dir)
            self._writefile(self._logs_file, '- save parameters\n')

    def _writefile(self, path, content, mode='a'):
        """Writes content to the file given by path."""
        if self.path:
            with open(path, mode) as f:
                f.write(content)

    def add(self, name, obj, overwrite=False):
        """Adds an object to the state (dictionary)."""
        if name in self.state.keys() and not overwrite:
            raise Exception('{} is already added !'.format(name) +
                            'To overwrite it you have to put overwrite=True')
        self.state[name] = obj
        self._writefile(self._logs_file, '- add : {}\n'.format(name))

    def add_scalar(self, tag, scalar_value, global_step=None, save_csv=True):
        """Adds a scalar to the grapher."""
        self.grapher.add_scalar(tag, scalar_value, global_step)
        if save_csv:
            self.write_to_csv(tag, [tag], [scalar_value])

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """Adds scalars to the grapher."""
        self.grapher.add_scalars(main_tag, tag_scalar_dict, global_step)

    def export_scalars_to_json(self, path):
        """Exports scalars to json"""
        self.grapher.export_scalars_to_json(path)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """Add histogram to summary."""
        self.grapher.add_histogram(tag, values, global_step, bins)

    def add_image(self, tag, img_tensor, global_step=None, caption=None):
        """Add image data to summary."""
        self.grapher.add_image(tag, img_tensor, global_step, caption)

    def add_figure(self, tag, figure, global_step=None, close=True):
        """Render matplotlib figure into an image and add it to summary."""
        self.grapher.add_figure(tag, figure, global_step, close)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4):
        """Add video data to summary."""
        self.grapher.add_video(tag, vid_tensor, global_step, fps)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100):
        """Add audio data to summary."""
        self.grapher.add_audio(tag, snd_tensor, global_step, sample_rate)

    def add_text(self, tag, text_string, global_step=None):
        """Add text data to summary."""
        self.grapher.add_text(tag, text_string, global_step)

    def add_graph_onnx(self, prototxt):
        self.grapher.add_graph_onnx(prototxt)

    def add_graph(self, model, input_to_model=None, verbose=False, **kwargs):
        """Adds a graph to the grapher."""
        self.grapher.add_graph(model, input_to_model, verbose, **kwargs)

    def add_embedding(self, mat, metadata=None, label_img=None,
                      global_step=None, tag='default', metadata_header=None):
        """Adds an embedding to the grapher."""
        self.grapher.add_embedding(mat, metadata, label_img, global_step,
                                   tag, metadata_header)

    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None):
        """Adds precision recall curve."""
        self.grapher.add_pr_curve(tag, labels, predictions, global_step, 
                                  num_thresholds, weights)

    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall, 
                         global_step=None, num_thresholds=127, weights=None):
        """Adds precision recall curve with raw data."""
        self.grapher.add_pr_curve_raw(tag, true_positive_counts,
                                    false_positive_counts,
                                    true_negative_counts,
                                    false_negative_counts,
                                    precision,
                                    recall, 
                                    global_step, num_thresholds, weights)

    def add_parameters(self, params):
        """Adds parameters."""
        self.params = _Params(params=vars(params))
        self._save_parameters()
        if self.grapher:
            self.add_text("params", str(vars(params)))

    def close(self):
        """Close the grapher."""
        self.grapher.close()

    def info(self, *argv):
        """Adds an info to the logging file."""
        msg = ' '.join(list(map(str, argv)))
        self._logger.info(msg)

    def store(self, loss, save_every=1, overwrite=True):
        """Checks if we have to store or if the current model is the best. 
        If it is the case save the best and return True."""

        # save in metada of logger actions
        if self._epoch_prints:
            self._epoch_prints = False
            self._writefile(self._logs_file, 
                            '- save every {} epochs with overwrite = {}\n'
                                .format(save_every, overwrite))
        
        # Specify the name for the checkpoint 
        checkpoint_name = 'checkpoint.pth' if overwrite \
                          else 'checkpoint_{}.pth'.format(self._epoch)

        # Save the checkpoint
        if self._epoch % save_every == 0 :
            self._save(pjoin(self.checkpoints_dir, checkpoint_name))

        self._epoch += 1
        
        # Check if the new loss is better than the stored
        # if True save as best and return True
        if loss < self._loss:
            self._save(pjoin(self.logs_dir, 'best.pth'))
            self._loss = loss
            return True
        return False

    def save(self):
        """Saves the grapher."""
        self.grapher.save()

    def register_plots(self, values, epoch, prefix, apply_mean=True, 
                       save_csv=True, info=True):
        """Helper to register a  dictionary with multiple list of scalars.
        It will compute the average for each list, send them to the grapher and
        return the new dictionary with the averages."""
        if save_csv:
            content = {}
        for k, v in values.items():
            if isinstance(v, dict):
                self.register_plots(values[k], epoch, prefix=prefix, apply_mean=apply_mean)
            if apply_mean:
                v = np.mean(v)
                k = '{}_mean'.format(k)
            if 'mean' in k or 'scalar' in k:
                key_name = k.split('_')[0]
                value = v.item() if not isinstance(v, (int, float, np.float32,
                                                       np.float64)) else v
                self.add_scalar('{}_{}'.format(prefix, key_name), 
                                value, epoch, save_csv=False)
                if save_csv:
                    content[key_name] = value
        if save_csv:
            head_line = list(content.keys())
            head_line.sort()
            line = [content[key] for key in head_line]
            self.write_to_csv(prefix, head_line, line)
        if info:
            self.info('[{}] Epoch: {:04d},'.format(prefix, epoch),
                      ', '.join('{}: {:.10f}'.format(key, content[key]) \
                                for key in sorted(content.keys())))
        return content

    def restore(self, folder=None, best=False):
        """Loads a state using torch.load()"""
        if best:
            path = pjoin(self.logs_dir, 'best.pth')
        else:
            if folder == self.logs_dir:
                warnings.warn('You are loading parameters from the current '
                              'directory where this experience is saved, '
                              'it may leads to an error')
        
            path = pjoin(folder, self.checkpoints_dir[len(self.logs_dir)+1:])
            checkpoint = [file for file in os.listdir(path) 
                                    if os.path.isfile(pjoin(path, file))]
            path = pjoin(path, sorted(checkpoint)[-1])
        
        if not self.path:
            raise ValueError('You have initialized the logger without '
                             'specifying a path. '
                             'You cannot restore any model...')

        state_restored = torch.load(path)

        for key in self.state.keys():
            if 'state_dict' in dir(self.state[key]):
                self.state[key] = self.state[key].load_state_dict(
                                                    state_restored[key])
            else:
                if type(self.state[key]) is list:
                    print(key)
                    self.state[key] += state_restored[key]
                elif type(self.state[key]) in [dict, set]:
                    self.state[key].update(state_restored[key])
                else:
                    self.state[key] = state_restored[key]
        
        self._epoch = self.state['epoch']
        self._writefile(self._logs_file, 
                        '\n- restored form {} at epoch {}'.format(path, 
                                                                  self._epoch))

    def restore_parameters(self, path):
        """Loads the parameters of a previous experience given by path"""
        if path == self.logs_dir:
            warnings.warn('You are loading parameters from the current '
                          'directory where this experience is saved. '
                          'This can lead to an error')
        self.params.update(path)
        self._writefile(self._logs_file,
                        '- load parameters from {}\n'.format(path))
        return self.params

    def warning(self, *argv):
        """Adds a warning to the logging file."""
        msg = ' '.join(list(argv))
        self._logger.warning(msg)

    def write_to_csv(self, file_name, head_line, content):
        """Write args in a csv file."""
        file_name = '{}.csv'.format(file_name)
        head_line = '{}\n'.format(', '.join(map(str, head_line)))
        content = '{}\n'.format(', '.join(map(str, content)))
        path = pjoin(self.grapher_data_folder, file_name)
        if os.path.isfile(path):
            self._writefile(path, content)
        else:
            self._writefile(self._logs_file, 
                            '- creating {} file\n'.format(path))
            self._writefile(path, '{}{}'.format(head_line, content), mode='w')