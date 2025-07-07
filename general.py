import json
import os
import subprocess
from os.path import isfile, join
from shutil import copyfile
from threading import Timer

class Config():
    """Class that loads hyperparameters from json file into attributes"""

    def __init__(self, source):
        """
        Args:
            source: path to json file or dict
        """
        self.source = source

        if type(source) is dict:
            self.__dict__.update(source)
        elif type(source) is list:
            for s in source:
                self.load_json(s)
        else:
            self.load_json(source)


    def load_json(self, source):
        with open(source) as f:
            data = json.load(f)
            self.__dict__.update(data)


    def save(self, dir_name):
        init_dir(dir_name)
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.save(dir_name)
        elif type(self.source) is dict:
            # 对象转JSON
            json.dumps(self.source, indent=4)
        else:
            copyfile(self.source, dir_name + self.export_name)
def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

def get_files(dir_name):
    files = [f for f in os.listdir(dir_name) if isfile(join(dir_name, f))]
    return files
def run(cmd, timeout_sec):
    """Run cmd in the shell with timeout"""
    proc = subprocess.Popen(cmd, shell=True)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        stdout,stderr = proc.communicate()
    finally:
        timer.cancel()
def delete_file(path_file):
    try:
        os.remove(path_file)
    except Exception:
        pass
def minibatches(data_generator, minibatch_size):
    """
    Args:
        data_generator: generator of (img, formulas) tuples
        minibatch_size: (int)

    Returns:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data_generator:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch