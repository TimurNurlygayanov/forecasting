import torch

from detecto.core import Dataset
from detecto.core import Model
from detecto.utils import read_image
from pathlib import Path
import warnings
import time
import multiprocessing as mp


# Suppress warning messages from PyTorch
warnings.filterwarnings('ignore', category=UserWarning)

dataset = Dataset('marked_images/')

labels = ['p']

model_name = 'saved_model.pth'
if not Path(model_name).is_file():
    model = Model(labels)
    losses = model.fit(dataset, epochs=15, learning_rate=0.001, verbose=True)  # validation_dataset,
    model.save(model_name)
else:
    model = Model.load(model_name, labels)


image = read_image('WW.png')


if __name__ == '__main__':
    labels, boxes, scores = model.predict(image)
