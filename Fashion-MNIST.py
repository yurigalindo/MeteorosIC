#Credit: Chris_Palmer from fast.ai forums
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from scipy.misc.pilutil import imsave
import tensorflow as tf
import numpy as np
import csv

WORK_DIRECTORY = 'Fashion-MNIST'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

# def maybe_download(filename):
#   """Download the data from Yann's website, unless it's already here."""
#   if not tf.gfile.Exists(WORK_DIRECTORY):
#     tf.gfile.MakeDirs(WORK_DIRECTORY)
#   filepath = os.path.join(WORK_DIRECTORY, filename)
#   if not tf.gfile.Exists(filepath):
#     filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
#     with tf.gfile.GFile(filepath) as f:
#       size = f.Size()
#     print('Successfully downloaded', filename, size, 'bytes.')
#   return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

train_data_filename = os.path.join(WORK_DIRECTORY,'train-images-idx3-ubyte.gz')
train_labels_filename = os.path.join(WORK_DIRECTORY,'train-labels-idx1-ubyte.gz')
test_data_filename = os.path.join(WORK_DIRECTORY,'t10k-images-idx3-ubyte.gz')
test_labels_filename = os.path.join(WORK_DIRECTORY,'t10k-labels-idx1-ubyte.gz')

# Extract it into np arrays.
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

if not os.path.isdir("Fashion-MNIST/train-images"):
   os.makedirs("Fashion-MNIST/train-images")

if not os.path.isdir("Fashion-MNIST/test-images"):
   os.makedirs("Fashion-MNIST/test-images")

# process train data
with open("Fashion-MNIST/train-labels.csv", 'w', newline='') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(train_data)):
    imsave("Fashion-MNIST/train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
    writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])

# repeat for test data
with open("Fashion-MNIST/test-labels.csv", 'w', newline='') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(test_data)):
    imsave("Fashion-MNIST/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
    writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])
