from __future__ import print_function
import random
import sys
sys.path.append("../")
import numpy as np  # noqa: E402
from tensorboard_evaluation import Evaluation  # noqa: E402
from agent.bc_agent import BCAgent  # noqa: E402
from utils import *  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import gzip  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import torch  # noqa: E402


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)
                         ], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # X_train = list(map(rgb2gray, X_train))
    X_train = rgb2gray(X_train)
    X_train = np.expand_dims(X_train, axis=1)
    X_valid = rgb2gray(X_valid)
    X_valid = np.expand_dims(X_valid, axis=1)

    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.
    y_train = list(map(action_to_id, y_train))
    y_valid = list(map(action_to_id, y_valid))

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    return X_train, y_train, X_valid, y_valid


def sample_minibatches(inputs, targets, batchsize, number_of_batches=1000):
    permutation = torch.randperm(len(inputs))
    while len(permutation) <= batchsize*number_of_batches + batchsize:
        permutation = torch.cat((permutation, torch.randperm(len(inputs))), 0)
    x_minibatches = []
    y_minibatches = []
    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)
    for i in range(0, number_of_batches*batchsize, batchsize):
        indices = permutation[i:i+batchsize]
        batch_x, batch_y = inputs[indices], targets[indices]
        x_minibatches.append(batch_x)
        y_minibatches.append(batch_y.tolist())
    return x_minibatches, y_minibatches


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent()
    agent.net.train()

    tensorboard_eval = Evaluation(tensorboard_dir, "train", [
        "Accuracy", "Loss", 'Validation Accuracy', "Validation Loss"])

    # TODO: implement the training
    # 1. write a method sample_minibatch and perform an update step
    x, y = sample_minibatches(X_train, y_train, batch_size)
    agent.update(x[0], y[0])
    x_val, y_val = sample_minibatches(
        X_valid, y_valid, batch_size, int(n_minibatches / 10))

    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # training loop
    num_correct = 0
    num_samples = 0
    num_correct_valid = 0
    num_samples_valid = 0
    j = 0
    k = 0
    print("*********************************************")
    for i in range(n_minibatches):
        if k >= 1000:
            k = 0
            x, y = sample_minibatches(X_train, y_train, batch_size)
        loss, outputs = agent.update(x[k], y[k])
        _, predictions = outputs.max(1)
        targets = torch.LongTensor(y[k])
        k += 1
        if i % 10 == 0:

            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

            outputs_valid = agent.predict(x_val[j])
            targets_valid = torch.LongTensor(y_val[j])
            loss_valid = agent.criterion(
                outputs_valid, targets_valid)
            _, predictions_valid = outputs_valid.max(1)
            num_correct_valid += (predictions_valid == targets_valid).sum()
            num_samples_valid = predictions_valid.size(0)

            print('accuracy:', "%.2f" %
                  ((num_correct.item()/num_samples)*100), '%,', "Loss:", "%.4f" % (loss), 'Validation accuracy:', "%.2f" %
                  ((num_correct_valid.item()/num_samples_valid)*100), '%,', "Validation Loss:", "%.2f" % (loss_valid))
            tensorboard_eval.write_episode_data(i, eval_dict={"Accuracy": (num_correct.item()/num_samples)*100,
                                                              "Loss": loss,
                                                              "Validation Accuracy": (num_correct_valid.item()/num_samples_valid)*100,
                                                              "Validation Loss": loss_valid, })
            num_correct = 0
            num_samples = 0
            num_correct_valid = 0
            num_samples_valid = 0
            j += 1

    # compute training/ validation accuracy and write it to tensorboard

    tensorboard_eval.close_session()

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid,
                n_minibatches=1000, batch_size=64, lr=1e-4)
