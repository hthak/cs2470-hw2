from __future__ import absolute_import
from matplotlib import pyplot as plt
#from sklearn.model_selection import learning_curve
from preprocess import get_data, get_next_batch
#from cnn import CNN
from mlp import MLP

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def train(model, optimizer, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    #TODO: Implement the training loop

    #shuffle dataset
    random_indices = tf.random.shuffle(tf.range(len(train_inputs)))
    train_inputs = tf.gather(train_inputs, random_indices)
    train_labels = tf.gather(train_labels, random_indices) #based of random_indices so labels stay same for data

    #put into batches
    for start in range(0, len(train_inputs), model.batch_size):
        end = start + model.batch_size
        batch_inputs = train_inputs[start:end]
        batch_labels = train_labels[start:end]

        #forward pass
        with tf.GradientTape() as tape:
            logits = model(batch_inputs)
            loss = model.loss(logits, batch_labels)

        #backwards
        gradients = tape.gradient(loss, model.trainable_variables)

        #update weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        model.loss_list.append(float(loss.numpy()))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return:
        test accuracy - this should be the average accuracy across
    all batches
        test preds - all of the model's predictions for each of the test inputs
    """
    # TODO: Implement the testing loop
    total_corr = 0
    total_images = 0
    predictions = []

    for start in range(0, len(test_inputs), model.batch_size):
        end = start + model.batch_size
        batch_inputs = test_inputs[start:end]
        batch_labels = test_labels[start:end]

        #forward
        logits = model(batch_inputs)

        #batch accuracy
        batch_accuracy = model.accuracy(logits, batch_labels)
        total_corr += float(batch_accuracy.numpy()) * len(batch_inputs)
        total_images += len(batch_inputs)

        #save predictions
        predictions.append(logits.numpy())

    test_acc = total_corr / total_images
    test_pred = np.concatenate(predictions, axis=0)
    return test_acc, test_pred

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, logits, image_labels, first_label, second_label, third_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "deer"
    :param third_label: the name of the third class, "dog"
    NOTE: DO NOT EDIT
    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            f"{label} Examples\nPL = Predicted Label\nAL = Actual Label")
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            predicted_index = predicted_labels[ind]
            actual_index = np.argmax(image_labels[ind], axis=0)
            labels = [first_label, second_label, third_label]
            pl = labels[predicted_index]
            al = labels[actual_index]
            ax.set(title=f"PL: {pl}\nAL: {al}")
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(logits, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

def main():
    '''
    Read in CIFAR10 data (limited to 3 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    Consider printing the loss, training accuracy, and testing accuracy after each epoch
    to ensure the model is training correctly.

    Students should receive a final accuracy
    on the testing examples for cat, deer and dog of >=75%.

    :return: None
    '''
    # TODO: Use the autograder filepaths to get data before submitting to autograder.
    #       Use the local filepaths when running on your local machine.
    AUTOGRADER_TRAIN_FILE = '../data/train'
    AUTOGRADER_TEST_FILE = '../data/test'

    LOCAL_TRAIN_FILE = 'data/train'
    LOCAL_TEST_FILE = 'data/test'

    classes = [3,4,5] # classes for cat, deer, & dog

    # TODO: assignment.main() pt 1
    # Load your testing and training data using the get_data function
    train_inputs, train_labels = get_data(LOCAL_TRAIN_FILE, classes)
    test_inputs, test_labels = get_data(LOCAL_TEST_FILE, classes)

    '''print(train_inputs.shape)
    print(train_labels.shape)
    print(test_inputs.shape)
    print(test_labels.shape)'''

    # TODO: assignment.main() pt 2
    # Initialize your model and optimizer
    model = MLP(classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    sample_images = train_inputs[:64]
    sample_labels = train_labels[:64]
    logits = model(sample_images)

    # TODO: assignment.main() pt 3
    # Train your model
    epochs = 10
    for epoch in range(epochs):
        train(model, optimizer, train_inputs, train_labels)
        print(f"Epoch {epoch + 1}/{epochs} completed")

    # TODO: assignment.main() pt 4
    # Test your model
    test_acc, test_pred = test(model, test_inputs, test_labels)
    print("test accuracy: ", test_acc)

    # TODO: assignment.main() pt 5
    # Save your predictions as either "predictions_cnn.npy" or "predictions_mlp.npy"
    #   depending on which model you are using
    # You will submit these prediction files to the autograder with predictions
    #    For the CAT, DEER, and DOG classes
    return


if __name__ == '__main__':
    main()
