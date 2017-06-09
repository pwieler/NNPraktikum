# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from util.loss_functions import AbsoluteError
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        # self.weight = np.random.rand(self.trainingSet.input.shape[1])/100 # no bias
        self.weight = np.insert(np.random.rand(self.trainingSet.input.shape[1])/100,0,1) # bias

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

<<<<<<< HEAD
        for i in range(1, self.epochs + 1):
            print "Epoch " + str(i)
            for inp, t in zip(self.trainingSet.input, self.trainingSet.label):
                out = self.classify(inp)
                self.updateWeights(inp, t-out)
=======
        # Try to use the abstract way of the framework
        from util.loss_functions import DifferentError
        loss = DifferentError()

        learned = False
        iteration = 0

        # Train for some epochs if the error is not 0
        while not learned:
            totalError = 0
            for input, label in zip(self.trainingSet.input,
                                    self.trainingSet.label):
                output = self.fire(input)
                if output != label:
                    error = loss.calculateError(label, output)
                    self.updateWeights(input, error)
                    totalError += error

            iteration += 1
            
            if verbose:
                logging.info("Epoch: %i; Error: %i", iteration, -totalError)
            
            if totalError == 0 or iteration >= self.epochs:
                # stop criteria is reached
                learned = True
>>>>>>> origin/Ex1

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
<<<<<<< HEAD
        # Write your code to do the classification on an input image

        return int(self.fire(testInstance))
=======
        return self.fire(testInstance)
>>>>>>> origin/Ex1

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
<<<<<<< HEAD

        # Write your code to update the weights of the perceptron here
        # self.weight += self.learningRate * error * input # no bias
        self.weight += self.learningRate * error * np.insert(input,0,1) # bias
         
=======
        self.weight += self.learningRate*error*input

>>>>>>> origin/Ex1
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        # return Activation.sign(np.dot(np.array(input), self.weight)) # no bias
        return Activation.sign(np.dot(np.array(np.insert(input,0,1)), self.weight)) # bias
