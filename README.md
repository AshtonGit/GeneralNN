# GeneralNN
Neural Network classifier written from scratch.
Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns
This project contains implementations for both binary and multi-class classification.


**Multi-Classifier**

Classifies an instance as one of many  possible classes.
A softmax activation function is used on networks final layer to normalize all outputs.
Before applying softmax an output could be negative or greater than one. After applying softmax each
output will be between (0,1) and the sum of all outputs will total 1 so that they can be interpreted 
as probabilities.

**Binary-Classifier**

Classifies each instance as being one of two groups (pass/fail, dog/not dog etc).
Sigmoid activation functions are used for all layers of the neural network.
The sigmoid function normalises a nodes output to between 0 and 1 but unlike softmax, 
the outputs of all nodes in the same layer do not sum up to 1 and so are not used as probablilities.

