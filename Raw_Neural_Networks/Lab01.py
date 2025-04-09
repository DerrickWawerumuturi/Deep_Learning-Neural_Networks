# In this lab, we will build a neural network from scratch
# and code how it performs predictions using forward propagation.
# Please note that all deep learning libraries have the entire training and prediction processes implemented, so in practice, you wouldn't need to build a neural network from scratch. However, completing this lab will help you better understand neural networks
# and how they work.

import numpy as np

# initialize the weights and biases
weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases

# compute the output for given input x1, x2
x_1 = 0.5
x_2 = 0.85

# compute the weighted sum og the inputs z1, at the first node
z_11 =x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

# compute the weighted sum og the inputs z2, at the second node
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))

# compute the activation of the first node
a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

# compute the activation of the second node
a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

# compute the weighted sum of inputs to the node in the output layer
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs to the node in the output layer is {}'.format(np.around(z_2, decimals=4)))

# compute the activation of the node in the output layer
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))