# importing the library
import numpy as np
print('\n Implementing XOR using Brackpropagation with 3 input layers \n')


# Creating the input array
X = np.array([[0,0,0] , [0,0,1] , [0,1,0] , [0,1,1] , [1,0,0] , [1,0,1] , [1,1,0] , [1,1,1]])
print('\n Input : \n', X)


# Creating the output array
actual_output = np.array([[0],[1],[1],[0],[1],[0],[0],[1]])
print ('\n Actual Output : \n' , actual_output)


# defining the Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))


# derivative of Sigmoid Function
# Gradient of sigmoid can be returned as x * (1 – x).
def derivatives_sigmoid(x):
    return x * (1 - x)


# initializing the variables
max_epoch = 5000
lr = 0.1
inputlayer_neurons = 3 
hiddenlayer_neurons = 8
output_neurons = 1


# initializing weight and bias
hidden_weights = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
hidden_bias = np.random.uniform(size=(1,hiddenlayer_neurons))
output_weights = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
output_bais = np.random.uniform(size=(1,output_neurons))
print('\n Initial hidden weights : \n' , hidden_weights )
print('\n Initial hidden bias : \n' , hidden_bias )
print('\n Initial output weights : \n' , output_weights )
print('\n Initial output bias : \n' , output_bais )

# training the model
for i in range(max_epoch):

    #Forward Propogation
    hidden_layer_input1 = np.dot(X,hidden_weights)
    hidden_layer_input = hidden_layer_input1 + hidden_bias
    hiddenlayer_activations = sigmoid(hidden_layer_input)
        #hidden_layer_input = np.dot(X,hidden_weights) + hidden_bias 
    
    output_layer_input1 = np.dot(hiddenlayer_activations,output_weights)
    output_layer_input = output_layer_input1 + output_bais
    predicted_output = sigmoid(output_layer_input)
        #output_layer_input = np.dot(hiddenlayer_activations,output_weights) + output_bais


    #Backpropagation
    error = actual_output - predicted_output
    d_predicted_output = error * derivatives_sigmoid(predicted_output)
    
    Error_at_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hiddenlayer = Error_at_hidden_layer * derivatives_sigmoid(hiddenlayer_activations)
    
    #Updating weights at output and hidden layer
    output_weights += hiddenlayer_activations.T.dot(d_predicted_output) * lr
    hidden_weights += X.T.dot(d_hiddenlayer) * lr
    
    #Updating bais at output and hidden layer
    output_bais += np.sum(d_predicted_output , axis = 0 , keepdims = True) * lr
    hidden_bias += np.sum(d_hiddenlayer , axis = 0 , keepdims = True) * lr

print ('\n Predicted Output from the training model : \n')
print (predicted_output)