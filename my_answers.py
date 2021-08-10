import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # NOTE: print once so I can understand what is going on
        self.print_once = True
        
        #### TODONE: Set self.activation_function to your implemented sigmoid function ####
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #NOTE: i don't fully understand this functionality, prefer a clear function see below
#        self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        def sigmoid(x):
        # Replace 0 with your sigmoid calculation here
            return 1/(1+np.exp(-x))
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]        
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        # NOTE: numpy view different from torch view.. don't use numpy view, use reshape
        xf = np.ravel(X).reshape((X.shape[0],-1))
        hin = np.dot(xf.transpose(), self.weights_input_to_hidden)
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
#        hidden_inputs = None # signals into hidden layer
        hidden_inputs = hin # + bias?
        # signals into hidden layer, flatten the signal, just in case....
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
#        final_inputs = None # signals into final output layer
        final_inputs = np.sum( np.dot(hidden_outputs, self.weights_hidden_to_output ) )  # + bias? 
        # signals into final output layer
        final_outputs = final_inputs # signals from final output layer, this is f(x) = x; derivative is just 1.
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        # REF: Udacity Gradient decent lesson

        # TODO: Output error - Replace this value with your calculations.
#        output_error_term = None
        error = np.ravel(y - final_outputs) # Output layer error is the difference between desired target and actual output.
        error = error.reshape((error.shape[0],-1))
        #NOTE: this seemed to be skipped in the outline, but form previous example 
        #  the output error term is calculated from the error * f'(x), which is this case is just the error
        output_error_term = error  # * 1, the derivative, should be above hidden_error....
        
        # TODO: Calculate the hidden layer's contribution to the error
        # because the derivative of the f(x) = x activation is just 1, then the output_error_term is just error
        # then the hidden_error is np.dot(error, )
#        hidden_error = None
        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.transpose())
    
        # TODO: Backpropagated error terms - Replace these values with your calculations.
#        hidden_error_term = None
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        xl = np.ravel(X)
        xl = xl.reshape((xl.shape[0],-1))
        # Weight step (input to hidden)
#        delta_weights_i_h += None
        delta_weights_i_h +=  np.dot(xl, hidden_error_term)
        
        # Weight step (hidden to output)        
#        delta_weights_h_o += None
        if self.print_once:
            print('b, hidden_outputs.t:',type(hidden_outputs))
            print('b, hidden_outputs.s:',hidden_outputs.shape)
            print('b, output_error_term.t:',type(output_error_term))
            print('b, output_error_term.s:',output_error_term.shape)
            self.print_once = False
            
        delta_weights_h_o += np.dot(hidden_outputs.transpose(), output_error_term )
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
#        self.weights_hidden_to_output += None # update hidden-to-output weights with gradient descent step
#        self.weights_input_to_hidden += None # update input-to-hidden weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #NOTE: not working with full data, set xf = features instead...
        xf  = features
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
#        hidden_inputs = None # signals into hidden layer
        #NOTE: not working with full data, set xf = features instead...
        hin = np.dot(xf, self.weights_input_to_hidden )
        hidden_inputs = hin # + bias?
#        hidden_outputs = None # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
#        final_inputs = None # signals into final output layer
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output )  # + bias? 
        
        final_outputs = final_inputs # signals from final output layer, this is f(x) = x; derivative is just 1.
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
#iterations = 100
#learning_rate = 0.1
#hidden_nodes = 2
#output_nodes = 1

iterations = 2000
learning_rate = 1.0
hidden_nodes = 6
output_nodes = 1
