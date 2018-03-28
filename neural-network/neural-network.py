from numpy import exp, array, random, dot


#Neural network class
class NeuralNetwork():

	#Contructor
	def __init__(self):

		#Generate random weights
		random.seed(1)
		self.synaptic_weights = 2 * random.random((3, 1)) - 1

	#Get sigmoid curve that normalizes inputs to 0 and 1
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	#Gradient of the sigmoid curve
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	#Train with inputs and outputs
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):

			#Pass training set through our neural net
			output = self.predict(training_set_inputs)

			#Calculate the error
			error = training_set_outputs - output
			
			#Multiply the error by input and by gradient of the sigmoid curve
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			
			#Adjust the weights
			self.synaptic_weights += adjustment

	#Predict using the given inputs and weights
	def predict(self, inputs):
		return self.__sigmoid(dot(inputs, self.synaptic_weights))


#Main method
if __name__ == '__main__':

	#initialize a single neural network
	neural_network = NeuralNetwork()

	#Print out weights before the training
	print ('Random starting synaptic weights:')
	print (neural_network.synaptic_weights)

	#The training set. 4 examples with 3 inputs and 1 output.
	#Train 10,000 times with small adjustments each time
	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = array([[0, 1, 1, 0]]).T
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	#Print out weights after the trainings
	print ('New synaptic weights after training: ')
	print (neural_network.synaptic_weights)

	#Test the neural network
	print ('Predicting')
	print (neural_network.predict(array([1, 0, 0])))
	print (neural_network.predict(array([0, 0, 1])))
	print (neural_network.predict(array([1, 0, 1])))
	print (neural_network.predict(array([0, 0, 0])))