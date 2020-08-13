from Layer import Layer
from Neuron import Neuron
from TrainingData import TrainingData
import math, random

class Synapse(object):
    def __init__(self, weight, neuron_left, neuron_right):
        self.weight = weight
        self.deltaweight = 0.000
        self.batch_deltaweight = 0.000
        self.neuron_left = neuron_left
        self.neuron_right = neuron_right
        self.__str__()

    def __str__(self):
        return f"Synapse(weight:{self.weight}, left:{self.neuron_left}, right:{self.neuron_right}"

class Net():
    def __init__(self):
        self.layers = []
        self.synapsesL0L1 = {}          # neuron connections per layer, layer 0 layer 1 layer 2
        self.synapsesL1L2 = {}

        self.update_type = 'batch_deltaweight' # deltaweight batch_deltaweight
        self.initial_weight_vals = 0.01
        self.epoch = 0
        self.delta = 0.000
        self.deltagradientL2 = 0.000    # last layer
        self.MSE = 0.000
        self.learning_rate = 1          # amount that weights are updated during training (step), range between 0.0 - 1.0
        self.momentum = 0               # technique used along with SDG, accumlates the gradient of the past steps to determine the direction to go for speed and accuracy
        self.expected_value = 0.000

        self.epoch_MSEs = []
        self.batch_MSE = 0.0

        self.output_ = 0.0
        
        self.training_data = TrainingData()
        self.build_all_layers_and_neurons()
        self.build_all_synapses()
        self.initialize_weights()
        
        # self.load_inputs()     

    def get_neuron(self, layer_index, neuron_index):                
        return self.layers[layer_index].get_neuron_from_layer(neuron_index)

    def set_input(self, Neuron, inputn):
       setattr(Neuron, 'inputn', inputn)
    
    def get_input(self, Neuron):        
        return float(getattr(Neuron, 'inputn'))
    
    def set_output(self, Neuron, output):
        setattr(Neuron, 'output', output)
    
    def get_output(self, Neuron):
        return float(getattr(Neuron,'output'))

    def initialize_weights(self):
        """
        referring to sigmoid function 
        x axis between -2,2 are the most active 
        which will increase the chance of improving weights
        closer to 0 will make the network difficult to do division 
        and assign takes to different neuron
        """
        # print("==========Layer 1 to Layer 2 Random Weight Synapse==========")
        # L0L1
        for i in range(0,3):
            for j in range(0,4):
                rand_weight = random.uniform(-2,2)
                setattr(self.synapsesL0L1[i,j], 'weight', rand_weight)

                # print(f"{self.synapsesL0L1[i,j]}")

        # print("\n")

        # print("==========Layer 2 to Layer 3 Random Weight Synapse==========")
        # include layer 1 bias
        for i in range(0,5):
            rand_weight = random.uniform(-2,2)
            setattr(self.synapsesL1L2[i,0], 'weight', rand_weight)

            # print(f"{self.synapsesL1L2[i,0]}")

    # connection between neurons
    # neuron_index1 = left , neuron_index2 = right
    def get_weight(self, layer_index, neuron_index1, neuron_index2):
        if layer_index == 0:
            return getattr(self.synapsesL0L1[neuron_index1, neuron_index2], 'weight')
        else:
            return getattr(self.synapsesL1L2[neuron_index1, neuron_index2], 'weight')

    def get_deltaweight(self, layer_index, neuron_index1, neuron_index2):
        if layer_index == 0:
            return getattr(self.synapsesL0L1[neuron_index1, neuron_index2], 'deltaweight')
        else:
            return getattr(self.synapsesL1L2[neuron_index1, neuron_index2], 'deltaweight')

    def get_batch_deltaweight(self, layer_index, neuron_index1, neuron_index2):
        if layer_index == 0:
            return getattr(self.synapsesL0L1[neuron_index1, neuron_index2], 'batch_deltaweight')
        else:
            return getattr(self.synapsesL1L2[neuron_index1, neuron_index2], 'batch_deltaweight')
    
    def delta_printL0L1(self):
        for i in range(0,3):
            prev_neuron = self.get_neuron(0,i)
            prev_neuron_output = self.get_output(prev_neuron)
                
            d = getattr(self.synapsesL0L1[0,i],'deltaweight')
            bd = getattr(self.synapsesL0L1[0,i],'batch_deltaweight')
        return (d,bd)

    def delta_printL1L2(self):
        for i in range(0,5):
            prev_neuron = self.get_neuron(1,i)
            prev_neuron_output = self.get_output(prev_neuron)
                
            d = getattr(self.synapsesL1L2[i,0],'deltaweight')
            bd = getattr(self.synapsesL1L2[i,0],'batch_deltaweight')
        return (d,bd)
       

    def build_all_layers_and_neurons(self):
        self.layers.append(Layer())
        
        self.layers[0].add_neuron_to_layer(Neuron(0, 0.000, 0.000))
        self.layers[0].add_neuron_to_layer(Neuron(1, 0.000, 0.000))
        # bias
        self.layers[0].add_neuron_to_layer(Neuron(2, 0.000, 1.0)) 

        self.layers.append(Layer())
        self.layers[1].add_neuron_to_layer(Neuron(0, 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(1, 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(2, 0.000, 0.000))
        self.layers[1].add_neuron_to_layer(Neuron(3, 0.000, 0.000))
        # bias 
        self.layers[1].add_neuron_to_layer(Neuron(4, 0.000, 1.0))

        self.layers.append(Layer())
        self.layers[2].add_neuron_to_layer(Neuron(0, 0.000, 0.000))
        # bias
        self.layers[2].add_neuron_to_layer(Neuron(1, 0.000, 0.000))
    
    def build_all_synapses(self):
        # print("==========Layer 0 to Layer 1 Synapse==========")
        for i in range(0,3):
            nl0 = self.get_neuron(0, i)

            for j in range(0,4):
                nl1 = self.get_neuron(1, j)
                self.synapsesL0L1[i,j] = Synapse(self.initial_weight_vals, nl0, nl1)

        # print("\n")

        # print("==========Layer 1 to Layer 2 Synapse==========")
        for i in range(0,5):
            nl1 = self.get_neuron(1,i)

            for j in range(0,1):
                nl2 = self.get_neuron(2,j)                
                self.synapsesL1L2[i, j] = Synapse(self.initial_weight_vals, nl1, nl2)        

    def load_inputs(self):
        inputs = self.training_data.get_next_inputs()        

        # EOF
        if inputs == [0,0,0]:            
            # print('====================Epoch done====================')
            self.epoch += 1
            # print(f"epoch: {self.epoch}")
            self.end_of_data_press_again = True
            self.training_data.move_to_top_of_file()
            
            if self.update_type == "batch_deltaweight":
                # print(f"====================update type is====================: {self.update_type}")
                self.update_weights('batch_deltaweight')
            
            self.epoch_MSEe_and_reset_synapse_batches()

        else:
            self.set_input(self.get_neuron(0,0), inputs[0])
            self.set_input(self.get_neuron(0,1), inputs[1])

            self.set_output(self.get_neuron(0,0), inputs[0])
            self.set_output(self.get_neuron(0,1), inputs[1])

            self.expected_value = float(inputs[2])

    # delta weight or batch delta weight
    def update_weights(self, update_type):
        # L0L1
        for i in range(0, 2):
            for j in range(0, 4):
                weight = self.get_weight(0, i, j)

                parsed_update_type = eval("self.get_" + update_type + "(0, " + str(i) + ", " + str(j) + ")")
                
                # print(parsed_update_type)

                weight_change = self.learning_rate * parsed_update_type + self.momentum * weight
                weight += weight_change

                setattr(self.synapsesL0L1[i,j], 'weight', weight)

                # print(f"synapses L0L1 update weights {self.synapsesL0L1[i,j]}")                
        
        # print("\n")

        # L2
        for i in range(0, 2):
            weight = self.get_weight(1, i, 0)

            parsed_update_type = eval("self.get_" + update_type + "(1, " + str(i) + ", 0)")

            # print(parsed_update_type)

            weight_change = self.learning_rate * parsed_update_type + self.momentum * weight
            weight += weight_change
            setattr(self.synapsesL1L2[i,0], 'weight', weight)

            # print(f"synapses L2 update weights bias: {self.synapsesL1L2[i,0]}")
    
    def calculate_MSE_and_deltagradient(self):
        """
        mean squared error: 
            measure networks performance
            error can be positive or negative when all neurons are summed which may cancel both out
            to prevent this, square the errors
            taking the absolute values would cause a jump which is dangerous to compute
            then take the average and backpropagate 
        
        delta rule (gradient descent):
            optimization algorithm used to minimize the cost function            
            for updating weights of inputs 

            taking the mountain example;
                the goal for a ball to roll down to the lowest descent
                which lowers the cost as quickly as possible

                the ball takes its first step down
                then recalculates the negative descent 
                by passing in coordinates of the new point
                and take another step specified 
                continue this process till the ball gets to the bottom (local minimum)
        """
        # output neuron of last layer
        self.delta = self.expected_value - self.get_output(self.get_neuron(2,0))
        self.MSE = 0.5 * pow(self.delta, 2)
        self.epoch_MSEs.append(self.MSE)
        # print(f"epoch_MSEs: {self.epoch_MSEs}")
        self.deltagradientL2 = self.delta * self.transfer_function_derivative(self.get_output(self.get_neuron(2,0)))

    # sigmoid
    def transfer_function(self, y):
        return 1 / (1 + math.pow(math.e, -y))

    def transfer_function_derivative(self, y):
        """
        or activation function
        
        derivative:
            slope on a curve
            used to find the max and min
            measures steepness of the graph
        
        calculating backpropagation error used to determine parameter updates
        that require the gradient of the activation function 
        for updating the layer

        sigmoid -> range 0 - 1, curves                 
        """
        # transfer function * 1 - transfer function
        return 1 / (1 + math.pow(math.e, -y)) * (1 - 1 / (1 + math.pow(math.e, -y)))
    
    def epoch_MSEe_and_reset_synapse_batches(self):
        """
        calculating average of batch
        by taking the list dividing list length
        to accumulate gradient batch samples for weight update
        """
        self.batch_MSE = sum(self.epoch_MSEs) / len(self.epoch_MSEs)
        self.epoch_MSEs.clear() # empties list

        # print('====================reset====================')        
        # resets to initial weights of 0.0
        # L0L1
        for i in range(0,3):
            for j in range(0,4):
                setattr(self.synapsesL0L1[i,j], 'batch_deltaweight', 0.0)

            # print(f"\nLayer 0 - 1 {self.synapsesL0L1[i,j]}")

        # L1L2
        for i in range(0,4):
            setattr(self.synapsesL1L2[i,0], 'batch_deltaweight', 0.0)

            # print(f'\nLayer 1 - 2 {self.synapsesL1L2[i,0]}')

    def run_epoch(self):
        there_is_data = True
        while there_is_data == True:
            there_is_data = self.load_inputs()
            self.end_of_data_press_again = False

            # batch learning 
            # don't update till EOF
            if there_is_data == False:
                break    
                # print('epoch done')
            else:
                self.forward_propL0L1()
                self.forward_propL1L2()
                self.calculate_MSE_and_deltagradient()
                self.backpropL2L1()
                self.backpropL1L0()

                if self.update_type == "deltaweight":
                    # print(f"====================update type is====================: {self.update_type}")
                    self.update_weights(self.update_type)

                # print(f"====================expected value====================: {self.expected_value}")
                # print('one data row finished')                
    
    def forward_propL0L1(self):
        """
        first neuron multiplied by weight value + second neuron multiplied by weight value  
        four neurons and each of them are using previous outputs and weights
        bias neurons doesn't care about inputs
        """
        # L1 starting from the hidden layer
        for i in range(0,4):
            sum = 0
            # to L0
            for j in range(0,3):
                prev_neuron = self.get_neuron(0,j)
                prev_neuron_output = self.get_output(prev_neuron)
                weight = self.get_weight(0,j,i)                
                sum += weight * prev_neuron_output
            
            this_neuron = self.get_neuron(1, i)
            self.set_input(this_neuron, str(sum))

            output = self.transfer_function(sum) 
            self.set_output(this_neuron, output)
            
            # print(f"forward propL0L1 weight: {weight} \nsum: {sum}\n")
            # print(f"left neuron : {i}\nright neuron: {j}")
    
    def forward_propL1L2(self):
        sum = 0
        # L1 with bias
        for i in range(0,5):
            prev_neuron = self.get_neuron(1,i)
            prev_neuron_output = self.get_output(prev_neuron)
            weight = self.get_weight(1,i,0)
            sum += weight * prev_neuron_output
            
            # to L2 output
            this_neuron = self.get_neuron(2, 0)
            self.set_input(this_neuron, str(sum))

            output = self.transfer_function(sum) 
            self.output_ = output
            self.set_output(this_neuron, output)
        
            # print(f"forward propL1L2 weight: {weight} \nsum: {sum}\nleft neuron : {i}\n\n")
            # print(f"output: {output}" )

    def backpropL2L1(self):
        """
        calculating derivatives and gradient descent 
        using output neurons of the hidden layer        

        batch weight a summation of the previous delta weights
        use when you don't want to update the weights right away
        known as pattern learning  
        """
        for i in range(0,4):            
            deltaweight = self.get_output(self.get_neuron(1,i)) * self.deltagradientL2
            setattr(self.synapsesL1L2[i,0], 'deltaweight', deltaweight)
            
            batch_deltaweight = getattr(self.synapsesL1L2[i,0], 'batch_deltaweight')
            new_batch_deltaweight = deltaweight + batch_deltaweight
            setattr(self.synapsesL1L2[i,0], 'batch_deltaweight', new_batch_deltaweight)

            # print(f"backprop L2L1 delta weight: {deltaweight} \nbatch delta weight: {batch_deltaweight} \nnew batch delta weight: {new_batch_deltaweight} \nsynapses: {self.synapsesL1L2[i,0]}")

        # bias 
        deltaweight = self.deltagradientL2
        setattr(self.synapsesL1L2[4, 0], 'batch_deltaweight', new_batch_deltaweight)
        
        # bias batch
        batch_deltaweight = getattr(self.synapsesL1L2[4,0], 'batch_deltaweight')
        new_batch_deltaweight = deltaweight + batch_deltaweight
        setattr(self.synapsesL1L2[4, 0], 'batch_deltaweight', new_batch_deltaweight)

        # print(f"bias: {deltaweight} \nbias batch: {deltaweight}, \nsynapses L1L2:{self.synapsesL1L2[4, 0]}")
        
    def backpropL1L0(self):
        """
        access to 2 neurons from layer 0 -> nl0
        access to 4 neurons from layer 1 -> nl1

        calculating output of derivatives and gradient descent 
        using output neurons of the hidden layer    
        """
        for i in range(0, 2):
            for j in range(0, 4):
                nl1 = self.get_neuron(1,j)
                nl0 = self.get_neuron(0,i)

                p1 = self.transfer_function_derivative(self.get_output(nl1)) * self.deltagradientL2
                p2 = self.get_weight(1, j, 0)   # layer 1, all neurons on layer 1, input neuron
                p3 = self.get_output(nl0)

                deltaweight = p1 * p2 * p3
                setattr(self.synapsesL0L1[i,j], 'deltaweight', deltaweight)

                batch_deltaweight = getattr(self.synapsesL0L1[i,j], 'batch_deltaweight')
                new_batch_deltaweight = deltaweight + batch_deltaweight
                setattr(self.synapsesL0L1[i, j], 'batch_deltaweight', new_batch_deltaweight)
            
                # print(f"neuron layer 1: {nl1}, \nneuron layer 0: {nl0}, \npart 1 (sigmoid derivative):{p1}, \npart 2 (weight of layer 1): {p2}, \npart 3 (input layer): {p3}, \n\nsynapses L0L1{self.synapsesL0L1[i,j]} \n\n\n")

        # bias 
        for i in range(0, 4):
            nl1 = self.get_neuron(1, i)

            p1 = self.transfer_function_derivative(self.get_output(nl1)) * self.deltagradientL2
            p2 = self.get_weight(1, i, 0)            

            deltaweight = p1 * p2
            setattr(self.synapsesL0L1[2, i], 'deltaweight', deltaweight)
            
            # bias batch
            batch_deltaweight = getattr(self.synapsesL0L1[2,i], 'batch_deltaweight')
            new_batch_deltaweight = deltaweight + batch_deltaweight
            setattr(self.synapsesL0L1[2, i], 'batch_deltaweight', new_batch_deltaweight)
        
            # print(f"bias: {deltaweight} \nbias batch: {new_batch_deltaweight}, \nsynapses L1L0:{self.synapsesL0L1[2, i]}")







