class Layer():
    def __init__(self):
        self.neurons = []
        # print(self.__str__())

    def add_neuron_to_layer(self, Neuron):
        self.neurons.append(Neuron)

    def get_neuron_from_layer(self, index):
        return self.neurons[index]

    def __str__(self):   
        return f"Layer({self.neurons})"