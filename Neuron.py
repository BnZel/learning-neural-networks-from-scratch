class Neuron():
    def __init__(self, index, inputn, output):
        self.index = index
        # self.coordinatesMidPoint = coordinates
        self.inputn = inputn
        self.output = output
        # print(self.__str__())
    
    def __str__(self):
        return f"Neuron(i:{self.index}, input:{self.inputn}, output:{self.output})"