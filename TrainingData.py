class TrainingData():
    def __init__(self):
        self.text_file = open('training_data.txt')
        self.next_line = ''
    
    def get_next_inputs(self):
        self.next_line = self.text_file.readline()
    
        if len(self.next_line) == 0:
            return [0,0,0]
        else:
            next_line_list = self.next_line.split(";")
            return [next_line_list[1], next_line_list[2], next_line_list[4]]
        
    def move_to_top_of_file(self):
        self.text_file.seek(0)
    
    def end_file_read(self):
        self.text_file.close()

