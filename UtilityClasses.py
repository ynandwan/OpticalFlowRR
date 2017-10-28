import numpy as np


class EWMA():
    def __init__(self,memory):
        self.memory = memory
        self.reset()

    def reset(self):
        self.count = 0 
        self.normalizingFactor = 1 
        self.memoryToPowerCount = 1 
        self.sum = 0 
        self.x = 0 
        self.avg = 0 
        self.y1 = 0 


    def update(self,x,n = 1): 
        self.count += n
        self.x = x 
        memoryToPowerN = self.memory**n
        if (self.memory != 1): 
            factor =  (1 - memoryToPowerN)/(1 - self.memory)
            self.memoryToPowerCount = self.memoryToPowerCount*memoryToPowerN
            self.normalizingFactor = (1 - self.memoryToPowerCount)/(1 - self.memory) 
        else:
            self.normalizingFactor = self.count
            factor = n 
    
        #   
        self.sum = (memoryToPowerN)*self.sum + self.x*factor
        self.avg = self.sum/self.normalizingFactor


