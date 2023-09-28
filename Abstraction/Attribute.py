class Attribute :
    def __init__(self,name,value) :
        self.name = name
        self.value = value

    def __str__ (self):
        return  "%s"%(self.value)
    def __repr__ (self):
        return  "%s"%(self.value)
    
    def encode (self,attrValues):
        index = attrValues[self.name].index(self.value)
        for k,v in attrValues.items():
            if self.name == k :
                l = [0 for j in v]
                l[index] = 1
        return l