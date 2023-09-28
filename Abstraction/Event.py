import numpy as np
from datetime import datetime
import pandas as pd

#from utils_abstraction import *
from src.Abstraction.utils_abstraction import getAttributes

class Event :
    def __init__(self,event=None,attrNames=[]):
        self.name = None
        self.attributes = {}
        self.timestamp = datetime.now()
        self.state = 'complete'
        if event:
            self.name = event["concept:name"]
            self.attributes = getAttributes(event,attrNames)
            try :
                self.timestamp = event["time:timestamp"]
            except:
                self.timestamp = datetime.now()
            try: 
                self.state = event["lifecycle:transition"]
            except:
                self.state = 'complete'
        else:
            self.name = None

    def create(self,features,attributes):
        self.name = features['name']
        self.ID = features['case']
        self.timestamp = features['timestamp']
        self.state = features['state']
        self.cluster = features['cluster']
        self.attributes = attributes

    def toDf(self,attrNames,case= None):
        attr = []
        if self.attributes != []:
            if type(self.attributes) is list:
                attr = [a for a in self.attributes]
            elif type(self.attributes) is dict:
                attr = [self.attributes[a] for a in self.attributes]
                
        if case != None:
            self.ID = case
        try :
            data = [self.name,self.ID,self.timestamp,self.state,self.cluster]
            column = ['concept:name','case','time:timestamp','lifecycle:transition','cluster']
        except:
            data = [self.name,self.ID,self.timestamp,self.state]
            column = ['concept:name','case','time:timestamp','lifecycle:transition']
        data.extend(attr)
        
        if type(self.attributes) is list:
            column.extend([attrNames[i] for i,a in enumerate(attrNames)])
        if type(self.attributes) is dict:
            column.extend([attrNames[i] for i,a in enumerate(attrNames)])
        df = pd.DataFrame([data],columns = column)
        return df
        
    def __str__(self):
        try:
            if self.attributes != []:
                strAttr = ','.join([str(a) for a in self.attributes])
                return "%s,%s,%s,%s,%s"%(self.name,self.ID, str(self.timestamp),self.state,strAttr)
            else:
                return "%s,%s,%s,%s"%(self.name,self.ID, str(self.timestamp),self.state)
        except:
            return str(self.name)
        
    def __repr__(self):
        return "<Event : concept:name = %s, lifecycle:transition = %s, time:timestamp = %s>"%(self.name,self.state, str(self.timestamp))
    
    def attribute (self, name):
        return self.attributes[name].value

    def encodeAttributes (self,attrValues):
        encoded = []
        for name,attr in self.attributes.items():
            oneHot = [0 for i in range(max(attrValues[name].values())+1)]
            oneHot[attrValues[name][attr.value]] = 0.5
            encoded += oneHot
        return np.array(encoded)