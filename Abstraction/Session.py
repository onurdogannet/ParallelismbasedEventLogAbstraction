import numpy as np
import pandas as pd

from src.Abstraction.utils_abstraction import *
from src.Abstraction.Event import Event
from src.Abstraction.Attribute import Attribute

class Session :
    def __init__ (self, caseid, events = []):
        self.ID = caseid
        self.events = events
        self.distinct = findDistinct(self.events)
    
    def __str__(self):
        return f"{self.ID} {self.events}"
       
    def addEvent (self, event):
        self.events.append(event)
        self.distinct = findDistinct(self.events)
    
    def timeEncoding (self,events):
        self.encoded = [0 for i in range(len(events))]
        self.frequency = [0 for i in range(len(events))]
        self.mode = "time"
        duration = [0 for i in self.distinct]
        checked = []
        for i,e in enumerate(self.events):
            self.frequency[events.index(e.name)] +=1
            index = self.distinct.index(e.name)
            if e.state == 'start':
                event_complete = next((n for n in self.events[i:] if n.state == 'complete' and n.name == e.name),None)
                if event_complete != None:
                    checked.append(i)
                    duration[index] = duration[index] + (event_complete.timestamp - e.timestamp).total_seconds()
                elif i+1<len(self.events):
                    duration[index] = duration[index] + (self.events[i+1].timestamp- e.timestamp).total_seconds()
            elif e.state == 'complete' and i not in checked:
                if i+1<len(self.events):
                    duration[index] = duration[index] + (self.events[i+1].timestamp- e.timestamp).total_seconds()
        for i,d in enumerate(duration):
            self.encoded[events.index(self.distinct[i])] = d

    def timeEncodingNew (self,eventsNames):
        self.encoded = {i:0 for i in eventsNames}
        checked = []
        for i,e in enumerate(self.events):
            if e.state == 'start':
                if i+1<(len(self.events)) and self.events[i+1].state == 'complete' and self.events[i+1].name == e.name:
                    event_complete = self.events[i+1]
                else:
                    event_complete = None
                if event_complete != None:
                    checked.append(i+1)
                    self.encoded[e.name] += (event_complete.timestamp - e.timestamp).total_seconds()
                elif i+1<len(self.events):
                    if self.encoded[e.name] == np.nan:
                        self.encoded[e.name] = 0 
                    self.encoded[e.name] += abs((self.events[i+1].timestamp- e.timestamp).total_seconds())
                elif self.encoded[e.name] == 0:
                    self.encoded[e.name] = np.nan
            elif e.state == 'complete' and i not in checked:
                if i>0:
                    if self.encoded[e.name] == np.nan:
                        self.encoded[e.name] = 0 
                    self.encoded[e.name] += abs((self.events[i-1].timestamp-e.timestamp).total_seconds())
                elif self.encoded[e.name] == 0:
                    self.encoded[e.name] = np.nan

    def frequencyEncoding(self,events):
        self.encoded = [0 for i in range(len(events))]
        self.mode = "freq"
        for i,e in enumerate(self.events):
            #if i > 0 and ( self.events[i-1].timestamp == e.timestamp and self.events[i-1].name == e.name):
            self.encoded[events.index(e.name)] += 1
        self.encoded = normalize(self.encoded)

    def freqEncodingNew(self,eventsNames):
        self.encoded = {i:0 for i in eventsNames}
        checked = []
        for i,e in enumerate(self.events):
            if e.state == 'start':
                self.encoded[e.name] +=1
                event_complete = next((n for n in self.events[i:] if n.state == 'complete' and n.name == e.name),None)
                if event_complete != None:
                    checked.append(i)
            elif e.state == 'complete' and i not in checked:
                self.encoded[e.name] +=1
        
    def attributesEncoding(self, attrValues):
        attrEncoded = []
        for j in attrValues.values():
            attrEncoded += [0 for i in range(max(j.values())+1)]
        for i in self.events:
            attrEncoded = np.array(attrEncoded)+i.encodeAttributes(attrValues)
        self.attrEncoded  = normalize(attrEncoded.tolist(),factor = 2)

    def addAttributesMean(self,attrNames):
        tot = [0.0 for _ in attrNames]
        num = [0 for _ in attrNames]
        for e in self.events:
            for i,a in enumerate(attrNames):
                if e.attribute(a) != None:
                    tot[i] += float(e.attribute(a))
                    num[i] += 1
        self.encoded.extend(list(np.divide(tot,num)))

    def addAttributesPrePost(self,attrNames):
        for a in attrNames:
            if self.events[0].attribute(a) != None and self.events[-1].attribute(a) != None:
                self.encoded.extend([float(self.events[0].attribute(a)), float(self.events[-1].attribute(a))])
            else:
                pre = next((e.attribute(a) for e in self.events if e.attribute(a) == None), 0)
                post = next((e.attribute(a) for e in reversed(self.events) if e.attribute(a) == None), 0)
                if pre != None:
                    pre = float(pre)
                else:
                    pre = float(0)
                if post != None:
                    post = float(post)
                else:
                    post = float(0)
                self.encoded.extend([pre,post])

    def addAttributesPrePostDF(self,attrNames):
        self.attrEncoded = {a:None for a in attrNames}
        for a in attrNames:
            if self.events[0].attribute(a) != None and self.events[-1].attribute(a) != None:
                self.attrEncoded['initial:'+a] = float(self.events[0].attribute(a))
                self.attrEncoded['final:'+a] = float(self.events[-1].attribute(a))
            else:
                pre = next((e.attribute(a) for e in self.events if e.attribute(a) == None), 0)
                post = next((e.attribute(a) for e in reversed(self.events) if e.attribute(a) == None), 0)
                if pre != None:
                    pre = float(pre)
                else:
                    pre = float(0)
                if post != None:
                    post = float(post)
                else:
                    post = float(0)
                self.attrEncoded['initial:'+a] = pre
                self.attrEncoded['final:'+a] = post

    def addAttributesMeanDF(self,attrNames):
        tot = {i:0.0 for i in attrNames}
        num = {i:0 for i in attrNames}
        for e in self.events:
            for a in attrNames:
                if e.attribute(a) != None:
                    tot[a] += float(e.attribute(a))
                    num[a] += 1
        self.attrEncoded = {k:float(tot[k]/num[k]) if num[k] >0 else np.nan for k in num }
            
    def joinEncoding(self):
        self.encoded = self.encoded + self.attrEncoded

    def abstract(self,center, cluster, distinctEvent,attrNames):
        ind = list(np.flip(np.argsort(center[:len(distinctEvent)])[-1:]))
        name = concatName(distinctEvent,ind)
        eventS = {
            "name" : name+str(cluster),
            "case" :self.ID,
            "timestamp" : self.events[0].timestamp,
            "state" :"start",
            "cluster": cluster
        }
        eventC = {
            "name" : name+str(cluster),
            "case" :self.ID,
            "timestamp" : self.events[-1].timestamp,
            "state" :"complete",
            "cluster": cluster
        }
        attributes = []
        if len(center) > len(distinctEvent):
            fromcenter = center[len(distinctEvent):]
            for i,an in enumerate(attrNames):
                attributes.append(Attribute(an,fromcenter[i]))
        else:
            attrV = {a:0 for a in attrNames}
            freq = {a:0 for a in attrNames}
            for e in self.events:
                for a in attrNames:
                    if e.attribute(a) != None:
                        try:
                            attrV[a] += float(e.attribute(a))
                            freq[a] +=1
                        except:
                            attrV[a] = e.attribute(a)
            attributes = [Attribute(k,safediv(attrV[k],freq[k])) if freq[k] > 0 else Attribute(k,None) for k in attrV]
        abstractedS = Event()
        abstractedC = Event()
        abstractedS.create(eventS,attributes)
        abstractedC.create(eventC,attributes)
        return [abstractedS.toDf(attrNames),abstractedC.toDf(attrNames)]
    
    def convert (self,center,cluster,distinctEvent,attrNames):
        es = self.abstract(center,cluster,distinctEvent,attrNames)
        strForm = str(es)
        return strForm

    def convertSession (self,center,cluster,distinctEvent,attrNames):
        frames = self.abstract(center,cluster,distinctEvent,attrNames)
        return pd.concat(frames,ignore_index = True)

    def export(self,attrNames,case):
        toDF = []
        for e in self.events:
            toDF.append(e.toDf(attrNames, case))
        return toDF
    
   