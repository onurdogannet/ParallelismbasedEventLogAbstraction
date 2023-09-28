import numpy as np 
import math
import pm4py as pm
# import statistics as st
import os as os
from sklearn import cluster as skcl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd
import warnings
warnings.simplefilter(action='ignore')
import time
from datetime import datetime
import pickle

from pm4py.stats import get_event_attributes
from src.Abstraction.utils_abstraction import *
from src.Abstraction.Parallelism import get_session_rules, divide_traces
from src.Abstraction.mergePetriNets import merge_subprocesses


from progressbar import progressbar as pb

class Abstractor_v4:
    def __init__(self, fileName, split, attrNames=[],noTrace = None,toSession = True, parallel = False):
        self.name = fileName
        self.hidepb = parallel
        start = time.time()
        self.log = importLog(fileName)
        if split: #split log into training, testing and evaluation
            dataframe = pm.convert_to_dataframe(self.log)
            train_df, test_df = pm.split_train_test(dataframe, train_percentage=0.8)
            trainingLog = pm.convert_to_event_log(train_df)
            testLog = pm.convert_to_event_log(test_df)
            self.log = trainingLog
            pm.write_xes(trainingLog, str(fileName.replace(".xes","")+ "_trainingtesting.xes"))
            pm.write_xes(testLog, str(fileName.replace(".xes","")+ "_evaluation.xes"))
        # self.threshold_value = self.thresholdView(os.path.dirname(os.path.abspath(__file__)))
        self.threshold_value = 12
        # self.threshold_value = self.calculate_time_diff(self.log) * 0.5
        print(f"{self.name}: Importing Time:",time.time()-start)
        self.sequence = get_session_rules(self.log)
        if toSession == True:
            if os.path.exists("sessions.pickle"): 
                with open("sessions.pickle", 'rb') as f:
                    self.sessions = pickle.load(f)
                with open("distinct.pickle", 'rb') as f:
                    self.distinct = pickle.load(f)
                print("Created sessions were read from file")
            else:
                self.sessions, self.distinct = divide_traces(self.log, self.sequence, self.threshold_value)
        if noTrace != None:
            self.tracefy(limit = noTrace)
        if attrNames== []:
            self.attrNames = get_event_attributes(self.log)
            toRemove = ['concept:name','org:resource','time:timestamp','(case)_variant-index','name','position','(case)_creator','(case)_variant']
            for i in toRemove:
                if i in self.attrNames:
                    self.attrNames.remove(i)    

    def thresholdView (self,imgpath):
        difference = []
        for caseid,case in enumerate(self.log):
            added = []
            for event_id, event in enumerate(case):
                # if event['lifecycle:transition'] == 'start' and event_id>0:
                #     difference.append ((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),event["time:timestamp"])).total_seconds())
                #     ec = next((e for e in case[event_id:] if e['concept:name'] == event['concept:name'] and e['lifecycle:transition'] == 'complete'),None)
                #     if ec != None and not started(ec, case, event_id):
                #         added.append(case.index(ec))
                # elif event['lifecycle:transition'] == 'complete' and event_id>0 and event not in added:
                #     difference.append((event["time:timestamp"]-next((e['time:timestamp'] for e in reversed(case[:event_id]) if e['lifecycle:transition'] == 'start'),case[-1]['time:timestamp'])).total_seconds())
                if event_id>0 and event not in added:
                    difference.append((event["time:timestamp"]-case[-1]['time:timestamp']).total_seconds())
                    
        if difference != []:
            # mean = np.mean([d for d in difference if d>0])
            # median = st.median([d for d in difference if d>0])
            mean = abs(np.mean([d for d in difference]))
            median = abs(np.median([d for d in difference]))
            if mean > median:
                print('Threshold mean',mean/60, ' min')
                threshold_value = mean
            else: 
                print('Threshold median',median/60, ' min')
                threshold_value = median
            plt.plot(difference,marker='o',linewidth = 0,zorder=1)
            plt.hlines(mean,xmin = 0, xmax = len(difference),color='red',linewidth=2,zorder = 2,label= 'Mean value = '+str(mean/60)+' min')
            plt.hlines(median,xmin = 0, xmax = len(difference),color='orange',linewidth=2,zorder = 3,label= 'Median value = '+str(median/60)+' min')
            plt.legend()
            plt.savefig(os.path.join(imgpath,'thresholdView.png'))
            plt.close()
            print('Threshold Variance' , np.std(np.divide(difference,[60 for i in difference])))
        return threshold_value

    def calculate_time_diff(self, log):
        log_df = pm.convert_to_dataframe(log)
        log_df.sort_values(by=['case:concept:name', 'time:timestamp'], inplace=True)
        mean_diff = log_df.groupby('case:concept:name')['time:timestamp'].diff().mean() / pd.Timedelta(minutes=1)
        median_diff = log_df.groupby('case:concept:name')['time:timestamp'].diff().median() / pd.Timedelta(minutes=1)
        if mean_diff > median_diff:
            threshold_value = round(mean_diff,2)
            print('\nThreshold mean ',round(mean_diff,2), ' min')
        else:
            threshold_value = round(median_diff,2)
            print('\nThreshold median ',round(median_diff,2), ' min')
        return threshold_value

    def encode (self, encoding, norm='session', useAttributes=False):
        if os.path.exists("encoded.pickle"):
            print("Encoded log was read from file.")
        else:
            start = time.time()
            if self.hidepb == False: print(f"\nEncoding started at {datetime.fromtimestamp(start)}")
            encDf = pd.DataFrame([], columns = self.distinct)
            loopvar = range(len(self.sessions)) if self.hidepb == True else pb(range(len(self.sessions)))
            # sessions_df = pd.DataFrame(self.sessions)
            # sessions_df.to_csv('sessions.csv', index=False)
            if encoding == "time":
                for i in loopvar:
                    self.sessions[i].timeEncodingNew(self.distinct)
                    encDf = encDf.append(pd.Series(self.sessions[i].encoded, index = encDf.columns),ignore_index= True)
                    # encDf = pd.concat([encDf, pd.Series(self.sessions[i].encoded, index=encDf.columns)], ignore_index=True)
                    if useAttributes:
                        self.sessions[i].addAttributesMeanDF(self.attrNames)
                        for j in self.attrNames:
                            encDf.loc[i,j] = self.sessions[i].attrEncoded[j]
            if encoding == 'freq':
                for i in loopvar:
                    self.sessions[i].freqEncodingNew(self.distinct)
                    encDf = encDf.append(pd.Series(self.sessions[i].encoded, index = encDf.columns),ignore_index= True)
                    if useAttributes:
                        self.sessions[i].addAttributesMeanDF(self.attrNames)
                        for j in self.attrNames:
                            encDf.loc[i,j] = self.sessions[i].attrEncoded[j]
            if encoding == 'time' or useAttributes:
                encDf = linearEstimator(encDf,encDf.columns,self.sessions,self.distinct)
            onlyEvents = encDf.loc[:,self.distinct]
            if useAttributes:
                onlyAttr = encDf.loc[:,self.attrNames]
                onlyAttr = self.normalizeAttrNew(onlyAttr,self.attrNames)
                encDf.loc[:,self.attrNames] = onlyAttr
            if norm == 'session':
                onlyEvents = onlyEvents.div(onlyEvents.sum(axis=1),axis=0).replace(np.nan,0) #her satırındaki elemanı, o satırın toplamına bölerek normalleştirir.
            elif norm == 'absolute':
                onlyEvents = self.normalizeEvents(onlyEvents)
            encDf.loc[:,self.distinct] = onlyEvents
            self.encodedLog = encDf
            # encDf.to_csv('encoded.csv', index=False)
            with open("encoded.pickle", "wb") as f:
                pickle.dump(encDf, f)
            print(f"{self.name}: Encoding Time:", round(((time.time()-start))/60,2), 'min')

    def tracefy (self,limit):
        log = self.log[0]
        traced = []
        trace = []
        weekend = []
        for event_id, event in enumerate(log):
            if event_id == 0 or trace == []:
                trace.append(event)
            elif event['lifecycle:transition'] == 'start' and (trace[0]["time:timestamp"].hour <limit and event["time:timestamp"].   day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour <limit) or (trace[0]["time:timestamp"].hour >= limit and ((
                    (event["time:timestamp"].day == (trace[0]["time:timestamp"].day)+1) and event["time:timestamp"].hour <limit) or 
                    (event["time:timestamp"].day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour>=limit))):
                trace.append(event)
            elif event['lifecycle:transition'] == 'complete':
                starter = next((i for i,e in enumerate(reversed(trace)) if e['lifecycle:transition'] == 'start' and e['concept:name'] == event['concept:name']),None)
                if starter != None and not completed(trace,starter,event['concept:name']):
                        trace.append(event)
                elif (trace[0]["time:timestamp"].hour <limit and event["time:timestamp"].   day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour <limit) or (trace[0]["time:timestamp"].hour >= limit and (((event["time:timestamp"].day == (trace[0]["time:timestamp"].day)+1) and event["time:timestamp"].hour <limit) or (event["time:timestamp"].day == trace[0]["time:timestamp"].day and event["time:timestamp"].hour>=limit))):
                    trace.append(event)
                else:
                    traced.append(trace)
                    trace = [event]
            else:
                traced.append( trace)
                trace = [event]
        self.log = traced

    def normalizeAttrNew(self,attributes,newNames):
        shift = {i: abs(math.floor(np.min(attributes.loc[:,[i]].values.tolist()))) if np.min(attributes.loc[:,[i]].values.tolist()) <0 else 0  for i in self.attrNames}
        for i in newNames:
            attributes.loc[:,i] = attributes.loc[:,i]+shift[i]
        lower = {i: np.min(attributes.loc[:,[i]].values.tolist()) for i in self.attrNames}
        higher = {i: np.max(attributes.loc[:,[i]].values.tolist()) for i in self.attrNames}
        for i in newNames:
            attributes.loc[:,i] = (attributes.loc[:,i]-lower[i]).div(higher[i]-lower[i])
        return attributes

    def normalizeEvents(self,events):
        lower = {i: np.min(events.loc[:,[i]].values.tolist()) for i in self.distinct}
        higher = {i: np.max(events.loc[:,[i]].values.tolist()) for i in self.distinct}
        for i in self.distinct:
            events.loc[:,i] = (events.loc[:,i]-lower[i]).div(higher[i]-lower[i])
        return events.fillna(0)
    
    def findFrequency(self,max):
        absFreq = np.array([0 for i in self.distinct])
        for s in self.sessions:
            absFreq = np.sum([absFreq,s.frequency], axis=0)
        return absFreq.tolist()

    def attrCipher(self):
        values = {i:{} for i in self.attrNames}
        unique = {i:0 for i in self.attrNames}
        for s in self.sessions:
            for e in s.events:
                for a in e.attributes.values():
                    if a.value not in values[a.name].keys():
                        values[a.name][a.value] = unique[a.name]
                        unique[a.name] +=1
        return values

    def optimize_dbscan_parameters(self,encodedLog, metric):
        # Find a better eps and min_samples values start
        from sklearn.metrics import silhouette_score
        eps_values = [1.0, 0.75, 0.50, 0.25]
        min_samples_values = [len(self.distinct)*i for i in range(1,4)]
        best_eps, best_min_samples, best_silhouette_score = None, None, -1
        silhouette = []
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = skcl.DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = dbscan.fit_predict(encodedLog)
                try:
                    silhouette_avg = silhouette_score(encodedLog, labels)
                    silhouette.append(silhouette_avg)
                    # print(f"For eps={eps} and min_samples={min_samples}, the Silhouette score is {silhouette_avg}")
                except ValueError:
                    break
                
                # if silhouette_avg > best_silhouette_score or (silhouette_avg > max(silhouette)*0.9 and silhouette_avg != best_silhouette_score): #a 10% decrease in silhouette is tolerated to not create more clusters.
                if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    best_eps = eps
                    best_min_samples = min_samples
        
        print(f"The best Silhouette score is {round(best_silhouette_score,3)} for eps={best_eps} and min_samples={best_min_samples}\n")
        
        return best_eps, best_min_samples

    def cluster (self, params={"alg":"KMeans"}):
        start = time.time()
        if self.hidepb == False: print(f"\nClustering started at {datetime.fromtimestamp(start)}")
        
        if os.path.exists("encoded.pickle"):
            with open("encoded.pickle", 'rb') as f:
                encodedLog = pickle.load(f)
            encodedLog = encodedLog.values.tolist()
        else:
            encodedLog = self.encodedLog.values.tolist()   
            
        if params["alg"].lower() == "kmeans":
            params["num"] = elbow(self.encodedLog, 1, 10, os.getcwd())
            cluster = TTestKMeans2(params["num"], encodedLog)
            if self.hidepb == True: print("SSE : ", cluster.inertia_)
            print(f"{self.name}: Clustering Time:", round((time.time()-start),2), 'sec')
            return cluster.predict(encodedLog),cluster.cluster_centers_
        elif params["alg"].lower() == "dbscan":
            best_eps = 0.25
            best_min_samples = 25
            metric = 'euclidean'
            # best_eps, best_min_samples = self.optimize_dbscan_parameters(encodedLog, metric)
            params['eps'] = best_eps
            params['min_samples'] = best_min_samples
            params['metric'] = metric
            cluster = skcl.DBSCAN(min_samples=best_min_samples, eps = best_eps, metric=metric).fit(encodedLog)
            # centers, y_pred = skcl.dbscan(encodedLog, eps=best_eps, metric="minkowski",algorithm="ball_tree")
            y_pred = cluster.labels_
            centers = calcCenters(y_pred, encodedLog)
            print(f"{self.name}: Clustering Time:", round((time.time()-start)/60,2), 'min')
            if "assignNoisy" in params and params["assignNoisy"] == True:
                y_pred, centers = assignNoisyPoints(y_pred, encodedLog, centers)
            return y_pred, centers, best_eps, best_min_samples
        elif params["alg"].lower() == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            params["num"] = elbow(encodedLog, 1, 10, os.getcwd())
            cluster = AgglomerativeClustering(n_clusters=params["num"], linkage='ward')
            y_pred = cluster.fit_predict(encodedLog)
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(encodedLog, y_pred)
            print("Silhouette Score:", silhouette_avg)
            print(f"{self.name}: Clustering Time:", round((time.time()-start),2), 'sec')
            return y_pred
    
    def exportSubP (self,y_pred,centers,path,SubDir):
        path = path+'/Clusters/'+SubDir.replace(".xes","")
        try:
            os.makedirs(path,exist_ok=True)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path, "and started to export sub-processes")
            for filename in os.listdir(path): #empty the sub process folder
                file_path = os.path.join(path, filename)
                os.remove(file_path)
            
            frames = [[] for i in range(max(y_pred)+1)]
            attrNames = get_event_attributes(self.log)
            if os.path.exists("enumerated_sessions.pickle"):
                attrNames.insert(0,'cluster')
                # caseid_check = 'Yes'
            if 'caseid' in attrNames: attrNames.remove('caseid')
            for i,s in enumerate(self.sessions):
                frames[y_pred[i]].extend(s.export(attrNames,i))

            for i in range(max(y_pred)+1):
                newFile = "cluster"+str(i)+'.xes'
                log = pd.concat(frames[i],ignore_index=True)
                # if caseid_check != 'Yes':
                #     log['case'] = log['caseid'] #change caseids in the abstracted log with the originals
                #     log = log.drop('caseid', axis=1)
                log = log.loc[:,~log.columns.duplicated()]
                log = pm.format_dataframe(log,case_id='case',activity_key='concept:name',timestamp_key='time:timestamp')
                log = pm.convert_to_event_log(log)
                fileNameAndPath=os.path.join(path,newFile)
                pm.write_xes(log,fileNameAndPath)
            if self.hidepb == True: print("Sessions were exported")
        # merge_subprocesses(fileNameAndPath)
  
    def convertLog(self, centers, y_pred, path, exportSes):
        start = time.time()
        if os.path.exists("enumerated_sessions.pickle"): 
            with open("enumerated_sessions.pickle", 'rb') as f:
                loopvar = pickle.load(f)
            caseid_check = 'Yes'
            frames = []
            log = pd.DataFrame()
        else:   
            caseid_check = 'No'
            frames = []
            log = pd.DataFrame()
            if self.hidepb == False: print(f"\nConversion started at {datetime.fromtimestamp(start)}")
            loopvar = enumerate(self.sessions) if self.hidepb == True else pb(enumerate(self.sessions), prefix = "Converting sessions | ")
            with open("enumerated_sessions.pickle", "wb") as f:
                pickle.dump(enumerate(self.sessions), f)
        
        for i,s in loopvar:
            abstracted = s.convertSession(centers[y_pred[i]], y_pred[i], self.distinct, self.attrNames)
            frames.append(abstracted)
        log = pd.concat(frames,ignore_index=True)
        if caseid_check != 'Yes' and 'caseid'in log.columns:
            #print('caseid silindi')
            log['case'] = log['caseid'] #change caseids in the abstracted log with the originals
            log = log.drop('caseid', axis=1)
        log = pm.format_dataframe(log,case_id='case',activity_key='concept:name',timestamp_key='time:timestamp')
        log = pm.convert_to_event_log(log)

        print('\nLow-level events are converted to high-level events')
        from pm4py.objects.log.obj import EventLog
        newlog = EventLog()
        caseidlist = []
        for trace in pb(log):
            caseid = int(float(str(trace.attributes['concept:name'])))
            if caseid not in caseidlist:
                caseidlist.append(caseid)
                newlog.append(trace)
            else:
                for newtrace in newlog:
                    if int(float(str(newtrace.attributes['concept:name']))) == caseid:
                        for event in trace:                                
                            newtrace.append(event)

        from pm4py.objects.log.util import sorting
        log = sorting.sort_timestamp(newlog)
        
        try:
            os.makedirs(path,exist_ok=True)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            exportname = os.path.join(path,os.path.basename(self.name).replace(".xes", "") + "_AbstractedLog.xes")
            pm.objects.log.exporter.xes.exporter.apply(log, exportname, parameters = {"show_progress_bar": False})
            #pm.write_xes(log,os.path.join(path,os.path.basename(self.name).replace(".xes", "") + "_AbstractedLog.xes"))

        if exportSes:
            self.exportSubP(y_pred,centers,path,os.path.basename(self.name))
        print(f"\n{self.name}: Conversion Time:",time.time()-start)   

    def renameTasks (self,names,datapath):
        if names != {}:
            try:
                for ic,case in enumerate(self.log):
                    for ie,event in enumerate(case):
                            if event['cluster'] in names:
                                self.log[ic][ie]['concept:name'] = names[event['cluster']]
                pm.write_xes(self.log,os.path.join(datapath,self.name+".xes"))
            except:
                print('Event-log not suitable for renaming')
        else:
            print('No name given')
            
    def betterPlotting (self,centers, y_pred, path, params, mode = "linear"):
        distinct = self.distinct
        attrValues = self.attrNames
        if (len(centers[0])> len(distinct)):
            attr = [c[len(distinct):] for c in centers]
            attrNames = attrValues
        else:
            attr = []
            attrNames = []
        centers = np.array([c[range(len(distinct))] for c in centers])

        #Normalizzazione solo per plotting
        if any(i>1 for c in centers for i in c):
            for i,c in enumerate(centers):
                lower =  min(c)
                if lower <0 :
                    lower = abs(lower)
                    centers[i] = [j+lower for j in c]
            centers = centers/centers.sum(axis=1,keepdims = True)
        newCenters= []
        newDistinct = []
        for i,e in enumerate(distinct):
            drop = True
            for c in centers:
                if c[i] >= 0.01: drop = False
            if not drop: newDistinct.append(e)
        for i,c in enumerate(centers):
            cn = []
            for e in newDistinct:
                cn.append(c[distinct.index(e)])
            if attr != []: cn = [*cn, *attr[i]]
            newCenters.append(cn)
        if attr != []: columns = newDistinct + attrNames
        else: columns = newDistinct
        df1 = pd.DataFrame(newCenters,index=range(max(y_pred)+1),columns =columns)
        logmin = 0.001
        fig, ax = plt.subplots()
        fig.set_size_inches((len(columns),len(newCenters)))
        
        if mode == "linear":
            sns.heatmap(df1, cmap="YlOrRd", linewidths=.5,xticklabels=True, yticklabels= True, ax = ax)
        else:
            sns.heatmap(df1, cmap="YlOrRd", linewidths=.5,norm =LogNorm(), vmin= max(centers.min().min(),logmin),xticklabels=True, yticklabels= True, ax= ax)
        if attr != []:
            ax.vlines([len(newDistinct)], *ax.get_ylim())
        params['threshold (min)'] = self.threshold_value
        ax.set_title(params)
        fig.savefig(path+".png",bbox_inches='tight') 
        ax.clear()