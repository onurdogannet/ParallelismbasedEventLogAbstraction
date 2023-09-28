import numpy as np 
import math
import pm4py as pm
from datetime import datetime
import os as os
from sklearn import cluster as skcl
import sklearn as sk 
from sklearn.neighbors import NearestNeighbors,BallTree
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import time

#import Event
from src.Abstraction.Attribute import Attribute
from progressbar import progressbar as pb

pd.set_option("display.max_rows", None, "display.max_columns", None) 

def importLog (fileName):
    log = pm.objects.log.importer.xes.importer.apply(fileName, parameters = {"show_progress_bar": False})
    #log = pm.read_xes(fileName)
    return log


def getAttributes (event, attributes):
    al = {}
    for a in attributes:
        try :
            al[a] = Attribute(a,event[a])
        except:
            al[a] = Attribute(a,None)
    return al

def returnState (e, where="lifecycle"):
    if where == "lifecycle" and "lifecycle:transition" in e:
        return e["lifecycle:transition"]
    elif where=="Activity" and "START" in e["Activity"]:
        return "start"
    else:
        return "complete"

def findDistinct (session):
    event_list = []
    for e in session:
        if e.name not in event_list:
            event_list.append(e.name)
    return event_list

def concatName (events,ind):
    name = ""
    for i,n in enumerate(ind):
        name += events[n]
        if i != len(ind)-1:
            name += " & "
    return name

def normalize(encoded,limit=-1,factor = 1):
    if limit != -1:
        esum = sum(encoded[0:limit])*factor
    else:
        esum = sum(encoded)*factor
    for i,e in enumerate(encoded[0:limit]):
        if e != 0 :
            encoded[i] = e/esum
    return encoded

def normalizeAttr (attr):
    asum = np.sum(attr,axis=0)
    return list(np.divide(attr,asum))

def distance (center, enc):
    difference = np.diff([center,enc], axis = 0)
    return np.sqrt(np.sum(np.power(difference,2)))

def allDistances (centers, enc, y_pred):
    alldist = []
    for i,e in enumerate(enc):
        if y_pred[i] == -1:
            for y in range(max(y_pred)+1) :
                dict1 = {"from": y, "dist": distance(centers[y],e) , "session":i}
                alldist.append(dict1)
    return pd.DataFrame(alldist,columns= ["from","dist","session"])

def assignNoisyPoints(y_pred,enc,centers):
    print("Noisy points are assigned to the closest cluster.")
    alldist = allDistances(centers, enc, y_pred)
    for i in range(len(enc)) :
        if y_pred[i] == -1:
            allMe = alldist[alldist["session"]==i]
            minInd = allMe[["dist"]].idxmin()
            y_pred[i] = alldist.at[int(minInd),"from"]
    return y_pred, calcCenters(y_pred, enc)

def calcCentersNew(y_pred, enc):
    centers = [[] for i in range(max(y_pred)+1)]
    clusters = [[]for i in range(max(y_pred)+1)]
    for i in range(len(enc)):
        if(y_pred[i]>= 0 ):
            clusters[y_pred[i]].append(enc.loc[i,:].to_dict())
            clusters.append(enc[i,:].to_dict())
    for i,_ in enumerate(clusters):
        cluster = pd.DataFrame.from_records(clusters[i])
        centers[i] = cluster.mean(axis=0).to_dict()
    return centers

def calcCenters(y_pred, enc):
    centers = [[] for i in range(max(y_pred)+1)]
    for v in range(max(y_pred)+1):
        cluster = []
        indexPos = [ i for i in range(len(y_pred)) if y_pred[i] == v ]
        for i in indexPos :
            cluster.append(enc[i])
        centers[v] = np.mean(cluster, axis=0)
    return centers

def groupSessions (y_pred,enc):
    clusters = [[]for i in range(max(y_pred)+1)]
    for i in range(len(enc)):
        if(y_pred[i]>= 0 ):
            clusters[y_pred[i]].append(enc.loc[i,:].to_dict())
            clusters.append(enc[i,:].to_dict())
    for i,_ in enumerate(clusters):
        cluster = pd.DataFrame.from_records(clusters[i])
    return clusters


def printCluster(centers, distinct):
    indexes = np.argsort([c[range(len(distinct))] for c in centers], axis = 1)
    for i,ind in enumerate(indexes):
        for j in ind[-1:]:
                    print("cluster",i," : ",distinct[j], centers[i][j])

def checkCluster(distinct, labels, sessions):
    '''
    for i,l in enumerate(labels):
        print("label : ",l)
        for e in sessions[i].events:
            if sessions[i].encoded[distinct.index(e.name)]!= 0 :
                print(e.name,sessions[i].encoded[distinct.index(e.name)])
    '''
    for s in sessions:
        for i,e in enumerate(s.encoded):
            if e >1:
                print(distinct[i])

def elbow (X, nfrom, nto,imgpath):
    sse = []
    for k in range(nfrom, nto+1):
        kmeans = skcl.KMeans(n_clusters=k)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    # print('elbow sse:',sse)
    sse_change_ratio = []
    for i in range(len(sse)-1):
        if sse[i+1] > 1e-3:
            ratio = sse[i] / sse[i+1]
        else: ratio = 1
        sse_change_ratio.append(ratio)
        # print(sse_change_ratio)
    opt_cluster_number = sse_change_ratio.index(max(sse_change_ratio)) + nfrom + 1
    # print(sse_change_ratio, opt_cluster_number)
    print('elbow cluster number:', opt_cluster_number)
    fig = plt.figure(figsize=(15, 5))
    # plt.plot(range(nfrom, nto+1), sse)
    plt.plot(range(nfrom+1, nto+1), sse_change_ratio)
    plt.ylim(0, max(sse_change_ratio) * 1.1)
    plt.xlabel('Cluster number')
    plt.title('SSE Change Ratio')
    # plt.ylabel('SSE Value')
    plt.ylabel('Change Ratio')
    plt.grid(True)
    plt.xticks(np.arange(nfrom, nto+1, 1.0))
    # plt.title('Elbow SSE Curve')
    # fig.savefig(imgpath+"/elbow_sse_"+str(nfrom)+"-"+str(nto)+".png")
    plt.title('Elbow SSE Change Curve')
    fig.savefig(imgpath+"/elbow_sse_change_ratio"+str(nfrom)+"-"+str(nto)+".png")
    plt.close()
    return opt_cluster_number

def multirun(num, enc, runs):
    best = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
    which = -1
    for i in range(runs):
        kmeans = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
        if best.inertia_ > kmeans.inertia_ :
            best = kmeans
            which = i 
    #print("run numero : ", which)
    return best

def TTestKMeans (num, enc):
    runs = 0
    results = []
    means = []
    inertias = []

    while runs < 50:
        results.append(skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc))
        inertias.append(results[runs].inertia_)
        means.append( np.mean(inertias))
        runs +=1
    plt.plot(means,label = 'media')
    plt.plot(inertias,label = 'inertia')
    plt.legend(loc="lower right", title="Legend Title", frameon=False)
    plt.savefig("./img/TTest.png")
    plt.close()
    return results[np.argmin(inertias)]

def confIntMean(a, conf=0.95):
     mean, sem, m = np.mean(a), stats.sem(a), stats.t.ppf((1+conf)/2., len(a)-1)
     return mean - m*sem, mean + m*sem

def TTestKMeans2 (num, enc):
    runs = 0
    results = []
    means = []
    inertias = []
    stop = False
    left = []
    right = []
    while stop == False:
        results.append(skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc))
        inertias.append(results[runs].inertia_)
        means.append(np.mean(inertias))
        runs +=1
        if runs>1:
            l,r = confIntMean(inertias)
            left.append(l)
            right.append(r)
        else:
            left = [means[0]]
            right = [means[0]]
        if runs>5 and abs(left[-1]-right[-1]) < means[-1]/100 :
            stop = True
            # print(min(inertias), np.argmin(inertias))
    left[0] = min(inertias)
    right[0] = max(inertias)
    plt.fill_between(range(runs),left,right,color= "C2", alpha= 0.2)
    plt.plot(means, 'C2',label = 'Media SSE')
    plt.plot(inertias,'C1',label = 'SSE')
    plt.plot([min(inertias) for i in inertias], 'C3--',label= "Min SSE")
    plt.annotate(round(min(inertias),3), xy=(runs/2.5,min(inertias)), xytext=(runs/2.5,min(inertias)-25))
    plt.xticks(np.arange(0, runs, step=5))
    plt.legend(loc="upper right", title="Legenda", frameon=False)
    plt.savefig("./TTest.png")
    plt.close()
    return results[np.argmin(inertias)]

def multirunMeans(num, enc, runs):
    best = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
    for i in range(runs):
        kmeans = skcl.KMeans(n_clusters=num,init = "k-means++").fit(enc)
        if best.inertia_ > kmeans.inertia_ :
            which = i 
    return which

def calculate_kn_distance(X,k):
    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(distance(X[i],X[j]))
        eucl_dist.sort()
        kn_distance.append(round(eucl_dist[k],2))
    return kn_distance

def epsEstimate (enc, imgpath):
    # enc = enc.values.tolist()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(enc)
    _,indices = nbrs.kneighbors(enc)
    eps_dist = []
    for i,e in enumerate(enc): 
        eps_dist.append(distance(e,enc[indices[i][-1]]))
    eps_dist.sort()
    
    tot = len(eps_dist)
    fig,ax1 = plt.subplots(figsize=(10,5))
    _, bins, _ = ax1.hist(eps_dist[:-math.floor(tot/5)],bins= 50)
    perc = [0 for i in bins]
    for i,v in enumerate(bins):
        for e in eps_dist:
            if e<v:
                perc[i]+=1
    for i,_ in enumerate(perc):
        perc[i] = (perc[i]/tot)*100
    ax1.set_xticks(bins)
    ax1.xaxis.set_tick_params(rotation=90)
    ax1.set_title("Eps Estimate ")
    ax1.set_ylabel('n')
    ax1.set_xlabel('Eps')
    ax2 = ax1.twinx()
    ax2.plot(bins,perc,color= 'red')
    ax2.set_yticks(np.linspace(0,100,num = 11))
    ax2.set_ylabel('percentage')
    ax1.grid(color= 'C0',alpha = 0.5)
    ax2.grid(color= 'red',alpha = 0.5)
    ax2.hlines(50,xmin=min(bins),xmax = max(bins),linestyle = 'dashed',color = 'k')
    ax2.tick_params(axis='y', colors='red')
    ax1.tick_params(axis='y', colors='C0')
    fig.tight_layout()
    fig.savefig(os.path.join(imgpath,"epsEstimate.png"))
    fig.clear()

def minPointsEstimate (enc, eps, imgpath):
    tree = BallTree(np.array(enc))
    allNgbr = []
    allNgbr.append(tree.query_radius(enc, eps, count_only=True))
    _, bins, _ = plt.hist (allNgbr, bins = 45)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(bins, rotation = 90 )
    # plt.title("MinPts Estimate "+ encoding)
    plt.title("MinPts Estimate ")
    plt.ylabel('Number of sessions')
    plt.xlabel('Number of neighbors')
    plt.tight_layout()
    plt.savefig(os.path.join(imgpath,"minptsEstimate.png"))
    plt.close()

def findRuns (enc, value, clusters):
    num = []
    i = 1
    while i <= value+1:
        nsum = 0
        for j in range(5):
            print(nsum)
            nsum += multirunMeans(clusters,enc,i)
        num.append(nsum/5)
        i += 10
    print(num)
    plt.plot(np.linspace(1,value+1),num)
    # plt.savefig(os.path.join(imgpath,"provaKmeans.png"))
    plt.close()

def completed (lista, index, name):
    for i in lista[index+1:]:
        try:
            if i.name == name and i.state == 'complete':
                return True
        except:
            if i['concept:name'] == name and i['lifecycle:transition'] == 'complete':
                return True
    return False

def linearEstimator(encEvents,events,sessions,distinct):
    start = time.time()
    withNan = {e:[] for e in events}
    noNan = {e:[] for e in events}
    midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for e in events:
        for i,j in encEvents.iterrows():
            if str(j[e]) == 'nan':
                timestamp = None
                if e in distinct:
                    for es in sessions[i].events:
                        if es.name == e:
                            timestamp = es.timestamp
                else:
                    timestamp = sessions[i].events[0].timestamp
                j['weekday'] = int(timestamp.strftime('%w'))
                j['day'] = (timestamp.replace(tzinfo=None) - datetime(1970,1,1)).days
                j['time'] = (timestamp.replace(tzinfo=None)-midnight).seconds
                withNan[e].append(j)
            elif j[e] != np.nan:
                timestamp = None
                if e in distinct:
                    for es in sessions[i].events:
                        if es.name == e:
                            timestamp = es.timestamp
                else:
                    timestamp = sessions[i].events[0].timestamp
                if timestamp != None:
                    j['weekday'] = int(timestamp.strftime('%w'))
                    j['day'] = (timestamp.replace(tzinfo=None) - datetime(1970,1,1)).days
                    j['time'] = (timestamp.replace(tzinfo=None)-midnight).seconds
                else:
                    j['weekday'] = -1
                    j['day'] = 0
                    j['time'] = 0
                noNan[e].append(j)
        if withNan[e] != []:
            df = pd.DataFrame(noNan[e])
            imputer = sk.impute.SimpleImputer(strategy='mean', missing_values=np.nan)
            y = df[[e]]
            y_train = y[:-30]
            y_test = y[-30:]
            x = df.drop([e],axis =1)
            cols = x.columns
            x = pd.DataFrame(data = imputer.fit_transform(x), columns =cols)
            x_train =x[:-30]
            x_test = x[-30:]
            regressor = sk.linear_model.LinearRegression().fit(x_train,y_train)
            toPred = pd.DataFrame(withNan[e]).drop([e],axis=1)
            cols = toPred.columns
            toPred = pd.DataFrame(data = imputer.transform(toPred), columns =cols)
            new = regressor.predict(toPred)
            y_pred =  regressor.predict(x_test)
            error = sk.metrics.mean_squared_error(y_test,y_pred,squared = False)
            print('test',y_test)
            print('pred',y_pred)
            print('RMSE: %.2f' % error)
            flat = new
            dfNew = pd.DataFrame(withNan[e])
            dfNew[e] = flat
            encEvents = encEvents.combine_first(dfNew.drop(['time','day','weekday'],axis=1))
    encEvents.fillna(0)
    print('Estimation Time :', time.time()-start)
    return encEvents
    
def started (event, case, id):
    ind = 0
    while case[ind] != event and ind<len(case) :
        ind += 1
    if ind==len(case):
        print("Event not in trace")
    starter = next((e for e in case[id+1:ind] if e['lifecycle:transition']== 'start' and e['concept:name']==event['concept:name']),None)
    if starter != None:
        return True
    else:
        return False

def safediv (attr,freq):
    if isinstance(attr, float):
        return attr/freq
    else:
        return attr