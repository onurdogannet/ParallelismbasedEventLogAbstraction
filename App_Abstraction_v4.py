from src.Abstraction.Abstractor_v4 import Abstractor_v4

import logging
from logging import log
import pm4py

import os
import platform
import sys
import time
import datetime
import winsound

# For this code to run, you need to graphviz to 0.15.0

if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, filename='src/Abstraction/eventabstraction.log', filemode='w', format='%(levelname)s - %(message)s')
    log(logging.DEBUG, f"OS version:\t{platform.platform()}\nPython version:\t{sys.version}\nPM4Py version:\t{pm4py.__version__}\n")

    #Parameters for K-Means
    # params = {"alg":"kmeans", "assignNoisy" : True}
    #Parameters for dbscan
    params = {"alg":"dbscan", "assignNoisy" : True}
    #Parameters for dbscan
    # params = {"alg":"agglomerative", "assignNoisy" : True}

    # Create output csv for time logging
    if not os.path.isfile("DurationLogging.csv"):
        with open("DurationLogging.csv", "w") as file:
            print("Creating DurationLogging.csv")
            file.write("Date, Log, Algo_duration")

#########################################

    filename = 'iot' #"simulatedLog4" 
    algo_start = time.time()
    
    abstractor = Abstractor_v4(fileName = os.path.join("Logs", str(filename + '.xes')), split = False, attrNames = [], parallel = False)
    
    #Encoding based on frequency of occurrences 
    abstractor.encode('freq')

    #Encoding based on time (slower, but necessary when the duration of the activities is more important than the sole occurrence)
    #abstractor.encode('time')

    # imgpath = os.getcwd()
    # epsEstimate(abstractor.encodedLog,imgpath)
    # minPointsEstimate(abstractor.encodedLog, 0.07,imgpath)
    # elbow(abstractor.encodedLog,10,40,imgpath)

    #create clusters and compute centroids
    y_pred, centers, best_eps, best_min_samples = abstractor.cluster(params)
    # y_pred, centers = abstractor.cluster(params)
    # y_pred = abstractor.cluster(params)

    #Visualize the clusters and centroids on a heatmap: A file "test.png" is created in the current directory
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    abstractor.betterPlotting(centers,y_pred,f"heatmap_{filename}_{current_date}",params)
    

    #Abstraction of the event log. The abstract event log will be created in "working directory/.AbstractedLogs/
    #The logs related to each cluster (and high-level activity) is created in the subdirectory "Clusters" of said folder
    abstractor.convertLog(centers, y_pred, path = '.AbstractedLogs', exportSes = True)

    #Close algorithm - log duration
    print(f"Total algorithm time for {filename}: {round((time.time() - algo_start) / 60, 2)} minutes")
    with open("DurationLogging.csv", "a") as file:
        file.write(f"\n{time.time()},{filename},{time.time() - algo_start}")

    winsound.Beep(500, 1000)
