import pm4py as pm
import time
import os
from datetime import datetime

from pm4py.algo.evaluation import algorithm


def split_log(log, training_rate):
    dataframe = pm.convert_to_dataframe(log)
    train_df, test_df = pm.split_train_test(dataframe, training_rate)
    trainingLog = pm.convert_to_event_log(train_df)
    testLog = pm.convert_to_event_log(train_df)
    return trainingLog, testLog

def computeQuality(log_directory, a_net, a_im, a_fm):        
    main_directory = os.path.join(os.path.dirname(os.path.dirname(log_directory)))
    sub_process_directory = (os.path.join(main_directory, ".AbstractedLogs","Clusters",os.path.splitext(os.path.basename(log_directory))[0]))
    number_of_clusters = len([f for f in os.listdir(sub_process_directory) if os.path.isfile(os.path.join(sub_process_directory, f))])
    if not os.path.isfile(os.path.join(main_directory,"QualityMetrics.csv")):
        with open(os.path.join(main_directory,"QualityMetrics.csv"), "w") as file:
            file.write("Date, Log, O_Fitness, A_Fitness, O_Precision, A_Precision, O_Generalization, A_Generalization, O_Simplicity, A_Simplicity, A_Average, Cluster")

    #Fitness is computed with testlog, remaining is computed with entire log
    
    print(f"\nQuality metrics are evaluated for the log by spliting training and test at {datetime.fromtimestamp(time.time())}")
    log = pm.read_xes(log_directory)
    trainingLog, testLog = split_log(log, training_rate=0.8)
    # Original Log Computation
    # net_t, im_t, fm_t = pm.discover_petri_net_inductive(trainingLog, noise_threshold=0.2)
    # print('Fitness is computing from test log')
    # q_o = algorithm.apply(testLog, net_t, im_t, fm_t)
    # fitness_o = round(q_o['fitness']['average_trace_fitness'],3)
    # print('Other quality metrics are computing from entire log')
    # net, im, fm = pm.discover_petri_net_inductive(log,  noise_threshold=0.2)
    # q_o = algorithm.apply(log, net, im, fm)
    # precision_o = round(q_o['precision'],3)
    # generalization_o = round(q_o['generalization'],3)
    # simplicity_o = round(q_o['simplicity'],3)
    fitness_o = 0.992
    precision_o = 0.115
    generalization_o = 0.963
    simplicity_o = 0.652
    
    file_path, file_name = os.path.split(log_directory)
    # new_file_name = file_name.replace("_trainingtesting.xes", "_evaluation.xes")
    new_file_name = file_name.replace("_trainingtesting.xes", ".xes")
    log_directory = os.path.join(file_path, new_file_name)
    log = pm.read_xes(log_directory)

    # Abstracted Log Computation
    print('Quality metrics are computing for the merged log')
    q_a = algorithm.apply(log, a_net, a_im, a_fm) #qa: quality metrics for abstracted log
    fitness_a = round(q_a['fitness']['average_trace_fitness'],3)
    precision_a = round(q_a['precision'],3)
    generalization_a = round(q_a['generalization'],3)
    simplicity_a = round(q_a['simplicity'],3)
    average_a = round(q_a['metricsAverageWeight'],3)
    
    print(f"\nOriginal Log:\nFitness={fitness_o}, Precision={precision_o}, Generalization={generalization_o}, Simplicity={simplicity_o}")
    print(f"\nAbstracted Log:\nFitness={fitness_a}, Precision={precision_a}, Generalization={generalization_a}, Simplicity={simplicity_a}")

    with open(os.path.join(main_directory,"QualityMetrics.csv"), "a") as file:
        basics = f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}, {os.path.basename(log_directory)}"
        line = f"{fitness_o},{fitness_a},"
        line += f"{precision_o}, {precision_a},"
        line += f"{generalization_o},  {generalization_a},"
        line += f"{simplicity_o}, {simplicity_a}"
        file.write(f"{basics}, {line}, {average_a}, {number_of_clusters}")
 

