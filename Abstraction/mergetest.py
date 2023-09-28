import pm4py as pm
import os
import time
from datetime import datetime
import random
import warnings
warnings.simplefilter(action='ignore')
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to, remove_transition, remove_place, remove_unconnected_components
# from src.Abstraction.computeQuality import computeQuality
import winsound

original_log_name = 'PLGLog13'
output_format = 'pdf'
noise_threshold = 0.2

# For consecutiveness
# abstracted_log_directory = "C:\\Users\\onurb\\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\\Projeler\\Web_Task_Mining\\Abstraction_v1\\.AbstractedLogs"
# sub_process_directory = os.path.join("C:\\Users\\onurb\\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\\Projeler\\Web_Task_Mining\\Abstraction_v1\\.AbstractedLogs\\Clusters",original_log_name)
# original_log_directory = "C:\\Users\\onurb\\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\\Projeler\\Web_Task_Mining\\Abstraction_v1\\Logs"

# For parallelism
abstracted_log_directory = "C:\\Users\\onurb\\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\\Projeler\\Web_Task_Mining\\Abstraction_v4\\.AbstractedLogs"
sub_process_directory = os.path.join("C:\\Users\\onurb\\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\\Projeler\\Web_Task_Mining\\Abstraction_v4\\.AbstractedLogs\\Clusters",original_log_name)
original_log_directory = "C:\\Users\\onurb\\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\\Projeler\\Web_Task_Mining\\Abstraction_v4\\Logs"
log_directory = os.path.join(original_log_directory, str(original_log_name+".xes"))

a_net, a_im, a_fm = pm.read_pnml(os.path.join(abstracted_log_directory, str(original_log_name+"_AbstractedModel.pnml")))
pm.save_vis_petri_net(a_net, a_im, a_fm, os.path.join(abstracted_log_directory, str(original_log_name+"_AbstractedModel.pdf")))
# pm.view_petri_net(a_net, a_im, a_fm, output_format)
    target_transitions = []
    source_transitions = []
    for arc in net.arcs:
        if arc.source.name == 'source':
            target_transitions.append(arc.target)
        elif arc.target.name == 'sink':
            source_transitions.append(arc.source)
            
    for i,target in enumerate(target_transitions):
        added_place = PetriNet.Place(str(in_places[0])+"_"+str(i))
        print('added_place:',added_place)
        net.places.add(added_place)
        add_arc_from_to(added_place, target, net)
        for place in net.places:
            if place.name == 'source':
                source_place = place
        remove_place(net, source_place) # buraya kadar source place silindi, başka place eklendi ve arc tanımlandı.

        for inplace in in_places:
            transition_in = PetriNet.Transition("tau_" + str(inplace))
            net.transitions.add(transition_in)
            add_arc_from_to(transition_in, added_place, net)
            add_arc_from_to(inplace, transition_in, net) # source place yerine eklenen place ile tau arasındaki arc tanımlandı.
    
    for i,source in enumerate(source_transitions):
        added_place = PetriNet.Place(str(out_places[0])+"_"+str(i))
        print('added_place:',added_place)
        net.places.add(added_place)
        add_arc_from_to(source, added_place, net)
        for place in net.places:
            if place.name == 'sink':
                target_place = place
        remove_place(net, target_place) # buraya kadar sink place silindi, başka place eklendi ve arc tanımlandı.
        
        for outplace in out_places:
            transition_out = PetriNet.Transition("tau_" + str(outplace))
            net.transitions.add(transition_out)
            add_arc_from_to(added_place, transition_out, net)
            add_arc_from_to(transition_out, outplace, net)

def add_invisible_transitions(net, im, fm, in_places, out_places, a):
    for i, place in enumerate(net.places):
        if place.name == "source":
            place.name = str(a.label)+"_"+str(random.randint(1, 1000000))
            ip = place
        elif place.name == "sink":
            place.name = str(a.label)+"_"+str(random.randint(1, 1000000))
            fp = place
        place.name = str(a.label)+"_"+str(random.randint(1, 1000000))
        # print(place.name)
        net.place = PetriNet.Place(place)
        
    for inplace in in_places:
        transition_in = PetriNet.Transition("tau_" + str(random.randint(1, 1000000)))
        net.transitions.add(transition_in)
        add_arc_from_to(transition_in, ip, net)
        add_arc_from_to(inplace, transition_in, net) # source place yerine eklenen place ile tau arasındaki arc tanımlandı.

    for outplace in out_places:
        transition_out = PetriNet.Transition("tau_" + str(random.randint(1, 1000000)))
        net.transitions.add(transition_out)
        add_arc_from_to(fp, transition_out, net)
        add_arc_from_to(transition_out, outplace, net)
    
print(f'\nSubprocesses are started to merge at {datetime.fromtimestamp(time.time())} for {original_log_name}')
high_level_activities = list({t for t in a_net.transitions if (t.label is not None and "-" not in t.name)})
# invisible_transitions = {t for t in a_net.transitions if t.label is None}
for i, a in enumerate(high_level_activities): 
    # cluster_number = a.label[-1] # for simulatedlogs ending with a number B1, B2 ...
    import re
    high_level_activity = a.label.split()[-1]  # Son kelimeyi alıyoruz
    cluster_number = re.findall(r'\d+', high_level_activity[::-1])
    cluster_number = [int(s[::-1]) for s in cluster_number[::-1]]
    cluster_number = cluster_number[-1]
    # cluster_number = int(''.join(filter(str.isdigit, high_level_activity)))  # Sadece sayısal karakterleri alıyoruz
    print(f"Cluster {cluster_number}, High level activity \"{a.label}\" is merging at {datetime.fromtimestamp(time.time())}.")
    
    in_places = []
    out_places = []
    for arc in a.in_arcs:
        in_places.append(arc.source)
    for arc in a.out_arcs:
        out_places.append(arc.target)
    
    sub_log = pm.read_xes(os.path.join(sub_process_directory, str('cluster'+str(cluster_number)+'.xes')))
    net, im, fm = pm.discover_petri_net_inductive(sub_log, noise_threshold)
    # if a.label in ['Activity C3', 'Activity U2', 'Activity D1']:
    # pm.view_petri_net(net, im, fm, output_format)
    add_invisible_transitions(net, im, fm, in_places, out_places, a)

    a_net.transitions.update(net.transitions)
    a_net.places.update(net.places)
    a_net.arcs.update(net.arcs)
    remove_transition(a_net, a)
    # if a.label in ['Activity C3', 'Activity U2', 'Activity D1']:
    # pm.view_petri_net(a_net, a_im, a_fm, output_format)
    # pm.write_pnml(a_net, a_im, a_fm, os.path.join(abstracted_log_directory, str(original_log_name+"_MergedModel.pnml")))
    
    # deger = input("Bir değer girin: ")
    # if deger == "1":
    #     break

# pm.view_petri_net(a_net, a_im, a_fm, output_format)
pm.write_pnml(a_net, a_im, a_fm, os.path.join(abstracted_log_directory, str(original_log_name+"_MergedModel.pnml")))
pm.save_vis_petri_net(a_net, a_im, a_fm, os.path.join(abstracted_log_directory, str(original_log_name+"_MergedModel.pdf")))

# from pm4py.algo.analysis.woflan import algorithm as woflan
# is_sound = woflan.apply(a_net, a_im, a_fm)

winsound.Beep(500, 1000)

##############################################################
start = time.time()
from pm4py.algo.evaluation import algorithm
main_directory = os.path.join(os.path.dirname(os.path.dirname(log_directory)))
sub_process_directory = (os.path.join(main_directory, ".AbstractedLogs","Clusters",os.path.splitext(os.path.basename(log_directory))[0]))
number_of_clusters = len([f for f in os.listdir(sub_process_directory) if os.path.isfile(os.path.join(sub_process_directory, f))])
if not os.path.isfile(os.path.join(main_directory,"QualityMetrics.csv")):
    with open(os.path.join(main_directory,"QualityMetrics.csv"), "w") as file:
        file.write("Date, Log, O_Fitness, A_Fitness, O_Precision, A_Precision, O_Generalization, A_Generalization, O_Simplicity, A_Simplicity, A_Average, Cluster")

#Fitness is computed with testlog, remaining is computed with entire log

print(f"\nQuality metrics are evaluated for the log by spliting training and test at {datetime.fromtimestamp(time.time())}")
log = pm.read_xes(log_directory)

# Original Log Computation

# dataframe = pm.convert_to_dataframe(log)
# training_rate=0.8
# train_df, test_df = pm.split_train_test(dataframe, training_rate)
# trainingLog = pm.convert_to_event_log(train_df)
# testLog = pm.convert_to_event_log(train_df)

# net_t, im_t, fm_t = pm.read_pnml(os.path.join(abstracted_log_directory, str(original_log_name+"_OriginalModel.pnml")))
# pm.save_vis_petri_net(net_t, im_t, fm_t, os.path.join(abstracted_log_directory, str(original_log_name+"_OriginalModel.pdf")))

# net_t, im_t, fm_t = pm.discover_petri_net_inductive(trainingLog, noise_threshold=0.2)
# print('Fitness is computing from test log')
# q_o = algorithm.apply(testLog, net_t, im_t, fm_t)
# fitness_o = round(q_o['fitness']['average_trace_fitness'],3)
# print(q_o)
# print('\nQuality metrics are computing for the original model\n')
# net, im, fm = pm.discover_petri_net_inductive(log,  noise_threshold=0.2)
# q_o = algorithm.apply(log, net_t, im_t, fm_t)
# fitness_o = round(q_o['fitness']['average_trace_fitness'],3)
# precision_o = round(q_o['precision'],3)
# generalization_o = round(q_o['generalization'],3)
# simplicity_o = round(q_o['simplicity'],3)
fitness_o = 0.416
precision_o = 0.008
generalization_o = 0.945
simplicity_o = 0.677
# print(q_o)

# print(f"\nOriginal Log:\nFitness={fitness_o}, Precision={precision_o}, Generalization={generalization_o}, Simplicity={simplicity_o}")

# from pm4py.algo.analysis.woflan import algorithm as woflan
# is_sound = woflan.apply(net_t, im_t, fm_t)

# precision = pm.precision_token_based_replay(log, net_t, im_t, fm_t)
# print('precision:', precision)

# precision = pm.precision_alignments(log, net_t, im_t, fm_t)
# print('precision:', precision)


# file_path, file_name = os.path.split(log_directory)
# new_file_name = file_name.replace("_trainingtesting.xes", "_evaluation.xes")
# # new_file_name = file_name.replace("_trainingtesting.xes", ".xes")
# log_directory = os.path.join(file_path, new_file_name)
# log = pm.read_xes(log_directory)

# Abstracted Log Computation

print('Quality metrics are computing for the merged model')
q_a = algorithm.apply(log, a_net, a_im, a_fm) #qa: quality metrics for abstracted log
fitness_a = round(q_a['fitness']['average_trace_fitness'],3)
precision_a = round(q_a['precision'],3)
generalization_a = round(q_a['generalization'],3)
simplicity_a = round(q_a['simplicity'],3)
average_a = round(q_a['metricsAverageWeight'],3)
# precision_a = pm.precision_alignments(log, net_t, im_t, fm_t)


print(f"\nAbstracted Log:\nFitness={fitness_a}, Precision={precision_a}, Generalization={generalization_a}, Simplicity={simplicity_a}")

# precision = pm.precision_token_based_replay(log, a_net, a_im, a_fm)
# print('precision:', precision)

precision = pm.precision_alignments(log, a_net, a_im, a_fm)
print('precision:', precision)


with open(os.path.join(main_directory,"QualityMetrics.csv"), "a") as file:
    basics = f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}, {os.path.basename(log_directory)}"
    line = f"{fitness_o},{fitness_a},"
    line += f"{precision_o}, {precision_a},"
    line += f"{generalization_o},  {generalization_a},"
    line += f"{simplicity_o}, {simplicity_a}"
    file.write(f"{basics}, {line}, {average_a}, {number_of_clusters}")

winsound.Beep(500, 1000)
print("Elapsed time: ",(round((time.time() - start)/60,2))," min")