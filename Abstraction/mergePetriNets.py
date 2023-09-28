import pm4py as pm
import os
import time
import winsound
from datetime import datetime

from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to, remove_transition
from src.Abstraction.computeQuality import computeQuality


def add_invisible_transitions(net, im, fm, in_place, out_place):
    ip = [place for place in im][0]
    fp = [place for place in fm][0]

    transition = PetriNet.Transition("tau")
    net.transitions.add(transition)
    add_arc_from_to(transition, ip, net)
    add_arc_from_to(in_place, transition, net)
    
    transition = PetriNet.Transition("tau")
    net.transitions.add(transition)
    add_arc_from_to(fp, transition, net)
    add_arc_from_to(transition, out_place, net)
    
    
def merge_subprocesses(fileNameAndPath):
    
    noise_threshold = 0.2
    
    original_log_name = os.path.basename(os.path.dirname(fileNameAndPath))
    abstracted_logs_folder = os.path.dirname(os.path.dirname(os.path.dirname(fileNameAndPath)))
    sub_process_directory = os.path.dirname(fileNameAndPath)

    print(f'Abstracted model is started to create at {datetime.fromtimestamp(time.time())}.')
    abstracted_log = pm.read_xes(os.path.join(abstracted_logs_folder, str(original_log_name+"_AbstractedLog.xes")))
    a_net, a_im, a_fm = pm.discover_petri_net_inductive(abstracted_log, noise_threshold)
        
    # transitions_to_remove = []
    # for t in a_net.transitions:
    #     if "skip" in t.name:
    #         transitions_to_remove.append(t)
    # for t in transitions_to_remove:
    #     for a in t.out_arcs:
    #         if str(a.target) != "sink":
    #             remove_transition(a_net, t)
                
    pm.view_petri_net(a_net, a_im, a_fm, "pdf")
    pm.write_pnml(a_net, a_im, a_fm, os.path.join(abstracted_logs_folder, str(original_log_name+"_AbstractedModel.pnml")))
    pm.save_vis_petri_net(a_net, a_im, a_fm, os.path.join(abstracted_logs_folder, str(original_log_name+"_AbstractedModel.pnml")))
    
    print(f'\nSubprocesses are started to merge at {datetime.fromtimestamp(time.time())}')
    high_level_activities = {t for t in a_net.transitions if t.label is not None}
    
    for i, a in enumerate(high_level_activities):
        print(f"Cluster {i}, High level activity: {a.label}  is merging at {datetime.fromtimestamp(time.time())}.")
        for arc in a.in_arcs:
            in_place = arc.source
        for arc in a.out_arcs:
            out_place = arc.target
        
        cluster_number = a.label[-1]
        sub_log = pm.read_xes(os.path.join(sub_process_directory, str('cluster'+cluster_number+'.xes')))
        net, im, fm = pm.discover_petri_net_inductive(sub_log, noise_threshold)
        add_invisible_transitions(net, im, fm, in_place, out_place)
        
        a_net.transitions.update(net.transitions)
        a_net.places.update(net.places)
        a_net.arcs.update(net.arcs)
        remove_transition(a_net, a)
    pm.view_petri_net(a_net, a_im, a_fm, "pdf")
    pm.write_pnml(a_net, a_im, a_fm, os.path.join(abstracted_logs_folder, str(original_log_name+"_MergedModel.pnml")))
    pm.save_vis_petri_net(a_net, a_im, a_fm, os.path.join(abstracted_logs_folder, str(original_log_name+"_MergedModel.pnml")))
    
    app_abstractor_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(fileNameAndPath))))
    log_directory = os.path.join(app_abstractor_directory, "Logs", str(original_log_name + '.xes'))
    winsound.Beep(500, 1000)
    computeQuality(log_directory, a_net, a_im, a_fm)