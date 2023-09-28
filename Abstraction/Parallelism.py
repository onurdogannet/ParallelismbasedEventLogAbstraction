import pm4py as pm
import time
from datetime import datetime
from collections import Counter
import itertools
import os
import pickle
from progressbar import progressbar as pb
from src.Abstraction.Session import Session
from src.Abstraction.Event import Event
from pm4py.stats import get_event_attributes

# log = pm.read_xes(r"C:\Users\onurb\OneDrive - BAKIRÇAY ÜNİVERSİTESİ\Projeler\Web_Task_Mining\Abstraction_v4\Logs\simulatedLog3.xes")

def get_session_rules(log, threshold=0.8):
    start = time.time()
    print(f'Generating session rules started at {datetime.fromtimestamp(start)}')
    from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
    fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
    dfg = fp_log['dfg']
    parallel = fp_log['parallel']
    #print("fp_log['sequence']", len(fp_log['sequence']),fp_log['sequence'])
    sequence = set()
 
    for pair in dfg:
        seq_value = dfg[pair]
        reversed_pair = tuple(reversed(pair))
        if reversed_pair in dfg:
            reversed_seq_value = dfg[reversed_pair]
        else:
            reversed_seq_value = 0
        dependency = (seq_value-reversed_seq_value)/(seq_value+reversed_seq_value+1)
        
        if dependency > threshold:
            sequence.add(pair)
            # print(pair, seq_value, reversed_pair,reversed_seq_value, dependency)
    # print("onur sequence:", sorted(sequence))
    
    seq_to_remove = []
    for pair in sequence:
        if pair in parallel:
            seq_to_remove.append(pair)
    # print("onur seq_to_remove:", sorted(seq_to_remove))
    
    if len(seq_to_remove) > 0:
         for seq in seq_to_remove:
            if seq in sequence:
                sequence.remove(seq)
    # print("onur sequence2:", sorted(sequence))
    return sequence

#get_session_rules(log, 0.8)

def get_session_rules_old(log): # splits gateways in different session
    
    start = time.time()
    print(f'Generating session rules started at {datetime.fromtimestamp(start)}')
    from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
    fp_log = footprints_discovery.apply(log, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
    sequence = heuristic_sequence_activities(log, threshold=0.8)
    parallel = fp_log['parallel']
    left_counts = Counter([tup[0] for tup in sequence])
    right_counts = Counter([tup[1] for tup in sequence])
    possible_split_activities = [elem for elem, count in left_counts.items() if count > 1]
    possible_join_activities = [elem for elem, count in right_counts.items() if count > 1]

    #Split activities
    possible_parallelism = []
    invalid_pairs = []
    seq_to_remove = []
    for j in possible_split_activities:
        possible_parallelism = [k[1] for k in sequence if k[0] == j]
        # print('possible_parallelism',possible_parallelism)
        parallelism_pairs = itertools.combinations(possible_parallelism, 2)
        for pair in parallelism_pairs:
            # print('pair',pair)
            if pair not in parallel:
                invalid_pairs = []
                invalid_pairs.append(pair)
                # print('not parallel invalid_pairs',invalid_pairs)
                for invalid_pair in invalid_pairs:
                    if right_counts[invalid_pair[0]] > 1:
                        seq_to_remove.append((j, invalid_pair[0]))
                        # print('seq_to_remove',seq_to_remove)
                    else: #right_counts[invalid_pair[1]] > 1:
                        seq_to_remove.append((j, invalid_pair[1]))
                        # print('seq_to_remove',seq_to_remove)
                    #else: print('Hata')          
            elif pair in parallel:
                invalid_pairs = []
                invalid_pairs.append(pair)
                # print('parallel invalid_pairs',invalid_pairs)
                for invalid_pair in invalid_pairs:
                    for k in sequence:
                        if k[0] == j and k[1] in invalid_pair:
                            seq_to_remove.append(k)
                            # print('seq_to_remove',seq_to_remove)
    #Join activities
    possible_parallelism = []
    invalid_pairs = []
    for j in possible_join_activities:
        possible_parallelism = [k[0] for k in sequence if k[1] == j]
        parallelism_pairs = itertools.combinations(possible_parallelism, 2)
        for pair in parallelism_pairs:
            if pair not in parallel:
                invalid_pairs = []
                invalid_pairs.append(pair)
                for invalid_pair in invalid_pairs:
                    if left_counts[invalid_pair[0]] > 1:
                        seq_to_remove.append((invalid_pair[0], j))
                    else: #left_counts[invalid_pair[1]] > 1:
                        seq_to_remove.append((invalid_pair[1], j))
                    #else: print('Hata')
            
            elif pair in parallel:
               invalid_pairs = []
               invalid_pairs.append(pair)
               for invalid_pair in invalid_pairs:
                   for k in sequence:
                       if k[0] in invalid_pair and k[1] == j:
                           seq_to_remove.append(k)
                   
    if len(seq_to_remove) > 0:
         for seq in seq_to_remove:
            if seq in sequence:
                sequence.remove(seq)
                        
    # print('SEQUENCE', sequence)
    return sequence


# sequence = get_session_rules(log)

def convert_caseid_tonumeric(log):
    # save caseids (string) to a list
    caseids=[]
    for i,trace in enumerate(log):
        if log[i].attributes not in caseids:
            caseids.append(log[i].attributes['concept:name'])
    
    # create a dict for current (string) and new caseids (numeric)
    id_dict = {}
    for val in caseids:
        if val not in id_dict:
            id_dict[val] = len(id_dict) + 1
    
    # replace current caseids with numeric caseids
    new_caseids = [id_dict[i] for i in caseids]
    for i,trace in enumerate(log):
        log[i].attributes['concept:name'] = str(new_caseids[i])

def divide_traces(log, sequence, time_threshold):
    start = time.time()
    print(f'\nCreating sessions started at {datetime.fromtimestamp(start)}.')
    
    #add caseid as an attribute to events
    for trace in log:
        if isinstance(trace.attributes['concept:name'], str):
            convert_caseid_tonumeric(log)
        caseid = trace.attributes['concept:name']
        for event in trace:
            event['caseid'] = caseid   
  
    divided_traces = []
    for trace in pb(log):
        divided_traces.append([trace[0]])
        remaining_trace = trace[1:]
        for i in range(len(trace)-1):
            for j in range(len(divided_traces)-1, -1, -1):
                if ((divided_traces[j][-1]['concept:name'], remaining_trace[0]['concept:name'])) in sequence:
                    time_difference = remaining_trace[0]['time:timestamp'] - divided_traces[j][-1]['time:timestamp']
                    if time_difference.total_seconds()/60 < time_threshold:
                        divided_traces[j].append(remaining_trace[0])
                        remaining_trace = remaining_trace[1:]
                        break
                    else:
                        divided_traces.append([remaining_trace[0]])
                        remaining_trace = remaining_trace[1:]
                        break
                elif j == 0:
                    divided_traces.append([remaining_trace[0]])
                    remaining_trace = remaining_trace[1:]
                    break 
         
    distinct = []
    for event in log:
        for act in event:
            distinct.append(act['concept:name'])
    distinct = list(set(distinct))
    distinct.sort()
    
    attrNames = get_event_attributes(log)

    sessions = []
    for caseid, case in enumerate(divided_traces):
        session = []
        for j,event in enumerate(case):
            session.append(Event(event,attrNames))
        sessions.append(Session(caseid, session))
    # print("\nElapsed time:", round((time.time()-start),2), 'sec')
    
    # Bu değişkenler time threshold'a göre değişir sadece. Aynı time threshold değeri için bir kere hespalayıp dosyaya yazdırıyorum.   
    with open(r"distinct.pickle", 'wb') as f:
        pickle.dump(distinct, f)   
    with open(r"sessions.pickle", 'wb') as f:
        pickle.dump(sessions, f)
    
    return sessions, distinct  
    
# sessions, distinct = divide_traces(log, sequence, 10)    
