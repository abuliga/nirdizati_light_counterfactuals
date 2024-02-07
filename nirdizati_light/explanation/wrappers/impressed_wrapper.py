import argparse
from joblib import Parallel, delayed
import random
import os
import networkx as nx
import numpy as np
import pandas as pd
import pm4py
from paretoset import paretoset
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.log.obj import EventLog
from nir
dizati_light.explanations.utils.Auto_IMPID import AutoStepWise_PPD
from nirdizati_light.explanations.utils.IMIPD import VariantSelection, create_pattern_attributes, Trace_graph_generator, Pattern_extension,\
    plot_only_pattern, Single_Pattern_Extender

# Parameters for the pareto optimization
logs = [log for log in os.listdir('../experiments/cf4impressed/impressed_datasets') if log.endswith('.csv')]
for log in logs:
    log = 'cf_bpi2012full_complex_direct_pattern_simple_trace_dice_206258_diversity_0.5_oneshot.csv'
    #log = 'full_complex_direct_pattern_filtered.csv'
   # log = 'cf_bpi2012full_complex_direct_pattern_simple_trace_dice_206258_diversity_0.5_oneshot_updated.csv'
    #log = 'cf_bpi2012full_direct_follow_simple_trace_dice_206258_diversity_0.5_oneshot.csv'
    #log = 'cf_bpic2012_O_ACCEPTED-COMPLETEfull_simple_trace_dice_206258_diversity_0.5_oneshot.csv'




    if not os.path.exists(output_path):
        os.makedirs(output_path)
def impressed_wrapper(log_path,output_path,discovery_type,case_id,activity,timestamp,outcome,outcome_type,delta_time,
                      max_gap,max_extension_step,factual_outcome,likelihood,encoding):
    # Load the log
    pareto_features = ['Outcome_Interest', 'Frequency_Interest', 'likelihood']
    pareto_sense = ['max', 'max', 'max']
    df = pd.read_csv(log_path,encoding='latin-1').dropna(axis=1)
    df = df[[case_id, activity, timestamp, outcome, likelihood]]
    df[activity] = df[activity].astype('string')
    df[activity] = df[activity].str.replace("_", "-")
    df[timestamp] = pd.to_datetime(df[timestamp],format='mixed')
    df[case_id] = df[case_id].astype('string')
    outcomes = df[outcome].unique()
    if outcome_type == 'binary':
        for i, out in enumerate(outcomes):
            df.loc[df[outcome] == str(out), outcome] = i
        df[outcome] = df[outcome].astype('uint8')
    elif outcome_type == 'numerical':
        df[outcome] = df[outcome].astype('float32')

    color_codes = ["#" + ''.join([random.choice('000123456789ABCDEF') for i in range(6)])
                   for j in range(len(df[activity].unique()))]

    color_act_dict = dict()
    counter = 0
    for act in df[activity].unique():
        color_act_dict[act] = color_codes[counter]
        counter += 1
    color_act_dict['start'] = 'k'
    color_act_dict['end'] = 'k'

    patient_data = df[[case_id, likelihood, outcome]]
    patient_data.drop_duplicates(subset=[case_id], inplace=True)
    patient_data.loc[:, list(df[activity].unique())] = 0
    selected_variants = VariantSelection(df, case_id, activity, timestamp)
    for case in selected_variants["case:concept:name"].unique():
        Other_cases = \
            selected_variants.loc[selected_variants["case:concept:name"] == case, 'case:CaseIDs'].tolist()[0]
        trace = df.loc[df[case_id] == case, activity].tolist()
        for act in np.unique(trace):
            Number_of_act = trace.count(act)
            for Ocase in Other_cases:
                patient_data.loc[patient_data[case_id] == Ocase, act] = Number_of_act

    activity_attributes = create_pattern_attributes(patient_data, outcome, factual_outcome,
                                                    list(df[activity].unique()), outcome_type)

    Objectives_attributes = activity_attributes[pareto_features]
    mask = paretoset(Objectives_attributes, sense=pareto_sense)
    paretoset_activities = activity_attributes[mask]
    paretoset_activities = activity_attributes
    paretoset_activities.to_csv(output_path + '/paretoset_1.csv', index=False)
    All_pareto_patterns = paretoset_activities['patterns'].tolist()

    if discovery_type == 'interactive':
        # ask the user to select the pattern of interest
        print("Please select the pattern of interest from the following list:")
        print(paretoset_activities['patterns'].tolist())
        Core_activity = input("Enter the name of the pattern of interest: ")

        all_pattern_dictionary = dict()
        all_extended_patterns = dict()
        All_Pareto_front = dict()
        EventLog_graphs = dict()
        Patterns_Dictionary = dict()
        all_variants = dict()
        filtered_cases = df.loc[df[activity] == Core_activity, case_id]
        filtered_main_data = df[df[case_id].isin(filtered_cases)]
        # Keep only variants and its frequency
        timestamp = timestamp
        filtered_main_data = pm4py.format_dataframe(filtered_main_data, case_id=case_id,
                                                    activity_key=activity,
                                                    timestamp_key=timestamp)
        filtered_main_log = pm4py.convert_to_event_log(filtered_main_data)
        variants = variants_filter.get_variants(filtered_main_log)
        pp_log = EventLog()
        pp_log._attributes = filtered_main_log.attributes
        for i, k in enumerate(variants):
            variants[k][0].attributes['VariantFrequency'] = len(variants[k])
            Case_ids = []
            for trace in variants[k]:
                Case_ids.append(trace.attributes['concept:name'])
            variants[k][0].attributes['CaseIDs'] = Case_ids
            pp_log.append(variants[k][0])

        selected_variants = pm4py.convert_to_dataframe(pp_log)
        all_variants[Core_activity] = selected_variants
        timestamp = 'time:timestamp'
        for case in selected_variants[case_id].unique():
            case_data = selected_variants[selected_variants[case_id] == case]
            if case not in EventLog_graphs.keys():
                Trace_graph = Trace_graph_generator(selected_variants, patient_data, Core_activity, delta_time,
                                                    case, color_act_dict,
                                                    case_id, activity, timestamp)

                EventLog_graphs[case] = Trace_graph.copy()
            else:
                Trace_graph = EventLog_graphs[case].copy()

            Patterns_Dictionary = Pattern_extension(case_data, Trace_graph, Core_activity,
                                                    case_id, Patterns_Dictionary, max_gap)

        patient_data.loc[:, list(Patterns_Dictionary.keys())] = 0
        for PID in Patterns_Dictionary:
            for CaseID in np.unique(Patterns_Dictionary[PID]['Instances']['case']):
                variant_frequency_case = Patterns_Dictionary[PID]['Instances']['case'].count(CaseID)
                Other_cases = \
                    selected_variants.loc[selected_variants[case_id] == CaseID, 'case:CaseIDs'].tolist()[
                        0]
                for Ocase in Other_cases:
                    patient_data.loc[patient_data[case_id] == Ocase, PID] = variant_frequency_case

        pattern_attributes = create_pattern_attributes(patient_data, outcome,
                                                       factual_outcome, list(Patterns_Dictionary.keys()), outcome_type)

        Objectives_attributes = pattern_attributes[pareto_features]
        mask = paretoset(Objectives_attributes, sense=pareto_sense)
        paretoset_patterns = pattern_attributes
        paretoset_patterns = pattern_attributes[mask]
        All_Pareto_front[Core_activity] = dict()
        All_Pareto_front[Core_activity]['dict'] = Patterns_Dictionary
        All_Pareto_front[Core_activity]['variants'] = selected_variants
        all_pattern_dictionary.update(Patterns_Dictionary)
        paretoset_patterns_to_save = paretoset_patterns.copy()
        All_pareto_patterns.extend(paretoset_patterns['patterns'].tolist())
        import itertools
        pattern_activities = [list(nx.get_node_attributes(
            Patterns_Dictionary[m]['pattern'], 'value').values()) for m in list(paretoset_patterns['patterns'])]
        pattern_relations = [list(nx.get_edge_attributes(
            Patterns_Dictionary[m]['pattern'], 'eventually').values()) for m in
                             list(paretoset_patterns['patterns'])]
        pattern_activities = [list(itertools.chain(*e)) for e in zip(pattern_activities, pattern_relations)]
        paretoset_patterns_to_save['activities'] = pattern_activities
        paretoset_patterns_to_save.to_csv(output_path + '/paretoset_2.csv', index=False)

        #paretoset_patterns.to_csv(output_path + '/paretoset_2.csv', index=False)
        # parallelize the plotting of the patterns
        Parallel(n_jobs=6)(delayed(plot_only_pattern)(Patterns_Dictionary, row['patterns'], color_act_dict, output_path)
                           for ticker, row in paretoset_patterns.iterrows())

        # extend the patterns
        continue_extending = input("Type 1 if you want to continue extending patterns or 0 to stop: ")
        continue_extending = int(continue_extending)
        counter = 3
        while continue_extending == 1:
            # ask the user to select the pattern of interest
            print("Please select the pattern of interest from the following list:")
            print(paretoset_patterns['patterns'].tolist())
            Core_pattern = input("Enter to the name of the pattern of interest: ")
            while any(nx.get_edge_attributes(Patterns_Dictionary[Core_pattern]['pattern'], 'eventually').values()):
                print("Patterns including eventually relations are not supported yet for extension")
                Core_pattern = input("Enter to the name of the pattern of interest: ")
                if Core_pattern == '-1':
                    break
            all_extended_patterns.update(Patterns_Dictionary)
            all_extended_patterns, Patterns_Dictionary, patient_data = Single_Pattern_Extender(
                all_extended_patterns,
                Core_pattern,
                patient_data, EventLog_graphs,
                all_variants, max_gap)

            pattern_attributes = create_pattern_attributes(patient_data, outcome,
                                                           factual_outcome, list(Patterns_Dictionary.keys()),
                                                           outcome_type)
            Objectives_attributes = pattern_attributes[pareto_features]
            mask = paretoset(Objectives_attributes, sense=pareto_sense)
            paretoset_patterns = pattern_attributes
            paretoset_patterns = pattern_attributes[mask]
            paretoset_patterns_to_save = paretoset_patterns.copy()
            import itertools

            pattern_activities = [list(nx.get_node_attributes(
                Patterns_Dictionary[m]['pattern'], 'value').values()) for m in
                                  list(Patterns_Dictionary.keys())]
            pattern_relations = [list(nx.get_edge_attributes(
                Patterns_Dictionary[m]['pattern'], 'eventually').values()) for m in
                                 list(Patterns_Dictionary.keys())]
            pattern_activities = [list(itertools.chain(*e)) for e in zip(pattern_activities, pattern_relations)]
            paretoset_patterns_to_save['activities'] = pattern_activities
            paretoset_patterns_to_save.to_csv(output_path + '/paretoset_%s.csv' % counter, index=False)
            All_pareto_patterns.extend(paretoset_patterns['patterns'].tolist())
            counter += 1
            # parallelize the plotting of the patterns
            Parallel(n_jobs=6)(delayed(plot_only_pattern)(Patterns_Dictionary, row['patterns'], color_act_dict, output_path)
                               for ticker, row in paretoset_patterns.iterrows())

            continue_extending = input("Enter 1 if you want to continue extending patterns or 0 to stop: ")
            continue_extending = int(continue_extending)
        if encoding:
            Encoded_patterns = patient_data[All_pareto_patterns]
            Encoded_patterns.loc[:, case_id] = patient_data[case_id]
            Encoded_patterns.loc[:, outcome] = patient_data[outcome]
            Encoded_patterns.to_csv(output_path + '/EncodedPatterns_InteractiveMode.csv', index=False)
    if discovery_type == 'auto':
        train_X, test_X = AutoStepWise_PPD(max_extension_step, max_gap,
                                           testing_percentage, df, patient_data, case_id,
                                           activity, outcome, outcome_type, timestamp,
                                           pareto_features, pareto_sense, delta_time,
                                           color_act_dict, output_path,
                                           factual_outcome)
        from sklearn import tree
        X = np.array(train_X)
        Y = patient_data[outcome].toarray()
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, Y)
        train_X.to_csv(output_path + "/training_encoded_log.csv", index=False)
        test_X.to_csv(output_path + "/testing_encoded_log.csv", index=False)
        #TODO: Add decision tree training here with rule extraction tomorrow
        #TODO: Plot the frequency curves for all generated patterns