import logging
import warnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.evaluation.common import evaluate_classifier
from nirdizati_light.explanation.common import ExplainerType, explain
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.log.common import get_log
from nirdizati_light.predictive_model.common import ClassificationMethods, get_tensor
from nirdizati_light.predictive_model.predictive_model import PredictiveModel, drop_columns
from nirdizati_light.explanation.wrappers.dice_wrapper import perform_model_analysis

import random
from declare4py.declare4py import Declare4Py
from declare4py.enums import TraceState
from dataset_confs import DatasetConfs


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def run_simple_pipeline(CONF=None, dataset_name=None):
    random.seed(CONF['seed'])
    np.random.seed(CONF['seed'])
    dataset_confs = DatasetConfs(dataset_name=dataset_name, where_is_the_file=CONF['data'])

    logger.debug('LOAD DATA')
    log = get_log(filepath=CONF['data'])

    logger.debug('ENCODE DATA')
    encodings = [EncodingType.SIMPLE_TRACE.value,EncodingType.SIMPLE_TRACE.value]
    for encoding in encodings:
        CONF['feature_selection'] = encoding
        encoder, full_df = get_encoded_df(log=log, CONF=CONF)
        # HAVE SEPARATE ENCODER FOR LORELEY IN ORDER TO SPLIT THE PREFIX and have the same evaluation
        logger.debug('TRAIN PREDICTIVE MODEL')
        # change label values
        #full_df.iloc[:, -1] -= 1

        # split in train, val, test
        train_size = CONF['train_val_test_split'][0]
        val_size = CONF['train_val_test_split'][1]
        test_size = CONF['train_val_test_split'][2]
        if train_size + val_size + test_size != 1.0:
            raise Exception('Train-val-test split does not sum up to 1')
        train_df,val_df,test_df = np.split(full_df,[int(train_size*len(full_df)), int((train_size+val_size)*len(full_df))])

        predictive_model = PredictiveModel(CONF, CONF['predictive_model'], train_df, val_df)
        predictive_model.model, predictive_model.config = retrieve_best_model(
            predictive_model,
            CONF['predictive_model'],
            max_evaluations=CONF['hyperparameter_optimisation_epochs'],
            target=CONF['hyperparameter_optimisation_target']
        )

        logger.debug('EVALUATE PREDICTIVE MODEL')
        if predictive_model.model_type is ClassificationMethods.LSTM.value:
            probabilities = predictive_model.model.predict(get_tensor(CONF, drop_columns(test_df)))
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
        elif predictive_model.model_type not in (ClassificationMethods.LSTM.value):
            predicted = predictive_model.model.predict(drop_columns(test_df))
            scores = predictive_model.model.predict_proba(drop_columns(test_df))[:, 1]

        actual = test_df['label']
        if predictive_model.model_type is ClassificationMethods.LSTM.value:
            actual = np.array(actual.to_list())

        initial_result = evaluate_classifier(actual, predicted, scores)
        logger.debug('COMPUTE EXPLANATION')
        if CONF['explanator'] is ExplainerType.DICE.value:
            cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
            full_df = pd.concat([train_df, val_df, test_df])
            cf_dataset.loc[len(cf_dataset)] = 0
            model_path = '../experiments/process_models/process_models_new'
            setup = {'genetic':['baseline']}
            heuristics=['heuristic_2']
            if 'sepsis' in dataset_name:
                support = 0.99
            else:
                support = 0.9
            model_path = '../experiments/process_models/process_models_new'
            model_path = model_path + '_' + str(support) + '/'
            conformant_traces,number_of_constraints, conformant_traces_ratio=\
                perform_model_analysis(model_path, dataset, CONF, encoder, full_df, support, log,dataset_confs)
            test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]

            full_df = full_df[full_df['trace_id'].isin(conformant_traces)]
            test_df_correct = test_df_correct[test_df_correct['trace_id'].isin(conformant_traces)]

            explain(CONF, predictive_model, encoder=encoder, cf_df=full_df.iloc[:, 1:],
                                   query_instances=test_df_correct.iloc[:, 1:],
                                   method=method, df=full_df.iloc[:, 1:], optimization='baseline', support=support,
                                   timestamp_col_name=[*dataset_confs.timestamp_col.values()][0],
                                   model_path=model_path,random_seed=CONF['seed']
                    ,adapted=CONF['adapted'],filtering=False
                    )
    logger.info('RESULT')
    logger.info('INITIAL', initial_result)
    logger.info('Done, cheers!')

    return {'initial_result', initial_result, 'predictive_model.config', predictive_model.config}


if __name__ == '__main__':
    dataset_list = {
        'BPIC15_1_f2':[15,20,25,30],
        'BPIC15_2_f2':[15,20,25,30],
        'BPIC15_3_f2':[15,20,25,30],
        'BPIC15_4_f2':[15,20,25,30],
        'BPIC15_5_f2':[15,20,25,30],
       'bpic2012_O_ACCEPTED-COMPLETE':[20,25,30,35],
       'bpic2012_O_CANCELLED-COMPLETE':[20,25,30,35],
       'bpic2012_O_DECLINED-COMPLETE':[20,25,30,35],
        'sepsis_cases_1':[7,10,13,16],
       'sepsis_cases_2':[7,10,13,16],
        'sepsis_cases_4':[7,10,13,16],
        'legal_complaints':[4,6,8,11],
        'BPIC17_O_ACCEPTED':[20,25,30,35],
        'BPIC17_O_CANCELLED':[20,25,30,35],
        'BPIC17_O_DECLINED':[20,25,30,35],

    }
    methods = {'genetic':False,'genetic_conformance':True,'multi_objective_genetic':True,
               'multi_objective_genetic_adapted':True}
    for dataset,prefix_lengths in dataset_list.items():
        for prefix in prefix_lengths:
            for method,optimization in methods.items():
                CONF = {  # This contains the configuration for the run
                    'data': os.path.join('..','datasets',dataset, 'full.xes'),
                    'train_val_test_split': [0.7, 0.15, 0.15],
                    'output': os.path.join('..', 'output_data'),
                    'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                    'prefix_length': prefix,
                    'padding': True,  # TODO, why use of padding?
                    'feature_selection': EncodingType.SIMPLE.value,
                    'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                    'attribute_encoding': EncodingTypeAttribute.LABEL.value,  # LABEL, ONEHOT
                    'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                    'predictive_model': ClassificationMethods.RANDOM_FOREST.value,  # RANDOM_FOREST, LSTM, PERCEPTRON
                    'explanator': ExplainerType.DICE.value,  # SHAP, LRP, ICE, DICE
                    'threshold': 13,
                    'top_k': 10,
                    'hyperparameter_optimisation': False,  # TODO, this parameter is not used
                    'hyperparameter_optimisation_target': HyperoptTarget.AUC.value,
                    'hyperparameter_optimisation_epochs': 20,
                    'time_encoding': TimeEncodingType.NONE.value,
                    'target_event': None,
                    'seed': 666,
                    'method': method,
                    'adapted': optimization,
                }
                run_simple_pipeline(CONF=CONF, dataset_name=dataset)
