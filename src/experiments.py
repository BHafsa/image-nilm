import warnings, os, pickle, yaml
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.config.optimizer.set_jit(True)



# u_name = os.uname()[1]
#
# if u_name == 'user-Legion-T530-28ICB':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# if u_name == 'ultron':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sys import argv
from time import time
from copy import deepcopy

# NILMTK Imports
from nilmtk.api import API
from disaggregate.Im2Seq import Im2Seq
# from nilmtk_contrib.disaggregate import FHMMExact

################### SETTINGS ###################

DATA_DIR = './data/'

settings = {
    'metrics': ['mae', 'nde', 'f1score', 'r2score'],
    'batch_size': 64,
    'epochs': 1,
    'sampling_interval': 10,
    'sequence_length': 120, 
    'chunksize': 200,
    'DROP_ALL_NANS': True,
    'experiment': {}
}

try:
    with open("out/I2S-settings.yaml", "w") as file:
        documents = yaml.dump(settings, file)

except FileNotFoundError:
    os.mkdir('out/')
    with open("out/I2S-settings.yaml", "w") as file:
        documents = yaml.dump(settings, file)

disaggregation_methods = {
    # OLD-SCHOOL NILM Algorithms
    # 'FHMMExact': FHMMExact({}),
    # baseline DNN Approaches
    # 'DAE': DAE({
    #         'n_epochs': settings['epochs'], 
    #         'batch_size': settings['batch_size'], 
    #         'sequence_length': settings['sequence_length']
    #     }
    #     ),
    # Im2Seq DNN Approaches
    'CNN_MTF': Im2Seq({
        'n_epochs': settings['epochs'],
        'batch_size': settings['batch_size'],
        'img_method': 'mtf',
        'sequence_length': settings['sequence_length'],
    }),

    'CNN_GASF': Im2Seq({
        'n_epochs': settings['epochs'],
        'batch_size': settings['batch_size'],
        'img_method': 'gasf',
        'sequence_length': settings['sequence_length'],
    }),

    'CNN_RP': Im2Seq({
        'n_epochs': settings['epochs'],
        'batch_size': settings['batch_size'],
        'img_method': 'rp',
        'sequence_length': settings['sequence_length'],
    })

}

################### DEFINE EXPERIMENTS ###################


basic_experiments = {


    # 'SynD-1': {
    #     'data_set': 'SynD',
    #     'house': 1,
    #     'ac_type': ['active'],

    #     'f1': {
    #         'train_dates': ['2019-10-01', '2020-01-01'],
    #         'test_dates': ['2020-02-01', '2020-02-15'],
    #     },

    #     'appliances': [
    #         'fridge',
    #         'dish washer',
    #         'washing machine',
    #         'microwave'
    #     ]
    # },


     'UK-2': {
        'data_set': 'ukdale2',
        'house': 1,
        'ac_type': 'active',

        'f1': {
            'train_dates': ['2013-07-25', '2013-07-26'],
            'test_dates': ['2013-10-15', '2013-10-16'],
        },

        'appliances': [
            'fridge',
            # 'dish washer',
            # 'washing machine',
            # 'microwave'
        ]
    },
    
    # 'REFIT-5': {
    #     'data_set': 'REFIT',
    #     'house': 6,
    #     'ac_type': 'active',

    #     'f1': {
    #         'train_dates': ['2014-07-25', '2014-10-25'],
    #         'test_dates': ['2014-11-15', '2014-11-30'],
    #     },

    #     'appliances': [
    #         'fridge',
    #         'dish washer',
    #         'washing machine',
    #         'microwave'
    #     ]
    # }
}

################### CONDUCT EXPERIMENTS ###################

for ex_name, experiment in basic_experiments.items():

    print('Doing {} now...'.format(ex_name))

    experiment_f1 = {
        'power': {
            'mains': experiment['ac_type'],
            'appliance': experiment['ac_type']
        },
        'sample_rate': settings['sampling_interval'],
        'appliances': experiment['appliances'],

        'chunksize': settings['chunksize'],
        'DROP_ALL_NANS': settings['DROP_ALL_NANS'],

        'methods': disaggregation_methods,
        'train': {
            'datasets': {
                '{}'.format(experiment['data_set']): {
                    'path': '{}{}.h5'.format(DATA_DIR, experiment['data_set']),
                    'buildings': {
                        experiment['house']: {
                            'start_time': experiment['f1']['train_dates'][0],
                            'end_time': experiment['f1']['train_dates'][1]
                        }
                    }
                }
            }
        },
        'test': {
            'datasets': {
                '{}'.format(experiment['data_set']): {
                    'path': '{}{}.h5'.format(DATA_DIR, experiment['data_set']),
                    'buildings': {
                        experiment['house']: {
                            'start_time': experiment['f1']['test_dates'][0],
                            'end_time': experiment['f1']['test_dates'][1]
                        }
                    }
                }
            },
            'metrics': settings['metrics']
        }
    }

    ################### START ###################

    start = time()
    # Conduct experiment in NILMTK
    api_results_f1 = API(experiment_f1)

    error_df_f1 = api_results_f1.errors
    error_keys_df_f1 = api_results_f1.errors_keys

    # Save results in Pickle file.
    df_dict = {
        'error_keys': api_results_f1.errors_keys,
        'errors': api_results_f1.errors,

        'train_mains': api_results_f1.train_mains,
        'train_submeters': api_results_f1.train_submeters,

        'test_mains': api_results_f1.test_mains,
        'test_submeters': api_results_f1.test_submeters,

        'gt': api_results_f1.gt_overall,
        'predictions': api_results_f1.pred_overall,
    }
    pickle.dump(df_dict, open("out/i2s_{}-{}-df_dict.p".format(experiment['data_set'], experiment['house']), "wb"))

    ################### RESULTS ###################
    print('Experiment took: {} minutes'.format(round((time()-start)/60,1)))

    for metric, f1_errors in zip(error_keys_df_f1, error_df_f1):
        ff_errors = round(f1_errors, 3)
        ff_errors.to_csv('out/i2s_{}.csv'.format(metric), sep='\t')

exit()
