# Install and import the required packages.
# !pip install -r requirements.txt

import os
if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists('models/MODEL'):
    os.chdir('models')
    os.mkdir('MODEL')
    os.chdir('..')
    
if not os.path.exists('models/best'):
    os.chdir('models')
    os.mkdir('best')
    os.chdir('..')
    
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from bayes_opt import BayesianOptimization 
import json

import functions
import plotting_functions


# # A Deep-Learning Framework to Predict the Dynamics of a Human-Driven Vehicle based on the Road Geometry

# ## PARAMETERS DEFINITION
# In this section, the main parameters are defined (for information, read the reference paper):
# - Track dataframe
# - Driver/Vehicle dataframe
# - List of features (primary, secondary and road features)
# - The parameters stride, dR, pR, tF
# - The batch size used during the Neural Network training
# - The number of laps covered by the training, validation and test sets

# Use the notebook to perform also training
train = False

# Name of the config file and model
configName = "Data30red"
fileName = "Data30red"

import json
with open("configs/" + configName + ".json", "r") as read_file:
    config_parameters = json.load(read_file)

# Name of the track dataset
track_name = config_parameters["track_name"]

# Name of the driver/vehicle dataset
dataset_name = config_parameters["dataset_name"]

# Load the track dataset
trackMap = functions.load_track(track_name)

# Load the driver/vehicle dataset
df = functions.load_dataset(dataset_name)

# All of the driver/vehicle/road features needed in the prediction framework
driver_vehicle_road_features = config_parameters["driver_vehicle_road_features"]

driver_vehicle_road_features.append('Session-Lap') # Session-Lap is needed to manage the dataset

# All of the driver/vehicle features
driver_vehicle_features = config_parameters["driver_vehicle_features"]

# The driver/vehicle features to predict (primary and secondary)
prediction_features = config_parameters["prediction_features"]

# The primary features (the features of interest in the prediction problem)
primary_features = config_parameters["primary_features"]

# The road geometry features
road_features = config_parameters["road_features"]

# The stride in the windowing process
stride = config_parameters["stride"]

# The parameter d^R
dR = config_parameters["dR"] # in meters

# The parameter p^R
pR = config_parameters["pR"]

# The parameter t^F (Prediction steps)
tF = config_parameters["tF"]

# The batch size used during training of the Neural Network
batch_size = config_parameters["batch_size"]

# The number of laps constituting the training, validation and test sets
total_laps = df['Session-Lap'].nunique() # total laps in the dataset
n_train = int(total_laps*config_parameters["train_percentage"]) # number of training laps
n_test = config_parameters["test_laps"] # number of test laps
n_dev = total_laps-n_train-n_test # number of validation laps

train_dev_test_laps = [n_train, n_dev, n_test]

# Use road information
road_on = config_parameters["road_on"]

# ## EXECUTE SOME UTILITIES
# In this section, some minor utilities are run.

# Reorder lists, so that the primary features are always first, and the order of the features is always the same in the different lists
driver_vehicle_road_features = primary_features + [el for el in driver_vehicle_road_features if (primary_features.count(el)==0)]
prediction_features = [el for el in driver_vehicle_road_features if (prediction_features.count(el)>0)]
road_features = [el for el in driver_vehicle_road_features if (road_features.count(el)>0)]

# Define the indices of the features
driver_vehicle_indices = []
for i in driver_vehicle_features:
    driver_vehicle_indices.append(driver_vehicle_road_features.index(i))

prediction_indices = []
for i in prediction_features:
    prediction_indices.append(driver_vehicle_road_features.index(i))
 
road_features_extended = []
for i in range(pR):
    for j in road_features:
        road_features_extended.append('future_' + j + '_' + str(i))
        
driver_vehicle_road_features.remove("Session-Lap") 
driver_vehicle_road_features = driver_vehicle_road_features + road_features_extended + ["Session-Lap"] # I put "Session-Lap" field at the very end

road_indices = []
for i in road_features_extended:
    road_indices.append(driver_vehicle_road_features.index(i))

# Store config parameters in a dictionary
config_parameters['df'] = df
config_parameters['trackMap'] = trackMap
config_parameters['road_features'] = road_features
config_parameters['driver_vehicle_road_features'] = driver_vehicle_road_features
config_parameters['train_dev_test_laps'] = train_dev_test_laps
config_parameters['driver_vehicle_indices'] = driver_vehicle_indices
config_parameters['road_indices'] = road_indices
config_parameters['prediction_indices'] = prediction_indices
config_parameters['primary_features'] = primary_features

# ## BAYESIAN OPTIMISATION
# In this section, Bayesian optimisation is launched. The routine loads the hyperparameters defined at each iteration by the Bayesian optimisation algorithm (read the reference paper), and the previously defined parameters. Then, it builds the input matrices to be fed to the Neural Network, trains the Neural Network and saves the model weights. At the end of Bayesian optimisation, the best hyperparameter combination is saved in a dedicated file.

if train:

    # Configure   
    def train_model_launcher(r, w, tP, u_ed, xi, save_results=False, i=''):
        bayes_opt_score = functions.train_model(r, w, tP, u_ed, xi, save_results=save_results, i=i, config_parameters=config_parameters, fileName=fileName)
        return bayes_opt_score

    optimizer = BayesianOptimization(f = train_model_launcher,
                                     pbounds={'r': (0.25, 0.5),
                                              'w': (0.0, 1.0), 
                                              'tP': (15, 40),
                                              'u_ed': (70, 110), 
                                              'xi': (0.3, 0.5)
                                             },
                                      verbose=2)

    # Run
    optimizer.maximize(init_points=2, n_iter=10)

    # Save the optimiser
    pickle.dump(optimizer, open('models/best/optimizer_{}.pkl'.format(fileName), 'wb'))

    # Save the best hyperparameters in a dedicated file
    targets = [e['target'] for e in optimizer.res]
    best_index = targets.index(max(targets))
    params = optimizer.res[best_index]['params']

    params_fname = '{}_results_best_params.json'.format(fileName)
    with open(os.path.join('models/best', params_fname), 'w') as f:
         json.dump(params, f, indent=2)


# ## CROSS-VALIDATION
# In this section, Cross-validation is used to assess the performance of the optimal model found with Bayesian optimisation. A CSV file reporting the scores on the different sets is generated.

if train:

    params_fname = '{}_results_best_params.json'.format(fileName)
    with open(os.path.join('models/best', params_fname), 'r') as f:
        params = json.load(f)

    # Run cross-validation
    cv_steps = 10 # steps of Cross-validation
    for i in range(0, cv_steps):
        train_model_launcher(r=params['r'],
                        w=params['w'],
                        tP=params['tP'],
                        u_ed=params['u_ed'],
                        xi=params['xi'],
                        save_results=True,
                        i=i)

        
extra_variables_names = dict()
extra_variables_names['vehicle'] = ['chassis_displacements.longitudinal',
                                    'chassis_displacements.lateral',
                                    'chassis_velocities.longitudinal',
                                    'chassis_velocities.lateral']
extra_variables_names['road'] = ['centerlineX',
                              'centerlineY',
                              'normalX',
                              'normalY',
                              'tangentX',
                              'tangentY',
                              'leftMarginX',
                              'leftMarginY',
                              'rightMarginX',
                              'rightMarginY']


# ## LOAD BEST MODEL WITH A RANDOM TEST SET AND SAVE THE RESULTS INTO A FILE
result_file_name = 'saved_example_newdriver'

model, test_dataset, x_test, meanv, stdv, extra_variables = functions.best_model(fileName, config_parameters=config_parameters, extra_variables_names=extra_variables_names)
meanv = meanv.to_numpy()
stdv = stdv.to_numpy()

# Predict the behaviour of the model on the test set
if road_on:
    X_true = np.array([np.squeeze(i[0][0], axis=0) for i in test_dataset])[:, :, prediction_indices] # I calculate the past only for the values I predict, for the plots
    Y_true = np.array([np.squeeze(i[1], axis=0) for i in test_dataset])
    Y_pred = model.predict(test_dataset)
else:
    X_true = np.array([np.squeeze(i[0], axis=0) for i in test_dataset])[:, :, prediction_indices]  # I calculate the past only for the values I predict, for the plots
    Y_true = np.array([np.squeeze(i[1], axis=0) for i in test_dataset])
    Y_pred = model.predict(test_dataset)

# Save in a pickle file to avoid loading everytime from scratches
dictionary = {'X_true': X_true, 'Y_true': Y_true, 'Y_pred': Y_pred, 'meanv': meanv, 'stdv': stdv, 'extra_variables': extra_variables}

# with open('pre_computed_tests/' + result_file_name + '.pickle', 'wb') as handle:
#     pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)