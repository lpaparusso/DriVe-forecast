# IMPORT PACKAGES
import numpy as np
import pandas as pd
import scipy
import scipy.io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pickle
from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import json

# FUNCTIONS DEFINITION
def load_dataset(dataset_name):
    
    """Load the driver/vehicle dataset"""
    
    df = pickle.load( open("dataPostCorrelation/"+ dataset_name, "rb" ) )
    df["Session-Lap"] = df["Session"] + " " + df["Lap"]
    
    return df


def load_track(track_name):

    """Load the track dataset"""
    
    mat = scipy.io.loadmat(track_name + "Map/" + track_name + "Data")

    mat.pop('__header__')
    mat.pop('__version__')
    mat.pop('__globals__')
    mat.pop('leftMarginX')
    mat.pop('leftMarginY')
    mat.pop('leftMarginZ')
    mat.pop('rightMarginX')
    mat.pop('rightMarginY')
    mat.pop('rightMarginZ')
    mat.pop('normalX')
    mat.pop('normalY')
    mat.pop('normalZ')
    mat.pop('tangentX')
    mat.pop('tangentY')
    mat.pop('tangentZ')
    mat.pop('curvature')

    for i in mat.keys():
        mat[i] = np.squeeze(mat[i]).tolist()

    trackMap = pd.DataFrame(mat) 
    
    return trackMap


def generate_fixed_sets(input_set, train_dev_test_laps):
    
    """Extract the training, validation and test sets by specifying the number of laps contained in each"""
    
    lap_names = input_set['Session-Lap'].unique()
    
    random_indices = np.random.permutation(lap_names.shape[0])
    shuffled_names = lap_names[random_indices]
    
    train_names = shuffled_names[:train_dev_test_laps[0]]
    dev_names = shuffled_names[train_dev_test_laps[0]:train_dev_test_laps[0]+train_dev_test_laps[1]] 
    test_names = shuffled_names[train_dev_test_laps[0]+train_dev_test_laps[1]:]
       
    df_train = pd.concat([input_set.loc[input_set["Session-Lap"]==i,:] for i in train_names])
    df_dev = pd.concat([input_set.loc[input_set["Session-Lap"]==i,:] for i in dev_names])
    df_test = pd.concat([input_set.loc[input_set["Session-Lap"]==i,:] for i in test_names])
    
    return (df_train, df_dev, df_test)


def insert_future(input_data, trackMap, road_features, dR, pR):
    
    """Insert the future road geometry information into the dataset"""
        
    # Create the names of the new columns of the extended dataset
    new_columns_names = []
    for i in range(pR):
        for j in road_features:
            new_columns_names.append('future_' + j + '_' + str(i)) 
    
    # Extend the track data, including the last point (coinciding with the first point)
    first_last_distance = np.linalg.norm(trackMap.iloc[-1,:][["centerlineX","centerlineY","centerlineZ"]].values - trackMap.iloc[0,:][["centerlineX","centerlineY","centerlineZ"]].values)
    track_extended = trackMap.copy()
    track_extended = track_extended.append(track_extended.iloc[0,:]).reset_index(drop=True)
    last_value = trackMap.iloc[-1,:]['curvilinearAb'] + first_last_distance
    track_extended.iloc[-1, :]['curvilinearAb'] = last_value
    
    # Define the interpolation function
    interp_function = scipy.interpolate.interp1d(track_extended['curvilinearAb'].to_numpy(),track_extended[road_features].to_numpy(), axis=0)
    
    # Declare useful vectors     
    index_closest=list(range(input_data.shape[0])) #initialise
    value_closest=list(range(input_data.shape[0]))

    value_farthest=list(range(input_data.shape[0]))
    values_range=list(range(input_data.shape[0]))

    new_columns=list(range(input_data.shape[0]))
    
    track_ab = trackMap['curvilinearAb'].values
    data_ab = input_data['curvilinearAb'].values

    for i in range(input_data.shape[0]):

        # Look for the current curvilinear abscissa (index,value), for all the time instants
        index_closest[i] = np.argmin(np.absolute(track_ab-data_ab[i]))  
        value_closest[i] = track_ab[index_closest[i]]

        # Calculate last future curvilinear abscissa, for all the time instants
        value_farthest[i] = value_closest[i] + dR

        # Calculate the whole future curvilinear abscissa, for all the time instants
        values_range[i] = np.linspace(value_closest[i], value_farthest[i], num=pR+1)[1:]
        values_range[i] = values_range[i] - last_value * (values_range[i]>last_value)

        # Create interpolated values of the quantities of interest                    
        new_columns[i] = interp_function(values_range[i]).reshape(-1)

    # Generate array and put it into the dataframe
    data_extended = input_data.copy()
    new_columns = np.array(new_columns).tolist()

    data_extended[new_columns_names] = new_columns

    return data_extended


def extract_features(input_data, driver_vehicle_road_features):
    
    """Extract all of the desired driver/vehicle/road features from the dataset"""
    
    data_extract = input_data.loc[:,driver_vehicle_road_features].copy()
    
    return data_extract


def window_stack(a, stride=1, dT=3):
    
    """Function called in apply_windowing to create stacked windows"""
    
    n = a.shape[0]
    temp_vec = np.array([a[i:i+dT, :] for i in range(0,n-dT+1,stride)])
    return temp_vec


def apply_windowing(input_set, stride, dT):
    
    """Apply windowing of given stride and window size to the dataset"""
    
    # Join the Session and Lap columns
    temp_set = input_set.copy()
    
    # Create numpy matrix
    numpy_data = [temp_set.loc[temp_set["Session-Lap"]==i, :].drop("Session-Lap", axis=1).to_numpy() for i in temp_set["Session-Lap"].unique()]
    windowed_data = np.vstack([window_stack(i, stride=stride, dT=dT) for i in numpy_data])
    
    return windowed_data


def data_preparation_routine_fixed(df, trackMap, road_features, dR, pR, driver_vehicle_road_features, stride, dT, train_dev_test_laps):

    """Routine to prepare data according to the given parameters"""

    print('Preparing the dataset...')
    # Separate train, dev and test laps
    df_train, df_dev, df_test = generate_fixed_sets(df, train_dev_test_laps)

    # Insert data of the future
    df_future = insert_future(df, trackMap, road_features, dR, pR)
    df_train_future = insert_future(df_train, trackMap, road_features, dR, pR)
    df_dev_future = insert_future(df_dev, trackMap, road_features, dR, pR)
    df_test_future = insert_future(df_test, trackMap, road_features, dR, pR)

    # Extract features of interest
    data_extract = extract_features(df_future,driver_vehicle_road_features)
    data_extract_train = extract_features(df_train_future,driver_vehicle_road_features)
    data_extract_dev = extract_features(df_dev_future,driver_vehicle_road_features)
    data_extract_test = extract_features(df_test_future,driver_vehicle_road_features)

    # Normalise data
    meanv = data_extract.drop(["Session-Lap"],axis=1).mean()
    stdv = data_extract.drop(["Session-Lap"],axis=1).std()

    data_nor_train = (data_extract_train.drop(["Session-Lap"],axis=1)-meanv)/stdv
    data_nor_dev = (data_extract_dev.drop(["Session-Lap"],axis=1)-meanv)/stdv
    data_nor_test = (data_extract_test.drop(["Session-Lap"],axis=1)-meanv)/stdv

    data_nor_train["Session-Lap"] = data_extract_train["Session-Lap"]
    data_nor_dev["Session-Lap"] = data_extract_dev["Session-Lap"]
    data_nor_test["Session-Lap"] = data_extract_test["Session-Lap"]

    # Apply windowing
    x_train = apply_windowing(data_nor_train, stride=stride, dT=dT)
    x_dev = apply_windowing(data_nor_dev, stride=stride, dT=dT)
    x_test = apply_windowing(data_nor_test, stride=stride, dT=dT)
    
    print('Dataset prepared')
    
    # Return the datasets
    return (x_train, x_dev, x_test, meanv, stdv)


def build_model(dT, tF, driver_vehicle_indices, road_indices, pR, u_e, r, u_d, prediction_indices, primary_features, w):
    
    """Build and compile the Neural Network"""
    
    # Define some parameters
    n_past = len(driver_vehicle_indices)
    n_road = len(road_indices)
    n_pred = len(prediction_indices)
    q = len(primary_features)
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=(dT-tF, n_past), name='Input_past')
    
    inputs_road = tf.keras.Input(shape=(n_road), name='Input_road')
    
    # LSTM encoder for the past
    lstm_encoder, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(u_e, 
                                                                                                                    return_state=True, 
                                                                                                                    name='encoder_past'))(inputs)
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    
    # LSTM encoder for the future road
    reshape_road = tf.keras.layers.Reshape((pR, -1))(inputs_road)
    lstm_encoder_2, forward_h_2, forward_c_2, backward_h_2, backward_c_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(u_e, 
                                                                                                                              return_state=True, 
                                                                                                                              name='encoder_road'))(reshape_road)
    state_h_2 = tf.keras.layers.Concatenate()([forward_h_2, backward_h_2])
    state_c_2 = tf.keras.layers.Concatenate()([forward_c_2, backward_c_2])
    
    # Concatenate the two hidden states and use a dense encoding
    concatenate = tf.keras.layers.Concatenate(axis=-1, name='Concatenate')
    concatenate_h = tf.keras.layers.Concatenate(axis=-1)([state_h, state_h_2])
    concatenate_c = tf.keras.layers.Concatenate(axis=-1)([state_c, state_c_2])
    
    dense_encoding = tf.keras.layers.Dense(u_e*2, name='Dense_encoding')
    dense_encoding2 = tf.keras.layers.Dense(u_e*2, name='Dense_encoding2')
    dense_encoding3 = tf.keras.layers.Dense(u_d, name='Dense_encoding3')
    dense_encoding4 = tf.keras.layers.Dense(u_d, name='Dense_encoding4')
    state_h_dense = dense_encoding(concatenate_h)
    state_h_dense =tf.keras.layers.Dropout(r)(state_h_dense)
    state_h_dense = dense_encoding3(state_h_dense)
    state_c_dense = dense_encoding2(concatenate_c)
    state_c_dense =tf.keras.layers.Dropout(r)(state_c_dense)
    state_c_dense = dense_encoding4(state_c_dense)
    encoder_state = [state_h_dense, state_c_dense]
    
    # LSTM recursive decoder
    lstm_list = []
    cell = tf.keras.layers.LSTMCell(u_d)
    lstm_cell, hidden_state = cell(inputs=encoder_state[0], states=encoder_state)
    lstm_list.append(lstm_cell)
    for i in range(tF-1):
        lstm_cell, hidden_state = cell(lstm_cell, states=hidden_state, training=True)
        lstm_list.append(lstm_cell)
    lstm_decoder = tf.stack(lstm_list, axis=1)
    
    lstm_decoder_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(u_d, 
                                                                        return_sequences=True, 
                                                                        name='decoder_2'))(lstm_decoder)
    
    # Time distributed dense layer
    dense_layer = tf.keras.layers.Dense(n_pred, name='dense_layer')
    outputs = tf.keras.layers.TimeDistributed(dense_layer)(lstm_decoder_2)
    
    # Declare model
    def custom_loss(y_actual,y_pred):
        weight = tf.constant([1]*q + [w]*(n_pred-q) ,dtype=tf.float32)
        custom_loss = tf.keras.backend.mean(tf.keras.backend.square((y_actual-y_pred)*weight), axis=-1)
        return custom_loss

    def custom_metrics(y_actual, y_pred):
        metrics = tf.keras.backend.mean(tf.keras.backend.abs((y_actual[:,:,:q]-y_pred[:,:,:q])), axis=-1)
        return metrics
    model = tf.keras.Model(inputs=[inputs, inputs_road], outputs=outputs)
            
    # Define custom loss and metrics     
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # the learning rate is dummy, because an adaptive learning rate will be specified later
    model.compile(loss=custom_loss, metrics=[custom_metrics], optimizer = optimizer)
    
    return model


def train_model(r, w, tP, u_ed, xi, save_results=False, i='', config_parameters=''):
    
    """Routine to create the dataset, create the Neural Network and save results"""
    
    # Define prefix name of the files
    fileName = 'Data30red'
    
    # Get the parameters
    r = r
    w = w
    tP = int(tP)
    u_e = int(u_ed*xi)
    u_d = int(u_ed-u_e)
    
    # And get the config parameters
    tF = config_parameters['tF']
    df = config_parameters['df']
    trackMap = config_parameters['trackMap']
    road_features = config_parameters['road_features']
    dR = config_parameters['dR']
    pR = config_parameters['pR']
    driver_vehicle_road_features = config_parameters['driver_vehicle_road_features']
    stride = config_parameters['stride']
    train_dev_test_laps = config_parameters['train_dev_test_laps']
    driver_vehicle_indices = config_parameters['driver_vehicle_indices']
    road_indices = config_parameters['road_indices']
    prediction_indices = config_parameters['prediction_indices']
    batch_size = config_parameters['batch_size']
    primary_features = config_parameters['primary_features']
    
    dT = int(tP+tF)
    
    # Prepare data to be fed to the Neural Network
    if i=='':
        np.random.seed(1) # to use the same training-validation-test split (Bayesian optimisation)
        tf.random.set_seed(1)
    else:
        np.random.seed(i * 18) # to change training-validation-test split at every iteration (Cross-validation)
        
    x_train, x_dev, x_test, meanv, stdv = data_preparation_routine_fixed(df,
                                                                         trackMap, 
                                                                         road_features,
                                                                         dR, 
                                                                         pR, 
                                                                         driver_vehicle_road_features, 
                                                                         stride, 
                                                                         dT, 
                                                                         train_dev_test_laps)
    # Create Tensorflow datasets and shuffle them
    def _generator_train():
    
        """Returns the training generator"""

        for i in x_train:
            x_values = i[:-tF, driver_vehicle_indices]
            future_values = i[-tF-1, road_indices]
            y_values = i[-tF:, prediction_indices]
            yield (x_values, future_values), y_values


    def _generator_dev():

        """Returns the validation generator"""

        for i in x_dev:
            x_values = i[:-tF, driver_vehicle_indices]
            future_values = i[-tF-1, road_indices]
            y_values = i[-tF:, prediction_indices]
            yield (x_values, future_values), y_values


    def _generator_test():

        """Returns the test generator"""

        for i in x_test:
            x_values = i[:-tF, driver_vehicle_indices]
            future_values = i[-tF-1, road_indices]
            y_values = i[-tF:, prediction_indices]
            yield (x_values, future_values), y_values
            
    train_dataset = tf.data.Dataset.from_generator(generator=_generator_train, 
                                                  output_types=((tf.float32, tf.float32), tf.float32))

    dev_dataset = tf.data.Dataset.from_generator(generator=_generator_dev,
                                                 output_types=((tf.float32, tf.float32), tf.float32))

    test_dataset = tf.data.Dataset.from_generator(generator=_generator_test,
                                                 output_types=((tf.float32, tf.float32), tf.float32))


    train_dataset = train_dataset.shuffle(x_train.shape[0], reshuffle_each_iteration=False).batch(batch_size)
    dev_dataset = dev_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)
    
    # Build and compile the Neural Network model
    model = build_model(dT, 
                        tF, 
                        driver_vehicle_indices, 
                        road_indices, 
                        pR, 
                        u_e, 
                        r, 
                        u_d,
                        prediction_indices, 
                        primary_features,
                        w)
    
    # Define the name of the file in which the best Neural Network weights will be saved
    hyperparams_name = '{}_{}.r_{}.w_{}.tP_{}.u_ed_{}.xi_{}'.format(
        fileName , i, round(r,5), round(w,5), tP, round(u_ed,5), round(xi,5))
    fname_param = os.path.join('results/MODEL', '{}.best.h5'.format(hyperparams_name))

    # Define early stopping option, model checkpoint and learning rate scheduler
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_custom_metrics', mode='min', patience=25)
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_custom_metrics', verbose=0, save_best_only=True, mode='min')
    
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: tf.math.maximum(1e-3 * 0.98**epoch,5e-4))
    
    # Train the model
    print("Training model...")
    ts = time.time()
    
    if (i): # Cross-validation
        print(f'Iteration {i}')
        np.random.seed(i * 18)
        tf.random.set_seed(i * 18)
        
    # First training phase
    history = model.fit(train_dataset, validation_data = dev_dataset,
                        epochs = 100,
                        callbacks = [early_stopping, model_checkpoint, learning_rate_scheduler],
                        verbose = 2,
                        workers = 20, 
                        use_multiprocessing = True)
    
    # Second training phase
    n_pred = len(prediction_indices)
    q = len(primary_features)
    
    def custom_loss(y_actual,y_pred):
        weight = tf.constant([1]*q + [w]*(n_pred-q) ,dtype=tf.float32)
        custom_loss = tf.keras.backend.mean(tf.keras.backend.square((y_actual-y_pred)*weight), axis=-1)
        return custom_loss

    def custom_metrics(y_actual, y_pred):
        metrics = tf.keras.backend.mean(tf.keras.backend.abs((y_actual[:,:,:q]-y_pred[:,:,:q])), axis=-1)
        return metrics
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=custom_loss, metrics=[custom_metrics], optimizer = optimizer)
    history = model.fit(train_dataset, validation_data = dev_dataset,
                        epochs = 500,
                        callbacks = [early_stopping, model_checkpoint],
                        verbose = 2,
                        workers = 20, 
                        use_multiprocessing = True)
    
    
    # Save the weights and history of the model
    model.save_weights(os.path.join(
        'results/MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        'results/MODEL', '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # Evaluate the metrics on the training and validation sets
    model.load_weights(fname_param)
    score = model.evaluate(dev_dataset, verbose=0)
    print('Dev loss: %.6f;  Dev metrics: %.6f' %
          (score[0], score[1]))

    # If we want to save results in a csv file
    if(save_results):
        print('Evaluating using the model that has the best metrics on the validation set')
        model.load_weights(fname_param)  # load best weights for current iteration

        # define the scores
        score_train = model.evaluate(train_dataset)
        score_dev = model.evaluate(dev_dataset)
        score_test = model.evaluate(test_dataset)
        score = score_dev + score_train + score_test

        # save to csv
        csv_name = os.path.join('results/best', '{}_results.csv'.format(fileName))
        if not os.path.isfile(csv_name):
            if os.path.isdir('results/best') is False:
                os.mkdir('results/best')
            with open(csv_name, 'a', encoding="utf-8") as file:
                file.write('Iteration,'
                           'loss_train,metrics_train,'
                           'loss_dev,metrics_dev,'
                           'loss_test,metrics_test'
                           )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write(f'{i},{score[2]},{score[3]},'
                       f'{score[0]},{score[1]},'
                       f'{score[4]},{score[5]}'
                       )
            file.write("\n")
            file.close()
        K.clear_session()

    # Bayesian optimisation is a maximisation algorithm; to minimise validation_metrics, return -value
    bayes_opt_score = - score[1] # metrics on validation set

    return bayes_opt_score


def best_model(fileName, config_parameters=''):
    
    """Routine to load the best trained model and a test set"""
    
    # Load file containing the best hyperparameters
    params_fname = '{}_results_best_params.json'.format(fileName)
    with open('results/best/'+ params_fname, 'r') as f:
        params = json.load(f)
        

    # Get the parameters
    r = params['r']
    w = params['w']
    tP = int(params['tP'])
    u_ed = params['u_ed']
    xi = params['xi']
    u_e = int(u_ed*xi)
    u_d = int(u_ed-u_e)    
    
    # And get the config parameters
    tF = config_parameters['tF']
    df = config_parameters['df']
    trackMap = config_parameters['trackMap']
    road_features = config_parameters['road_features']
    dR = config_parameters['dR']
    pR = config_parameters['pR']
    driver_vehicle_road_features = config_parameters['driver_vehicle_road_features']
    stride = config_parameters['stride']
    train_dev_test_laps = config_parameters['train_dev_test_laps']
    driver_vehicle_indices = config_parameters['driver_vehicle_indices']
    road_indices = config_parameters['road_indices']
    prediction_indices = config_parameters['prediction_indices']
    batch_size = config_parameters['batch_size']
    primary_features = config_parameters['primary_features']
    
    dT = int(tP+tF)    
    
    # Prepare data to be fed to the Neural Network
    np.random.seed()        
    x_train, x_dev, x_test, meanv, stdv = data_preparation_routine_fixed(df,
                                                                         trackMap, 
                                                                         road_features,
                                                                         dR, 
                                                                         pR, 
                                                                         driver_vehicle_road_features, 
                                                                         stride, 
                                                                         dT, 
                                                                         train_dev_test_laps)
    
    # Create Tensorflow datasets and shuffle them
    def _generator_train():
    
        """Returns the training generator"""

        for i in x_train:
            x_values = i[:-tF, driver_vehicle_indices]
            future_values = i[-tF-1, road_indices]
            y_values = i[-tF:, prediction_indices]
            yield (x_values, future_values), y_values


    def _generator_dev():

        """Returns the validation generator"""

        for i in x_dev:
            x_values = i[:-tF, driver_vehicle_indices]
            future_values = i[-tF-1, road_indices]
            y_values = i[-tF:, prediction_indices]
            yield (x_values, future_values), y_values


    def _generator_test():

        """Returns the test generator"""

        for i in x_test:
            x_values = i[:-tF, driver_vehicle_indices]
            future_values = i[-tF-1, road_indices]
            y_values = i[-tF:, prediction_indices]
            yield (x_values, future_values), y_values
            
    train_dataset = tf.data.Dataset.from_generator(generator=_generator_train, 
                                                  output_types=((tf.float32, tf.float32), tf.float32))

    dev_dataset = tf.data.Dataset.from_generator(generator=_generator_dev,
                                                 output_types=((tf.float32, tf.float32), tf.float32))

    test_dataset = tf.data.Dataset.from_generator(generator=_generator_test,
                                                 output_types=((tf.float32, tf.float32), tf.float32))


    train_dataset = train_dataset.shuffle(x_train.shape[0], reshuffle_each_iteration=False).batch(batch_size)
    dev_dataset = dev_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)
    
    

    # Build and compile the Neural Network model
    model = build_model(dT, 
                        tF, 
                        driver_vehicle_indices, 
                        road_indices, 
                        pR, 
                        u_e, 
                        r, 
                        u_d,
                        prediction_indices, 
                        primary_features,
                        w)  

    # Load the weights of the best model
    hyperparams_name = '{}_{}.r_{}.w_{}.tP_{}.u_ed_{}.xi_{}'.format(
        fileName , '', round(r,5), round(w,5), tP, round(u_ed,5), round(xi,5))
    
    fname_param = 'results/MODEL/{}.best.h5'.format(hyperparams_name)
    model.load_weights(fname_param)
    
    return model, test_dataset, x_test, meanv, stdv
