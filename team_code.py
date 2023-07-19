#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np, os, sys, time, json
import pandas as pd
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from helper_code import *
from vae import VanillaVAE
import torch
import wfdb
import wandb # just a shim
from sklearn.preprocessing import RobustScaler

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    if verbose >= 1:
        print('Training VAE')
    pt_records = prepare_mmap(data_folder, patient_ids, verbose)
    with torch.autograd.detect_anomaly():
        train_vae(pt_records, model_folder, verbose)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

    features = np.vstack(features)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    models_dict = joblib.load(filename)
    fname2 = os.path.join(model_folder, 'vae.pth')
    vae = VanillaVAE(18, 400, 1024, [512])
    with open(fname2, 'rb') as f:
        vae.load_state_dict(torch.load(f))
    models_dict['vae'] = vae
    

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    vae = models['vae']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    print(outcome_model.predict_proba(features))
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def get_latents(vae: VanillaVAE, data_folder, patient_id):
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)
    if num_recordings > 0:
        for recording_id in recording_ids:
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))

def split_recordings(pt_records, WINDOW_LEN):
    '''splits every row of pt_records, which represents a single 30-second
    recording, into multiple segments of length WINDOW_LEN'''
    train_idx = []
    for _, row in pt_records.dropna().iterrows():
        for i in np.arange(0, 30_000-WINDOW_LEN, WINDOW_LEN):
            train_idx.append((row['Folder'], row['Record'], i))
    rng = np.random.default_rng(seed=42)
    rng.shuffle(train_idx)
    return train_idx


def prepare_mmap(data_folder, patient_ids, verbose=1, force_reload=False):
    if verbose >= 1:
        print('Loading patient records...')
    records = []
    folders = []
    for patient_id in patient_ids:
        rec = find_recording_files(data_folder, patient_id)
        records += rec
        folders += [str(os.path.join(data_folder, patient_id)).replace('\\', '/')] * len(rec)
    pt_records = pd.DataFrame({'Record': records, 'Folder': folders})
    pt_records.reset_index(drop=True, inplace=True)
    # the CSV's have blank rows for hours with no recording
    pt_records = pt_records.dropna()
    if verbose >= 1:
        print('Done loading patient records')

    if not force_reload and os.path.exists('sigdata.npy'):
        if verbose >= 1:
            print('Using old memmap')
        return pt_records

    if verbose >= 1:
        print('Building memmap...')
    # get unique records in the train dataset
    records_idx = pt_records[['Folder', 'Record']].reset_index(drop=True)
    # makes 'index' a column that we can lookup based on the Record
    record_lookup = records_idx.reset_index().set_index('Record')
    
    # load records into a memmap
    # fixed cost of 160 s to save load times by 30-40% per batch later
    mmap = np.memmap('sigdata.npy', mode='w+', shape=(len(records_idx), 30000, len(wandb.config.channels)), dtype='float32')
    scaler = RobustScaler(quantile_range=(10,90))
    for i, row in records_idx.iterrows():
        rec = wfdb.rdrecord(os.path.join(row['Folder'], row['Record']), 
                            channel_names=wandb.config.channels)
        # only read first 5 min
        sig = rec.p_signal[:30_000,:]
        mmap[i] = scaler.fit_transform(sig)
    # assumes that all recordings have same sig_names in same order
    sig_names = rec.sig_name
    with open('sig_names.json', 'w') as f:
        f.write(json.dumps(sig_names))
    if verbose >= 1:
        print('Done building memmap')

    return pt_records


def build_batch(train_idx, mmap, record_lookup, first_idx, BATCH_CNT, WINDOW_LEN):
    '''returns a tensor-ified batch using the information in the rows
    of train_idx to get the EEG data from rows first_idx to first_idx+BATCH_CNT'''
    df_batch = np.array([])
    for b in range(BATCH_CNT):
        scaled_sig = mmap[record_lookup.loc[train_idx[first_idx+b][1]]['index']]
        # which index of the recording to start the window at
        start_idx = train_idx[first_idx+b][2]
        df_batch = np.append(df_batch,
                             scaled_sig[start_idx:start_idx+WINDOW_LEN])
    df_batch = df_batch.reshape(BATCH_CNT, WINDOW_LEN, len(wandb.config.channels))
    return torch.tensor(np.transpose(df_batch,axes=[0,2,1])).float()

def train_vae(pt_records, model_folder, verbose=1):
    # to prevent running for too long
    start = time.time()

    wandb.config.kld_weight_beta = 0.05
    wandb.config.latent_dim = 400
    wandb.config.epochs = 1
    wandb.config.learning_rate = 1e-5
    wandb.config.momentum = 0.8
    wandb.config.batch_size = 64
    wandb.config.window_len = 1024
    wandb.config.alpha = 0.001 # increases weight of amp_shift_loss
    wandb.config.gamma = 0.001 # increases weight of amp_loss
    
    vae = VanillaVAE(len(wandb.config.channels), wandb.config.latent_dim, wandb.config.window_len, 
                     hidden_dims=[128,256,512])
    optim = torch.optim.SGD(vae.parameters(), lr=wandb.config.learning_rate, momentum=wandb.config.momentum) # eps=wandb.config.learning_rate*1e-5)
    
    # get unique records in the train dataset
    records_idx = pt_records[['Folder', 'Record']].reset_index(drop=True)
    # makes 'index' a column that we can lookup based on the Record
    record_lookup = records_idx.reset_index().set_index('Record')

    # load records into a memmap
    # fixed cost of 160 s to save load times by 30-40% per batch later
    mmap = np.memmap('sigdata.npy', mode='r', shape=(len(records_idx), 30000, len(wandb.config.channels)), dtype='float32')
    
    # number of frames in one window (sampled at 100 Hz)
    WINDOW_LEN = wandb.config.window_len
    # number of windows in one batch
    BATCH_CNT = wandb.config.batch_size
    for epoch in range(wandb.config.epochs):
        train_idx = split_recordings(pt_records, WINDOW_LEN)

        batch_num = 0
        for i in range(0, len(train_idx)-BATCH_CNT, BATCH_CNT):
            optim.zero_grad()
            
            # train for 10 hrs max to avoid going over the limit
            if time.time() > start + 3600 * 10:
                break
            
            inp = build_batch(train_idx, mmap, record_lookup, i, BATCH_CNT, WINDOW_LEN)
            rand_prob = torch.rand(BATCH_CNT)
            out = vae(inp, rand_prob)
            loss_vars = vae.loss_function(*out, debug=False, batch_size=BATCH_CNT, rand_prob=rand_prob)
            loss = loss_vars['loss']
            if verbose >= 2:
                print(f"batch {batch_num}: loss = {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 250)
            optim.step()
            batch_num += 1
                
    torch.save(vae.state_dict(), os.path.join(model_folder, 'vae.pth'))
    

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model, vae):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model, 'vae': vae}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F3', 'P3', 'F4', 'P4']
    group = 'EEG'

    # Extract ECG features.
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    # Extract features.
    # return np.hstack((patient_features, eeg_features, ecg_features))
    # TODO restore eeg and ecg features
    return patient_features

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features
