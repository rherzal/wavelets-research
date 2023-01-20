import h5py
import numpy as np
from scipy.interpolate import interp1d
import pywt
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.linear_model import LinearRegression

def load_measurements(filename):
    with h5py.File(filename) as f:
        CLOCKS_PER_SECOND = 1000000
        # load the data form the h5 file
        joint_data = np.array(f['Body markers'])
        emg_data = np.array(f['emg signals'])
        # move the time-stamps to origin and convert to seconds
        t0 = joint_data[0,0]
        joint_data[:,0] = [(t - t0) / CLOCKS_PER_SECOND for t in joint_data[:,0]]

        # print("----TIME----")
        # print(joint_data[:,0])
        # print("----END TIME----")

        t0 = emg_data[0,0]
        emg_data[:,0] = [(t - t0) / CLOCKS_PER_SECOND for t in emg_data[:,0]]
        # returns the data as a tuple
        return joint_data, emg_data

def interpolate_measurements(data, sampling_rate = 20000):
    joint_data, emg_data = data
    # generate equally spaced out timestamps needed for wavelet conversion
    equally_spaced_out_timestamps = np.arange(0, joint_data[-1,0], 1/sampling_rate)
    # use interpolation witch-craft to get the samples
    f_emg = interp1d(emg_data[:,0], emg_data[:,1:], axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
    f_joint = interp1d(joint_data[:,0], joint_data[:,1:], axis=0, bounds_error=False, fill_value='extrapolate', kind='linear')
    interpolated_emg_data = f_emg(equally_spaced_out_timestamps)
    interpolated_joint_data = f_joint(equally_spaced_out_timestamps)
    # returns the data as a tuple, again

    return interpolated_joint_data, interpolated_emg_data, sampling_rate, equally_spaced_out_timestamps


def get_window_emg(emg_data, channel, window_length, initial_time):
    # slices the emg data to a single window
    # windows_length and initial_time measured in samples

    sliced_data = emg_data[initial_time:(initial_time + window_length), channel]

    return sliced_data



def transform_wavelet(emg_data, wavelet, level):
    # Does the n-level DWT on the emg_data

    coefficients = pywt.wavedec(data=emg_data, wavelet=wavelet, level=level)
    
    return level, coefficients


def generate_regressors_slice(emg_data_slice, level, back_time):
    # generates the regressors for only one sliced channel of the emg data

    level, coefficients = transform_wavelet(emg_data=emg_data_slice, wavelet='db4', level=level)
    return_matrix = np.empty((((level + 1) * back_time), 1))
    for i in range(level + 1):
        # return_matrix[i, 0] = coefficients[i][-1]
        # return_matrix[i+1, 0] = coefficients[i][-2]
        for j in range(back_time):
            return_matrix[i+j, 0] = coefficients[i][-(j+1)]
    return return_matrix


def generate_regressors(emg_data, channel, window_length, initial_time, final_time, level, back_time):
    # generates the data for the SISO, emg in emg out regression 

    i = initial_time
    X = np.empty((((level + 1) * back_time), 0))
    Y = np.empty((0,0))
    while i <= final_time:
        sliced_emg = get_window_emg(emg_data, channel, window_length, i)
        return_matrix = generate_regressors_slice(emg_data_slice=sliced_emg, level=level, back_time=back_time)
        X = np.append(X, return_matrix, 1)
        Y = np.append(Y, emg_data[i + window_length, channel])

        i += 1
    return np.transpose(X), Y

def generate_multiple_regressors(emg_data, window_length, initial_time, final_time, level, back_time):
    # generates the data for the MIMO emg in emg out regression
    i = initial_time
    n_channels = emg_data.shape[1]
    X = np.empty(((n_channels * back_time * (level + 1)), 0))
    Y = np.empty((n_channels, 0))
    
    while i <= final_time:
        for channel in range(n_channels):
            sliced_emg = get_window_emg(emg_data=emg_data, channel=channel, window_length=window_length, initial_time=i)
            return_matrix = generate_regressors_slice(emg_data_slice=sliced_emg, level=level, back_time=back_time)
            if channel == 0:
                tmp_column_x = return_matrix
                tmp_column_y = np.array([emg_data[i + window_length, channel]])
            else:
                tmp_column_x = np.append(tmp_column_x, return_matrix, 0)
                tmp_column_y = np.append(tmp_column_y, np.array([emg_data[i + window_length, channel]]), 0)
        X = np.append(X, tmp_column_x, 1)
        Y = np.append(Y, tmp_column_y.reshape((n_channels, 1)), 1)
        i += 1
    
    return np.transpose(X), np.transpose(Y)


def forward_kinematics(Theta):
    # converts joint data at a single timestamp into hand position


    # measured in meters
    l1      = 0
    l2      = 0.29
    l3      = 0.32

    Alpha   = np.array([0, np.pi/2, np.pi/2, -np.pi/2, 0])
    D       = np.array([0, 0, l2, 0, 0])
    R       = np.array([l1, 0, 0, 0, l3])


    R0 = np.identity(4)

    for i in range(5):
        Ri = np.array([
            [np.cos(Theta[i]), -np.sin(Theta[i]), 0, R[i]], 
            [np.sin(Theta[i])*np.cos(Alpha[i]), np.cos(Theta[i])*np.cos(Alpha[i]), -np.sin(Alpha[i]), -D[i]*np.sin(Alpha[i])], 
            [np.sin(Theta[i])*np.sin(Alpha[i]), np.cos(Theta[i])*np.sin(Alpha[i]), np.cos(Alpha[i]), D[i]*np.cos(Alpha[i])],
            [0, 0, 0, 1]
        ], dtype=object)
        
        R0 = np.matmul(R0, Ri)
    
    return R0


def convert_joint_data(joint_data):
    # converts the array of joint datas into an array of position data


    # joint_data has each joint data in a different column
    # position should have 3 lines for each of the carthesian coordinates of the hand
    position = np.empty([3, joint_data.shape[0]])

    for i in range(joint_data.shape[0]):
        R = forward_kinematics(joint_data[i, 0:5])

        position[0, i] = R[0, 3]
        position[1, i] = R[1, 3]
        position[2, i] = R[2, 3]

    return position

def generate_multiple_position_regressors_only_emg(emg_data, position_data, window_length, initial_time, final_time, level, back_time):
    # generates the data fot the MIMO emg to position

    i = initial_time
    n_channels = emg_data.shape[1]
    X = np.empty(((n_channels * back_time * (level + 1)), 0))
    Y = np.empty((3, 0))

    while i <= final_time:

        # print(position_data[:, i+window_length].shape)
        Y = np.append(Y, position_data[:, i+window_length].reshape((3, 1)), 1)
        

        for channel in range(n_channels):
            
            sliced_emg = get_window_emg(emg_data=emg_data, channel=channel, window_length=window_length, initial_time=i)
            return_matrix = generate_regressors_slice(emg_data_slice=sliced_emg, level=level, back_time=back_time)
            if channel == 0:
                tmp_column_x = return_matrix
            else:
                tmp_column_x = np.append(tmp_column_x, return_matrix, 0)
        
        X = np.append(X, tmp_column_x, 1)
        
        i += 1

    return np.transpose(X), np.transpose(Y)

def generate_multiple_position_regressors_emg_and_position(emg_data, position_data, window_length, initial_time, final_time, level, back_time):
    i = initial_time
    n_channels = emg_data.shape[1]
    X = np.empty(((n_channels * back_time * (level + 1)) + 3, 0))
    Y = np.empty((3, 0))

    while i <= final_time:

        # print(position_data[:, i+window_length].shape)
        Y = np.append(Y, position_data[:, i+window_length].reshape((3, 1)), 1)
        

        for channel in range(n_channels):
            
            sliced_emg = get_window_emg(emg_data=emg_data, channel=channel, window_length=window_length, initial_time=i)
            return_matrix = generate_regressors_slice(emg_data_slice=sliced_emg, level=level, back_time=back_time)
            if channel == 0:
                tmp_column_x = return_matrix
            else:
                tmp_column_x = np.append(tmp_column_x, return_matrix, 0)

        tmp_column_x = np.append(tmp_column_x, position_data[:,i+window_length-1].reshape((3,1)), 0)
        
        X = np.append(X, tmp_column_x, 1)
        
        i += 1

    return np.transpose(X), np.transpose(Y)


