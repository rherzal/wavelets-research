from utils import *


def generate_marching_input(emg_data, window_length, initial_time, final_time, level, back_time):
    # generates the data for the MIMO emg in emg out regression
    i = initial_time
    n_channels = emg_data.shape[1]
    X = np.empty(((n_channels * back_time * (level + 1)), 0))
    
    while i <= final_time:
        for channel in range(n_channels):
            sliced_emg = get_window_emg(emg_data=emg_data, channel=channel, window_length=window_length, initial_time=i)
            return_matrix = generate_regressors_slice(emg_data_slice=sliced_emg, level=level, back_time=back_time)
            if channel == 0:
                tmp_column_x = return_matrix
            else:
                tmp_column_x = np.append(tmp_column_x, return_matrix, 0)
        X = np.append(X, tmp_column_x, 1)
        i += 1
    
    return np.transpose(X)






filename = ".\\intention_dataset\\Tassos_elbow_flex_01.h5"


sampling_rate = 3000
number_of_movements = 10
scales = 1


joint_data, emg_data, sampling_rate, time = interpolate_measurements(load_measurements(filename=filename), sampling_rate)

channel2print = 5
back_time = 2
levels = 3

X, Y = generate_multiple_regressors(emg_data=emg_data, window_length=50, initial_time=0, final_time=350, level=levels, back_time=back_time)

reg = LinearRegression().fit(X, Y)

print(emg_data.shape)

Y1 = reg.predict(X)

new_data = emg_data[0:50, :]


forward_horizon = 100

for i in range(forward_horizon):
    new_data_x = generate_marching_input(emg_data=new_data, window_length=50, initial_time=i, final_time=i, level=levels, back_time=back_time)
    new_data_y = reg.predict(new_data_x)
    new_data = np.append(new_data, new_data_y, 0)


plt.axvline(x = time[50], color = 'b', label = 'axvline - full height')
plt.plot(time, emg_data[:,channel2print])

plt.plot(time[0:150], new_data[:,channel2print])

plt.plot(time[50:401], Y1[:, channel2print])

plt.show()