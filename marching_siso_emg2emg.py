from utils import *

def generate_marching_input(emg_data, channel, window_length, initial_time, final_time, level, back_time):
    # generates the data for the SISO, emg in emg out regression 

    i = initial_time
    X = np.empty((((level + 1) * back_time), 0))

    while i <= final_time:
        sliced_emg = get_window_emg(emg_data, channel, window_length, i)
        return_matrix = generate_regressors_slice(emg_data_slice=sliced_emg, level=level, back_time=back_time)
        X = np.append(X, return_matrix, 1)
        i += 1
    return np.transpose(X)



filename = ".\\intention_dataset\\Tassos_elbow_flex_01.h5"


sampling_rate = 3000
number_of_movements = 10
scales = 1


joint_data, emg_data, sampling_rate, time = interpolate_measurements(load_measurements(filename=filename), sampling_rate)

channel2print = 3
back_time = 2
levels = 5
forward_horizon = 200



X, Y = generate_regressors(emg_data=emg_data, channel=channel2print, window_length=100, initial_time=0,final_time=350, level=levels, back_time=back_time)

reg = LinearRegression().fit(X, Y)

plt.plot(time, emg_data[:, channel2print])

Y1 = reg.predict(X)

# plt.plot(time[99:450], Y1[:])

X1, Y1 = generate_regressors(emg_data=emg_data, channel=channel2print, window_length=100, initial_time=0,final_time=0, level=levels, back_time=back_time)

Y1 = reg.predict(X1)


Z = np.zeros((1, 6))
Z[0, channel2print] = Y1

print(Z)


new_data = emg_data[0:100,:]

for i in range(forward_horizon):
    X1= generate_marching_input(emg_data=new_data, channel=channel2print, window_length=100, initial_time=i,final_time=i, level=levels, back_time=back_time)
    Y1 = reg.predict(X1)
    print(Y1)
    Z = np.zeros((1, 6))
    Z[0, channel2print] = Y1

    new_data = np.append(new_data, Z, 0)


print(new_data.shape)

plt.axvline(x = time[99], color = 'b', label = 'axvline - full height')

plt.plot(time[0:300], new_data[:,channel2print])

plt.show()
