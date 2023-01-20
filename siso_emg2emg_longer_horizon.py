from utils import *

from matplotlib.widgets import Slider, Button



filename = ".\\intention_dataset\\freestyle_01.h5"


sampling_rate = 3000
number_of_movements = 10
scales = 1


joint_data, emg_data, sampling_rate, time = interpolate_measurements(load_measurements(filename=filename), sampling_rate)

channel2print = 1
back_time = 5
levels = 2

forward_horizon = 10

def generate_regressors_longer_horizon(emg_data, channel, window_length, initial_time, final_time, level, back_time, forward_horizon):
    # generates the data for the SISO, emg in emg out regression 

    i = initial_time
    X = np.empty((((level + 1) * back_time), 0))
    Y = np.empty((forward_horizon, 0))
    while i <= final_time:
        sliced_emg = get_window_emg(emg_data, channel, window_length, i)
        return_matrix = generate_regressors_slice(emg_data_slice=sliced_emg, level=level, back_time=back_time)
        X = np.append(X, return_matrix, 1)
        return_matrix_y = emg_data[i+window_length:i+window_length+forward_horizon, channel]
        Y = np.append(Y, return_matrix_y.reshape((10,1)), 1)

        i += 1
    return np.transpose(X), np.transpose(Y)


X, Y = generate_regressors_longer_horizon(emg_data=emg_data, channel=channel2print, window_length=50, initial_time=0,final_time=350, level=levels, back_time=back_time, forward_horizon=forward_horizon)

reg = LinearRegression().fit(X, Y)


Y1 = reg.predict(X)

print(Y1.shape)

fig, ax = plt.subplots()
fig.subplots_adjust(left=0.25, bottom=0.25)

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])

slider = Slider(
    ax=axfreq,
    label='i',
    valmin=0,
    valmax=350,
    valinit=0,
)


ax.plot(time, emg_data[:,channel2print])

def update(val):
    ax.plot(time[(49+int(slider.val)):(59+int(slider.val))], Y1[int(slider.val),:])

slider.on_changed(update)


plt.title("training data channel" + str(channel2print))
plt.xlabel("time")
plt.ylabel("signal value")


plt.plot(time, emg_data[:,channel2print])
plt.plot(time[49:59], Y1[0,:])





for i in range(350):
    plt.plot(time[(49+i):(59+i)], Y1[i,:])

plt.show()



