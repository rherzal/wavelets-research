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

levels = 5
back_time = 2

channel2print = 3
window_length = 50

train_k = 350



X_train, Y = generate_regressors(emg_data=emg_data, channel=channel2print, window_length=window_length, initial_time=0,final_time=train_k-window_length, level=levels, back_time=back_time)

reg = LinearRegression().fit(X_train, Y)

Y1 = reg.predict(X_train)

X_val, Y = generate_regressors(emg_data=emg_data, channel=channel2print, window_length=window_length, initial_time=train_k+1,final_time=500-window_length, level=levels, back_time=back_time)
Y_val = reg.predict(X_val)


fig, ax = plt.subplots()
plt.subplot(1,2,1)
plt.title('Training Data')
plt.plot(time[0:train_k], emg_data[0:train_k, channel2print], label='original data')
plt.plot(time[window_length:train_k+1], Y1[:], label='predicted data')
plt.legend()



plt.subplot(1,2,2)
plt.title('Validation Data')
plt.plot(time[train_k+1:500], emg_data[train_k+1:500, channel2print], label='original data')
plt.plot(time[train_k+window_length+1:501], Y_val[:], label='predicted data')
# plt.show()

forward_horizon = 10
valY = emg_data[train_k+1:500, channel2print]

new_data = valY[0:window_length]
# X1 = generate_regressors_slice(emg_data_slice=new_data, level=level, back_time=back_time)
generated_data = np.empty([1,0])

# forward_horizon = 10
# for i in range(window_length, valY.size, forward_horizon):
#     new_data = valY[i-window_length:i]
#     for j in range(forward_horizon):
#         X1 = generate_regressors_slice(emg_data_slice=new_data[-window_length:], level=levels, back_time=back_time)
#         Ypred = reg.predict(np.transpose(X1))
#         new_data = np.append(new_data, Ypred)
#         generated_data = np.append(generated_data, Ypred)

fig1, ax1 = plt.subplots()
plt.title('Longer Horizon')
plt.plot(time[train_k+1:500], valY, color='black')
# plt.plot(time[train_k+window_length+1:500+1], generated_data[:])

rmse = np.empty((10,1))

for forward_horizon in range(1, 11):
    generated_data = np.empty([1,0])
    for i in range(window_length, valY.size, forward_horizon):
        new_data = valY[i-window_length:i]
        for j in range(forward_horizon):
            X1 = generate_regressors_slice(emg_data_slice=new_data[-window_length:], level=levels, back_time=back_time)
            Ypred = reg.predict(np.transpose(X1))
            new_data = np.append(new_data, Ypred)
            generated_data = np.append(generated_data, Ypred)
    plt.plot(time[train_k+window_length+1:train_k+window_length+1 + generated_data.size], generated_data[:])
    rmse[forward_horizon-1] = np.sqrt(np.mean(np.square(emg_data[train_k+window_length+1:train_k+window_length+1 + generated_data.size, channel2print] - generated_data[:])))

plt.legend(['Orig', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

fig2, ax2 = plt.subplots()
plt.title('RMSE')
plt.plot(range(1,11), rmse)


plt.show()