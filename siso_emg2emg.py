from utils import *

filename = ".\\intention_dataset\\Tassos_elbow_flex_02.h5"


sampling_rate = 3000
number_of_movements = 10
scales = 1


joint_data, emg_data, sampling_rate, time = interpolate_measurements(load_measurements(filename=filename), sampling_rate)

channel2print = 1
back_time = 5
levels = 0


X, Y = generate_regressors(emg_data=emg_data, channel=channel2print, window_length=50, initial_time=0,final_time=350, level=levels, back_time=back_time)

reg = LinearRegression().fit(X, Y)

Y1 = reg.predict(X)

plt.title("training data channel" + str(channel2print))
plt.xlabel("time")
plt.ylabel("signal value")
plt.plot(time, emg_data[:,channel2print])
plt.plot(time[50:401], Y1[:])


print(Y1[0])

X1, Y1 = generate_regressors(emg_data=emg_data, channel=channel2print, window_length=50, initial_time=350, final_time=480, level=levels, back_time=back_time)
Y2 = reg.predict(X1)


plt.figure()
plt.title("validation data channel " + str(channel2print))
plt.xlabel("time")
plt.ylabel("signal valuel")

plt.plot(time, emg_data[:,channel2print])
plt.plot(time[400:531], Y2[:])

plt.show()
