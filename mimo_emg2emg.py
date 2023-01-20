from utils import *


filename = ".\\intention_dataset\\Tassos_elbow_flex_01.h5"


sampling_rate = 3000
number_of_movements = 10
scales = 1


joint_data, emg_data, sampling_rate, time = interpolate_measurements(load_measurements(filename=filename), sampling_rate)



channel2print = 0
back_time = 2
levels = 3


X, Y = generate_multiple_regressors(emg_data=emg_data, window_length=50, initial_time=0, final_time=350, level=levels, back_time=back_time)

reg = LinearRegression().fit(X, Y)

Y1 = reg.predict(X)


plt.title("training data channel" + str(channel2print))
plt.xlabel("time")
plt.ylabel("signal value")
plt.plot(time, emg_data[:,channel2print])
plt.plot(time[50:401], Y1[:, channel2print])

X1, Y1 = generate_multiple_regressors(emg_data=emg_data, window_length=50, initial_time=350, final_time=480, level=levels, back_time=back_time)
Y2 = reg.predict(X1)


plt.figure()
plt.title("validation data channel " + str(channel2print))
plt.xlabel("time")
plt.ylabel("signal valuel")

plt.plot(time, emg_data[:,channel2print])
plt.plot(time[399:530], Y2[:,channel2print])

plt.show()
