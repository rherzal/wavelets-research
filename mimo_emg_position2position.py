from utils import *

filename = ".\\intention_dataset\\freestyle_01.h5"


sampling_rate = 3000
number_of_movements = 10
scales = 1


joint_data, emg_data, sampling_rate, time = interpolate_measurements(load_measurements(filename=filename), sampling_rate)

channel2print = 0
back_time = 2
levels = 1


position = convert_joint_data(joint_data=joint_data)



# Predicting Intention one timestep with emg and current position

X, Y = generate_multiple_position_regressors_emg_and_position(emg_data=emg_data, position_data=position, window_length=50, initial_time=0, final_time=350, level=levels, back_time=back_time)
reg = LinearRegression().fit(X, Y)
Y1 = reg.predict(X)
print(Y1[150:200, 0])   
plt.title("Training Data")
plt.subplot(3, 1, 1)
plt.plot(time, position[0,:])
plt.plot(time[49:400], Y1[:, 0])
plt.ylabel("X position")
plt.subplot(3, 1, 2)
plt.ylabel("Y position")
plt.plot(time, position[1, :])
plt.plot(time[49:400], Y1[:, 1])
plt.subplot(3, 1, 3)
plt.ylabel("Z position")
plt.plot(time, position[2, :])
plt.plot(time[49:400], Y1[:, 2])
plt.figure()
plt.title("Validation data")
X1, Y2 = generate_multiple_position_regressors_emg_and_position(emg_data=emg_data, position_data=position, window_length=50, initial_time=350, final_time=480, level=levels, back_time=back_time)
Y2 = reg.predict(X1)
plt.subplot(3, 1, 1)
plt.plot(time, position[0,:])
plt.plot(time[399:530], Y2[:, 0])
plt.ylabel("X position")
plt.subplot(3, 1, 2)
plt.ylabel("Y position")
plt.plot(time, position[1, :])
plt.plot(time[399:530], Y2[:, 1])
plt.subplot(3, 1, 3)
plt.ylabel("Z position")
plt.plot(time, position[2, :])
plt.plot(time[399:530], Y2[:, 2])


plt.show()