from utils import * 

filename = ".\\intention_dataset\\Tassos_elbow_flex_01.h5"


sampling_rate = 3000
number_of_movements = 10
scales = 1


joint_data, emg_data, sampling_rate, time = interpolate_measurements(load_measurements(filename=filename), sampling_rate)

channel2print = 0
back_time = 10
levels = 1
forward_horizon = 50


plt.show()


A = [0, 1, 2, 3, 4, 5]

coefficient = pywt.wavedec(data=A, wavelet='db4', level=0)

print(coefficient)