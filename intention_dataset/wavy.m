emg_data = h5read('Tassos_elbow_flex_01.h5', '/emg signals');

t0 = emg_data(1, 1);
sampling_rate = 20000;


for i = 1:size(emg_data, 2)
    emg_data(1, i) = (emg_data(1, i) - t0) / 1000000;
end

equally_spaced_timestamps = 0:(1 / sampling_rate):emg_data(1, end);

new_emg_data = interp1(emg_data(1,:), emg_data(2, :), equally_spaced_timestamps);

plot(equally_spaced_timestamps, new_emg_data);
hold on;
plot(emg_data(1, :), emg_data(2, :));

figure()
cwt(new_emg_data, 'amor', 20000, VoicesPerOctave=20);

