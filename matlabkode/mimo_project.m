%% MIMO project
clear all;
close all;
clc;

set_groot;
load("mimo_project.mat");

%% Define LFM pulses with upchirp and downchirp

t = 0:1/fs:T_p; % Interval definition of transfer signal
alpha = B/T_p; % Chirp rate
s_tx_up = exp(1i*2*pi*((fc - B/2)*t + alpha*t.^2 / 2)); % Transfer signal up chirp
s_tx_down = exp(1i*2*pi*((fc + B/2)*t - alpha*t.^2 / 2)); % Transfer signal down chirp

%% stft upchirp
figure;
stft(s_tx_up, fs);

%% stft downchirp
figure;
stft(s_tx_down, fs)


%% Upchirp LFM pulse
figure;
subplot(2,1,1);
zero_vec = complex(zeros(1,200));
s_tx_up_plot = [zero_vec s_tx_up zero_vec];
t_plot = 0:1/fs:(length(s_tx_up_plot)-1)/fs;

plot(t_plot*1000, real(s_tx_up_plot));
hold on;
plot(t_plot*1000, abs(s_tx_up_plot));
hold off;
ylim([-1.1,1.1]);
grid("on");
title("LFM pulse");
xlabel("Time [ms]");
ylabel("Amplitude");


subplot(2,2,3);

N = length(s_tx_up);
pxx = 1/(N*fs)*abs(fftshift(fft(s_tx_up))).^2;
f = (fs/N*(-N/2:N/2-1)/1000);
pxx = pxx / max(pxx(:));
plot(f, db(pxx));
xlim([0, max(f)]);
grid("on");
title("Periodogram");
xlabel("Frequency [kHz]")
ylabel("Power [dB]")


subplot(2,2,4);
[c_, lags] = xcorr(s_tx_up, "normalized");

plot(lags/fs*1000, abs(c_));
grid("on");
title("Autocorrelation");
xlabel("Lag [ms]");
ylabel("Correlation coefficient")


%% Downchirp LFM pulse
figure;
subplot(2,1,1);
zero_vec = complex(zeros(1,200));
s_tx_down_plot = [zero_vec s_tx_down zero_vec];
t_plot = 0:1/fs:(length(s_tx_down_plot)-1)/fs;

plot(t_plot*1000, real(s_tx_down_plot));
hold on;
plot(t_plot*1000, abs(s_tx_down_plot));
hold off;

ylim([-1.1,1.1]);
grid("on");
title("LFM pulse");
xlabel("Time [ms]");
ylabel("Amplitude");


subplot(2,2,3);

N = length(s_tx_down);
pxx = 1/(N*fs)*abs(fftshift(fft(s_tx_down))).^2;
pxx = pxx / max(pxx(:));
f = (fs/N*(-N/2:N/2-1)/1000);
plot(f, db(pxx));
xlim([0, max(f)]);
grid("on");
title("Periodogram");
xlabel("Frequency [kHz]")
ylabel("Power [dB]")


subplot(2,2,4);
[c_, lags] = xcorr(s_tx_down, "normalized");

plot(lags/fs*1000, abs(c_));
grid("on");
title("Autocorrelation");
xlabel("Lag [ms]");
ylabel("Correlation coefficient")


%% Pulse compression
% Selecting channel 16 from first transmit corresponding to upchirped tx signal
x = tdma_data(:, 16, 1);
t_x = 0:1/fs:(length(x)-1)/fs;

figure;
subplot(2,1,1);
plot(t_x*1000, abs(x));

grid("on");
title("Before pulse compression")
xlabel("Time [ms]");
ylabel("Magnitude");

x_compressed = pulse_compression(s_tx_up, x);

subplot(2,1,2);
plot(t_x*1000, abs(x_compressed));

grid("on");
title("After pulse compression")
xlabel("Time [ms]");
ylabel("Magnitude");


%% FWHM of peak pulse compressed
width = fwhm_width(x_compressed, 8, fs) * 1000; % ms


%% Virtual array
figure;
subplot(3,1,1);

scatter(rx_pos, zeros(1,32), 100, "x", "LineWidth",2);
hold on;
scatter(tx_pos, zeros(1,2), 100, "d", "LineWidth",2)
hold off;
grid("on");

xlabel("x pos [m]");
ylabel("y pos [m]");

xlim([-0.6, 0.601])
title("Physical array")

virtual_array = zeros(2,32);
for i=1:2
    for j=1:32
        virtual_array(i,j) = (tx_pos(i) - rx_pos(j))/2;
    end
end

subplot(3,1,2);
scatter(virtual_array(1,:), zeros(1,32), 15, "filled", "LineWidth",2);

grid("on");

xlabel("x pos [m]");
ylabel("y pos [m]");

xlim([-0.6, 0.601])
title("Virtual array one transmitter")


subplot(3,1,3);
scatter(virtual_array(1,:), zeros(1,32), 15, "filled", "LineWidth",2);
hold on;
scatter(virtual_array(2,:), zeros(1,32), 15, "filled", "LineWidth",2);
hold off;

grid("on");

xlabel("x pos [m]");
ylabel("y pos [m]");

xlim([-0.6, 0.601])
title("Virtual array two transmitters")


%% Measure resolution
d = (virtual_array(2,1) - virtual_array(1,1));

lambda = c/fc;

R = 4; % m
L = abs(rx_pos(1) - rx_pos(32)); % Length of array
dbeta = lambda / L; % lateral resolution radians
dx = R * dbeta; % lateral resolution m at 4m 
dy = c / (2*B); % axial resolution

%% CDMA (Code Division Multiple Access) dataset
cdma_data_compressed = complex(zeros(N_t, N_rx, 2));
for rx=1:N_rx
    cdma_data_compressed(:,rx,1) = pulse_compression(s_tx_down, cdma_data(:,rx));
    cdma_data_compressed(:,rx,2) = pulse_compression(s_tx_up, cdma_data(:,rx));
end

%% TDMA (Time Division Multiple Access) dataset for 2 and one transmits
tdma_data_compressed = complex(zeros(N_t, N_rx, 2));
tdma_data_compressed_one = complex(zeros(N_t, N_rx, 1));
for rx=1:N_rx
    tdma_data_compressed(:,rx,1) = pulse_compression(s_tx_up, tdma_data(:,rx,1));
    tdma_data_compressed(:,rx,2) = pulse_compression(s_tx_up, tdma_data(:,rx,2));
    tdma_data_compressed_one(:,rx) = pulse_compression(s_tx_up, tdma_data(:,rx,1));
end

%% Define grid and beamform the datasets
d = abs(rx_pos(1) - rx_pos(2)); 
dxy = d/6;

x = -5:dxy:5+dxy; 
y = 0:dxy:5+dxy;

tdma_image_two = beamform_das(x, y, tdma_data_compressed, fs, c, rx_pos, tx_pos);
tdma_image_one = beamform_das(x, y, tdma_data_compressed_one, fs, c, rx_pos, tx_pos);
cdma_image = beamform_das(x, y, cdma_data_compressed, fs, c, rx_pos, tx_pos);

%% Vizualize beamformed data

show_image(x, y, tdma_image_two, "TDMA dataset using both transmit");

show_image(x, y, tdma_image_one, "TDMA dataset using only first transmit");

show_image(x, y, cdma_image, "CDMA dataset, MIMO setup");

%% Vizualize CDMA before and after pulse compression
figure;
subplot(2,1,1)
plot(t_x*1000, abs(cdma_data(:,16)));
grid("on");
xlabel("Time [ms]");
ylabel("Magnitude");
title("Downchirp channel 16, before pulse compression");

subplot(2,1,2)
plot(t_x*1000, abs(cdma_data_compressed(:,16,2)));
grid("on");
xlabel("Time [ms]");
ylabel("Magnitude");
title("Downchirp channel 16, after pulse compression");

%% Measure resolution of the different images
reflector = max(tdma_image_two(:));
[row, col] = find(tdma_image_two==reflector);
    
R = sqrt(y(row)^2 + x(col)^2);
L = abs(rx_pos(1) - rx_pos(32));

dx_t = R * lambda / L;

dy_t = c / (2*B);


[dx_m, dy_m] = measure_resolution(tdma_image_two, x, y);
disp("TDMA two transmits:");
disp("dx_m = " + dx_m + ", dx_t = " + dx_t);
disp("dy_m = " + dy_m + ", dy_t = " + dy_t);

[dx_m, dy_m] = measure_resolution(tdma_image_one, x, y);
disp("TDMA one transmit:");
disp("dx_m = " + dx_m + ", dx_t = " + dx_t);
disp("dy_m = " + dy_m + ", dy_t = " + dy_t);

[dx_m, dy_m] = measure_resolution(cdma_image, x, y);
disp("CDMA:");
disp("dx_m = " + dx_m + ", dx_t = " + dx_t);
disp("dy_m = " + dy_m + ", dy_t = " + dy_t);

%% Center cut on scatterer

reflector = max(cdma_image(:));
[row, col] = find(cdma_image==reflector);

reflector2 = max(tdma_image_two(:));
[row2, col2] = find(tdma_image_two==reflector2);

figure;
subplot(2,1,1);
plot(x,db(cdma_image(row,:)));
hold on; plot(x, db(tdma_image_two(row2,:))); hold off;
ylim([-80,0]);
grid("on");
xlabel("x");
ylabel("Power [dB]");
legend("CDMA", "TDMA 2tx");
xticks([-5,-2.5,0,2.5,5])

subplot(2,1,2);
plot(y,db(cdma_image(:,col)));
hold on; plot(y, db(tdma_image_two(:,col2))); hold off;

ylim([-80,0]);
grid("on");
xlabel("y");
ylabel("Power [dB]");
legend("CDMA", "TDMA 2tx");

%% Cross talk 

[cc, lags] = xcorr(s_tx_up, s_tx_down, "normalized");

figure;
t_lags = [-flip(t), t(2:end)];
plot(t_lags*1000, abs(cc));
ylim([0,1.0]);
title("Max: " + round(max(abs(cc)), 3) + ", BT = " + B*T_p);
ylabel("Correlation coefficient")
xlabel("Lags [ms]")
grid("on");

%% DAS with hamming

tdma_data_compressed_hamming = complex(zeros(N_t, N_rx, 2));
w_rx = hamming(N_t);
w_tx = hamming(length(s_tx_up)).';
s_tx_up_tapered = w_tx.*s_tx_up;
for rx=1:N_rx
    tdma_data_compressed_hamming(:,rx,1) = pulse_compression(s_tx_up_tapered, w_rx.*tdma_data(:,rx,1));
    tdma_data_compressed_hamming(:,rx,2) = pulse_compression(s_tx_up_tapered, w_rx.*tdma_data(:,rx,2));
end

%%
tdma_image_two_hamming = beamform_das(x, y, tdma_data_compressed_hamming, fs, c, rx_pos, tx_pos);

show_image(x, y, tdma_image_two_hamming, "TDMA dataset using both transmit, hamming taper");

[dx_m, dy_m] = measure_resolution(tdma_image_two_hamming, x, y);
disp("dx_m = " + dx_m + ", dx_t = " + dx_t);
disp("dy_m = " + dy_m + ", dy_t = " + dy_t);

%% DAS with hamming one transmit

tdma_data_compressed_one_hamming = complex(zeros(N_t, N_rx, 1));
w_rx = hamming(N_t);
w_tx = hamming(length(s_tx_up)).';
s_tx_up_tapered = w_tx.*s_tx_up;
for rx=1:N_rx
    tdma_data_compressed_one_hamming(:,rx) = pulse_compression(s_tx_up_tapered, w_rx.*tdma_data(:,rx,1));
end

%%
tdma_image_one_hamming = beamform_das(x, y, tdma_data_compressed_one_hamming, fs, c, rx_pos, tx_pos);

show_image(x, y, tdma_image_one_hamming, "TDMA dataset using one transmit, hamming taper");
exportgraphics(gca(), f_dir + "tdma_image_one_hamming.pdf","ContentType","vector","BackgroundColor","none");

[dx_m, dy_m] = measure_resolution(tdma_image_one_hamming, x, y);
disp("dx_m = " + dx_m + ", dx_t = " + dx_t);
disp("dy_m = " + dy_m + ", dy_t = " + dy_t);

%% Beamforming using virtual array positioons

[N_t, N_rx, N_tx] = size(tdma_data_compressed);

n_x = length(x);
n_y = length(y);

full_image = complex(zeros(n_x,n_y));
for tx=1:N_tx % for all transmit
    for rx=1:N_rx % for all recieve
        tmp_img = complex(zeros(n_x,n_y));
        for i=1:n_x
            for j=1:n_y
                rc = sqrt((x(i) + virtual_array(tx, rx))^2 + (y(j) - 0)^2);
                t_delay = (2*rc) / c;

                t_sample = round(t_delay * fs); % nearest neighbour
    
                if t_sample > 0 && t_sample < N_t
                    tmp_img(i,j) = tdma_data_compressed(t_sample,rx,tx);
                end
            end 
        end
        full_image = full_image + tmp_img;
    end 
end
full_image = (full_image / max(full_image(:))).';

show_image(x, y, full_image, "TDMA dataset using virtual array positions");

[dx_m, dy_m] = measure_resolution(full_image, x, y);
disp("dx_m = " + dx_m + ", dx_t = " + dx_t);
disp("dy_m = " + dy_m + ", dy_t = " + dy_t);


%% Functions %%
function show_image(x, y, full_image, title_text)
    figure;
    imagesc(x, y, db(full_image)) ;
    colormap("default");
    colorbar();
    clim([-50,0]);
    xlabel("x");
    ylabel("y", "Rotation",0);
    xticks([-5,-2.5,0,2.5,5])
    title(title_text);
end

function data_compressed = pulse_compression(s_tx, s_rx)
    [N_t, N_h] = size(s_rx);
    % Define array
    match_filtered_data = zeros(2*N_t-1,N_h);
    % Apply match filtering
    for tau = 1:N_h
        match_filtered_data(:,tau) = xcorr(s_rx(:,tau), s_tx);
    end
    % Select only positive delays
    data_compressed = match_filtered_data(N_t:end,:);
end

function full_image = beamform_das(x, y, data, fs, c, rx_pos, tx_pos)
    [N_t, N_rx, N_tx] = size(data);

    n_x = length(x);
    n_y = length(y);
    
    full_image = complex(zeros(n_x,n_y));
    for tx=1:N_tx % for all transmit
        for rx=1:N_rx % for all recieve
            tmp_img = complex(zeros(n_x,n_y));
            for i=1:n_x
                for j=1:n_y
                    r_tx = sqrt((x(i) - tx_pos(tx))^2 + (y(j) - 0)^2);
                    r_rx = sqrt((x(i) - rx_pos(rx))^2 + (y(j) - 0)^2);
                    t_delay = (r_tx + r_rx) / c;
                    t_sample = round(t_delay * fs); % nearest neighbour
        
                    if t_sample > 0 && t_sample < N_t
                        tmp_img(i,j) = data(t_sample,rx,tx);
                    end
                end 
            end
            full_image = full_image + tmp_img;
        end 
    end
    full_image = (full_image / max(full_image(:))).';
end

function width = fwhm_width(y, n_up, fs)
    y_upsampled = abs(resample(y, n_up, 1));

    halfMaxValue = max(y_upsampled) / 2;

    leftIndex = find(y_upsampled >= halfMaxValue, 1, 'first');
    rightIndex = find(y_upsampled >= halfMaxValue, 1, 'last');

    width = (rightIndex -leftIndex) / (n_up*fs);
end

function width = fwhm_t(x, data, n_up)
    data = abs(data);
    data = resample(data, n_up, 1);
    x_up = resample(x, n_up, 1);
    
    max_value = max(data);
    half_max_value = max_value / 2;
    
    left_index = find(data >= half_max_value, 1, 'first');
    right_index = find(data >= half_max_value, 1, 'last');
    
    width = (x_up(right_index) - x_up(left_index));
end

function [dx_m, dy_m] = measure_resolution(full_image, x, y)
    reflector = max(full_image(:));
    [row, col] = find(full_image==reflector);
    
    x_reflector = full_image(row,:);
    y_reflector = full_image(:,col);
    
    dx_m = fwhm_t(x, x_reflector, 8);
    dy_m = fwhm_t(y, y_reflector, 8);
end

