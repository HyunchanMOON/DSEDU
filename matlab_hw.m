clc;
clear;
close all;

% pulse 1 - cosine squared
T= 2*pi;
N = 256;
t = linspace(0, T, N);
x = square(0.6*t);
plot(t/pi, x, '.-', t/pi, cos(0.6*t))
xlabel('t / \pi')
grid on

X = fourier(x);

figure()
bar(abs(X))
n_arr = [1,3,5,10];

xt = X';
figure()
for i=1:length(n_arr)
    rex = wavsyn(n_arr(i), X);
    plot(rex)
    hold on
end

legend('1', '3', '5', '10')


for i=1:N
    fprintf('C(%d) ',i-1)
    fprintf(num2str(xt(i)))
    fprintf(' |C(%d)| ',i-1)
    fprintf(num2str(abs(xt(i))))
    fprintf('\n')
    fprintf('\n')
end




%% pulse 2 - duty 25%

clc;
clear;
close all;

N=256;
T= 2*pi;

t = linspace(0, T, N);

pulsewidth = 1e-0;
pulseperiods = [0:2*pi]*4e-0;

x = pulstran(t,pulseperiods,@rectpuls,pulsewidth);

plot(t,x)


X = fourier(x);
figure()
bar(abs(X))

n_arr = [1,3,5,10];

xt = X';
figure()
for i=1:length(n_arr)
    rex = wavsyn(n_arr(i), X);
    plot(rex)
    hold on
end

legend('1', '3', '5', '10')


for i=1:N
    fprintf('C(%d) ',i-1)
    fprintf(num2str(xt(i)))
    fprintf(' |C(%d)| ',i-1)
    fprintf(num2str(abs(xt(i))))
    fprintf('\n')
    fprintf('\n')
end



%% pulse 3 - duty 12.5%

clc;
clear;
close all;

N=256;
T= 2*pi;

t = linspace(0, T, N);

pulsewidth = 0.5e-0;
pulseperiods = [0:2*pi]*4e-0;

x = pulstran(t,pulseperiods,@rectpuls,pulsewidth);

plot(t,x)


X = fourier(x);
figure()
bar(abs(X))

n_arr = [1,3,5,10];

xt = X';
figure()
for i=1:length(n_arr)
    rex = wavsyn(n_arr(i), X);
    plot(rex)
    hold on
end

legend('1', '3', '5', '10')


for i=1:N
    fprintf('C(%d) ',i-1)
    fprintf(num2str(xt(i)))
    fprintf(' |C(%d)| ',i-1)
    fprintf(num2str(abs(xt(i))))
    fprintf('\n')
    fprintf('\n')
end


%% pulse 4 - one cycle sine

clc;
clear;
close all;

N=256;
T= 2*pi;

Fs = N/T;
t = linspace(0, T, N);
x = sin(2*pi*Fs*t);
figure();
plot(t,x)
X = fourier(x);

figure()
bar(abs(X))

n_arr = [1,3,5,10];

xt = X';
figure()
for i=1:length(n_arr)
    rex = wavsyn(n_arr(i), X);
    plot(rex)
    hold on
end

legend('1', '3', '5', '10')




for i=1:N
    fprintf('C(%d) ',i-1)
    fprintf(num2str(xt(i)))
    fprintf(' |C(%d)| ',i-1)
    fprintf(num2str(abs(xt(i))))
    fprintf('\n')
    fprintf('\n')
end


%% pulse 5 - two cycle sine

clc;
clear;
close all;

N=256;
T= 2*pi;

Fs = N/T;
t = linspace(0, T, N);
x = sin(2*2*pi*Fs*t);
figure();
plot(t,x)
X = fourier(x);


figure()
bar(abs(X))

n_arr = [1,3,5,10];

xt = X';
figure()
for i=1:length(n_arr)
    rex = wavsyn(n_arr(i), X);
    plot(rex)
    hold on
end

legend('1', '3', '5', '10')

for i=1:N
    fprintf('C(%d) ',i-1)
    fprintf(num2str(xt(i)))
    fprintf(' |C(%d)| ',i-1)
    fprintf(num2str(abs(xt(i))))
    fprintf('\n')
    fprintf('\n')
end


%% pulse 6 - triangle

clc;
clear;
close all;

N=256;
T= 2*pi;

Fs = N/T;
t = 0:1/Fs:T-1/Fs;
y = sawtooth(2*pi*t,1/2);


spread = abs(max(y) - min(y));
nexttile;
floor = max(y) - 0.25 * spread;
x = y;
x(x<floor) = floor;
plot(t,x);
title('25% Duty Cycle');


X = fourier(x);


figure()
bar(abs(X))


n_arr = [1,3,5,10];

xt = X';
figure()
for i=1:length(n_arr)
    rex = wavsyn(n_arr(i), X);
    plot(rex)
    hold on
end

legend('1', '3', '5', '10')


for i=1:N
    fprintf('C(%d) ',i-1)
    fprintf(num2str(xt(i)))
    fprintf(' |C(%d)| ',i-1)
    fprintf(num2str(abs(xt(i))))
    fprintf('\n')
    fprintf('\n')
end





