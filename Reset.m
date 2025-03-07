%参数重置，返回为初始状态与动作
%初始化参数
clc;clear;
lower_bound_v0 = 60/3.6;    % lower bound of initial velocity for ego car (m/s) 
upper_bound_v0 = 120/3.6;   % upper bound of initial velocity for ego car (m/s) 
t_gap = 1.4;     % 车头时距常数
D_default = 10; % 默认设置的初始间距
free_mode = 0;  % 自由流模式（1km）无前车
levh_mode = 1;  % 有前车模式
prob_levh = 0.9; % 非自由流即有前车模式的概率是0.9
mode = bernoulli_sample(prob_levh);
x0_ego = 0;    % initial position for ego car (m)-本车的初始位置为0(固定)
% v0_ego = lower_bound_v0 + (upper_bound_v0-lower_bound_v0)*rand();    % initial velocity for ego car (m/s)-本车的初速度(均匀分布采样)
v0_ego = 100/3.6;   % initial velocity for ego car  100km/h
v0_lead = lower_bound_v0 + (upper_bound_v0-lower_bound_v0)*rand();   % initial velocity for lead car (m/s)-前车的初速度(均匀分布采样)
v_des = sample_integer_interval(20,60,10)/3.6;   % 低附着路段的期望预稳速度m/s (20-60km/h整数)
acc_p = 0;  % 前车加速行为
dec_p = 1;  % 前车减速行为
prob_accp = 0.7;  % 前车减速行为的概率是0.7
beh = bernoulli_sample(prob_accp);
t1 = 5 + (100-5)*rand();   % 前车开始加减速行为的时间 从5s到100s之间均匀分布


if beh==acc_p
    ap_maxmin = 0 + (3-0)*rand();   % 前车加速度在0-3m/s^2间均匀分布
    dt = 0 + (4-0)*rand();     % 前车保持加速的时长在0-4s内均匀分布
elseif beh==dec_p
    ap_maxmin = -3 + (0-(-4))*rand(); % 前车加速度在-3-0m/s^2间均匀分布
    dt = 0 + (2-0)*rand();     % 前车保持最大减速度的时长在0-2s内均匀分布
end

if mode==free_mode
    x0_lead = 1000;   % initial position for lead car (m)-前车的初始位置
elseif mode==levh_mode
    x0_lead = max(min(t_gap*v0_ego+D_default+5*randn(),80),15);
end


v_set = 100/3.6;
amin_ego = -3;
amax_ego = 2;
Ts = 0.1;
Tf = 200;
model_name='DRL_ACC';
load_system(model_name)%加载模型
%设置仿真初始化参数
%求解方式，以及求解步长
%设置输入的初值：

%初始动作
acceleration=0;

set_param('DRL_ACC/Gain', 'Gain',mat2str(acceleration));
set_param(model_name,'StopTime',num2str(Tf))
set_param(model_name,'SolverType','Fixed-step','FixedStep',num2str(Ts))
set_param(model_name, 'SimulationCommand', 'start');%开始,此时只执行一个步长
set_param(model_name, 'SimulationCommand', 'pause');%暂停

function sample = bernoulli_sample(p)
%BERNOULLI_SAMPLE  Generates a sample from a Bernoulli distribution.
%
%   SAMPLE = BERNOULLI_SAMPLE(P) returns a 1 with probability P and a 0 with
%   probability 1-P.  P must be between 0 and 1.

% Input validation
if p < 0 || p > 1
    error('Probability P must be between 0 and 1.');
end

% Generate a random number between 0 and 1
u = rand();

% Check if the random number is less than or equal to the probability
if u <= p
    sample = 1;  
else
    sample = 0;  
end
end

function sample = sample_integer_interval(lower_bound, upper_bound, interval)
%SAMPLE_INTEGER_INTERVAL Samples integer values with a specified interval.
%
%   SAMPLE = SAMPLE_INTEGER_INTERVAL(LOWER_BOUND, UPPER_BOUND, INTERVAL)
%   samples an integer value within the range [LOWER_BOUND, UPPER_BOUND]
%   with the specified interval.  The sampled value will be
%   LOWER_BOUND + k * INTERVAL, where k is a non-negative integer
%   such that the result is within the specified bounds.
%
%   Example: sample_integer_interval(10, 60, 10) will sample from the set
%   {10, 20, 30, 40, 50, 60}.

% Input validation
if interval <= 0
    error('Interval must be a positive value.');
end

if lower_bound >= upper_bound
    error('Lower bound must be less than the upper bound.');
end

if mod(lower_bound, interval) ~= 0
    error('Lower bound must be a multiple of the interval.')
end

% Determine the number of possible values
num_values = floor((upper_bound - lower_bound) / interval) + 1;

% Generate a random integer index
random_index = randi([1, num_values]);

% Calculate the sampled value
sample = lower_bound + (random_index - 1) * interval;

end