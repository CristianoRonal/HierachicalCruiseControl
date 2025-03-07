clc;clear;
v0=5:5:35;
x0=0:100:1000;
a0=-3:1:2;
vdes=(10:10:60)/3.6;
n_v = length(v0);
n_x = length(x0);
n_a = length(a0);
n_vdes = length(vdes);

v0_1 = v0(randi([1,n_v]));
% a0_1 = a0(randi([1,n_a]));
a0_1=0;
x0_1 = x0(randi([1,n_x]));
vdes_1 = vdes(randi([1,n_vdes]));


fprintf('测试初速度 %d\n',v0_1*3.6);
fprintf('测试初始加速度 %d\n',a0_1);
fprintf('测试初始位置 %d\n',x0_1);
fprintf('测试期望预稳速度 %d\n',vdes_1*3.6);

[v_s,a_s,v_t,a_t]=Speed_replan(v0_1,a0_1,x0_1,vdes_1);
fprintf('测试结果1：%d\n',v_t*3.6);
fprintf('测试结果2：%d\n',a_t);