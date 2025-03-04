function [v_t,a_t] = Speed_replan(v0,a0,x0,vdes)
% 全局速度规划（1km全长）
% v0当前时刻车辆的实际行驶速度(m/s)
% a0上一时刻车辆采取的实际加速度(m/s^2)
% x0当前时刻车辆的纵向位置坐标
% 需要知道路段上的限速情况，以及低附区段的期望安全速度
% 现在仅模拟前方出现低附路段，从1km外行驶进入至驶离的过程
% 滑动窗口（规划的长度）设置在1000m
% 假设低附路段的起点位于carsim中路段X Axis的1000m，结束在1100m

% 参数
% Np=30;
vlim=120/3.6;  % 正常路段的限速值为120km/h
s_slide=1000; % 滑动窗口（全局速度规划的总长度）1000m
x_lowmu=1000; % 低附路段的起点X Axis=1000
x_lowmu_end=1100; % 低附路段的终点X Axis=1100
s_lowmu=x_lowmu_end-x_lowmu; % 低附路段的长度100m
s_left=round(x_lowmu-x0);  % 本车当前距离低附路段起点的剩余长度
ds=1;
N=s_slide/ds+1;
delta_t=0.1; % Ts采样时间

% 约束
amax=0.5; % 最大加速度m/s^2
amin=-0.5; % 最小加速度m/s^2
amax1=2;
amin1=-3;

jmax=1;
jmin=-1;
% vdes=60/3.6; % 低附路段的期望安全速度60km/h
vmax=zeros(N,1);

% 当速度差过大时，调整amax和amin
if (vdes^2-vlim^2)/(2*s_slide) < amin1
    amin1=(vdes^2-vlim^2)/(2*s_slide);
end
if (vdes^2-vlim^2)/(2*s_slide) > amax1
    amax1=(vdes^2-vlim^2)/(2*s_slide);
end

if (vdes^2-v0^2)/(2*s_left) < amin
    amin=(vdes^2-v0^2)/(2*s_left);
end
if (vdes^2-v0^2)/(2*s_left) > amax
    amax=(vdes^2-v0^2)/(2*s_left);
end


if x0<s_lowmu
    vmax(1:s_left/ds)=vlim;
    vmax(s_left/ds+1:end)=vdes;
elseif x0<x_lowmu
    vmax(1:s_left/ds)=vlim;
    vmax(s_left/ds+1:(s_left+s_lowmu)/ds)=vdes;
    vmax((s_left+s_lowmu)/ds+1:end)=vlim;
else
    vmax(1:(x_lowmu_end-x0)/ds)=vdes;
    vmax((x_lowmu_end-x0)/ds+1:end)=vlim;
end

v_forward=zeros(N,1);
v_backward=zeros(N,1);
a_forward=zeros(N,1);
a_backward=zeros(N,1);
v_forward(1)=v0;
a_forward(1)=a0;

for k=2:N
    dt=ds/v_forward(k-1);
    a_forward(k)=min(amax1,a_forward(k-1)+jmax*dt);
    v_forward(k)=min(vmax(k),v_forward(k-1)+a_forward(k-1)*dt);
    v_forward(k)=max(0,v_forward(k));
end


v_backward=v_forward;
a_backward(N)=0;


for k=1:N-1
    dt=ds/v_backward(N+1-k);
    a_backward(N-k)=min(amin1,a_backward(N+1-k)-jmin*dt);
    v_backward(N-k)=min(v_forward(N-k),v_backward(N+1-k)-a_backward(N+1-k)*dt);
    v_backward(N-k)=max(0, v_backward(N-k));
end

vf0=v_backward;


% s=0:ds:s_slide;
% figure(1);
% plot(s, v_forward,'LineWidth',2);
% hold on
% plot(s,vf0,'r-','LineWidth',2);
% hold on
% plot(s,vmax,'b-','LineWidth',1);
% hold on
% xlabel('s(m)');
% ylabel('v(m/s^2)');
% legend('speed profile','maximum velocity');


ds_1 = ds * ones(N-1, 1);
a_max=amax*ones(1,N); % 最大加速度m/s^2
a_min=amin*ones(1,N); % 最小加速度m/s^2
vmin=zeros(1,N);
v_squared_min=vmin.^2;
vf_squared=(vf0.^2)';
vf = vf0(1:N-1);
j_max=jmax*ones(1,N);
j_min=jmin*ones(1,N);

% 决策变量数量：b 和 a
num_vars = 2 * N; % b_1, ..., b_N 和 a_1, ..., a_N
% 目标函数
f = zeros(num_vars, 1);
f(1:N) = -1; % 对 b 的系数为 -1（最小化 -sum(b)）

% 上界和下界
lb=[v_squared_min,a_min];
ub=[vf_squared,a_max];

% 构建等式约束 b(i+1)-b(i)=2a(i)(s(i+1)-s(i))
Aeq = sparse(N-1, num_vars); % 使用稀疏矩阵
beq = zeros(N-1, 1);
row_idx = (1:N-1)'; % 行索引
col_bi = row_idx; % b_i 的列索引
col_bi1 = row_idx + 1; % b_{i+1} 的列索引
col_ai = N + row_idx; % a_i 的列索引

% 填充 Aeq 矩阵
Aeq(sub2ind(size(Aeq), row_idx, col_bi)) = -1; % -b_i
Aeq(sub2ind(size(Aeq), row_idx, col_bi1)) = 1; % +b_{i+1}
Aeq(sub2ind(size(Aeq), row_idx, col_ai)) = -2 * ds; % -2*a_i*(s_i+1-s_i)

% (17e) 的不等式约束
% j_min * delta_s <= vf_i * (a_{i+1} - a_i) <= j_max * delta_s
A = sparse(2*(N-1), num_vars); % (17e) 包含 2*(N-1) 个不等式
b = zeros(2*(N-1), 1);
row_idx1 = (1:N-1)'; % 第一部分约束的行索引
row_idx2 = (N:2*(N-1))'; % 第二部分约束的行索引

col_ai = N + row_idx1; % a_i 的列索引
col_ai1 = N + row_idx1 + 1; % a_{i+1} 的列索引

% 第一部分：j_min * delta_s <= (a_{i+1} - a_i) * v_f
A(sub2ind(size(A), row_idx1, col_ai)) = -vf ./ ds_1; % -v_f * a_i / delta_s
A(sub2ind(size(A), row_idx1, col_ai1)) = vf ./ ds_1; % +v_f * a_{i+1} / delta_s
b(row_idx1) = (j_max(1:end-1))' .* ds; % j_max * delta_s

% 第二部分： (a_{i+1} - a_i) * v_f <= j_max * delta_s
A(sub2ind(size(A), row_idx2, col_ai)) = vf ./ ds_1; % +v_f * a_i / delta_s
A(sub2ind(size(A), row_idx2, col_ai1)) = -vf ./ ds_1; % -v_f * a_{i+1} / delta_s
b(row_idx2) = -(j_min(1:end-1))' .* ds; % -j_min * delta_s

% 使用 linprog 求解
% options = optimoptions('linprog', 'Algorithm', 'interior-point', 'Display', 'none');
% [x_opt, fval, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options);

% 提取结果
% b_opt = x_opt(1:N); % 优化后的 b
% a_opt = x_opt(N+1:end); % 优化后的 a

% 准备MOSEK优化问题结构
prob = [];

% 目标函数
prob.c = f;

% 约束矩阵
% 将等式和不等式约束合并
prob.a = [Aeq; A];

% 约束上下界
% 对于等式约束，上下界相等
% 对于不等式约束 Ax <= b，下界为-inf
prob.blc = [beq; -inf(size(b))];  % 约束下界
prob.buc = [beq; b];              % 约束上界

% 变量上下界
prob.blx = lb';  % 变量下界
prob.bux = ub';  % 变量上界

% 设置MOSEK参数
param = [];
% 设置详细输出
param.MSK_IPAR_LOG = 0;
% 设置求解器精度
param.MSK_DPAR_INTPNT_TOL_INFEAS = 1e-12;
% 关闭基本识别
param.MSK_IPAR_INTPNT_BASIS = 'MSK_OFF';

% 调用mosekopt求解
[r,res] = mosekopt('minimize',prob,param);

x_opt = res.sol.itr.xx;
% 目标函数值
% fval = res.sol.itr.pobjval;

% 提取优化变量
b_opt = x_opt(1:N);
a_opt = x_opt(N+1:end);

v_s=sqrt(b_opt);
a_s=a_opt;

% if v_s(1) ~= vf0(1)
%     Eq_idx = find(v_s==vf0);
%     eq_idx = Eq_idx(1);
%     v_s(1:eq_idx)=vf0(1:eq_dix);
% end

% figure(2);
% plot(s',v_s,'b-','LineWidth',2);
% hold on
% plot(s',vf0,'k-','LineWidth',1);
% hold on
% xlabel('distance(m)');
% ylabel('speed profile(m/s)');
% legend('replanned velocity','planned velocity');

% figure(3)
% plot(s',a_opt,'k-','LineWidth',2);
% hold on
% xlabel('distance(m)');
% ylabel('acceleration profile(m/s^2)');
% legend('replanned acceleration');

v_t=vs_to_vt(v_s,s_slide,delta_t,ds);
v_t=v_t(1);
a_t=as_to_at(a_s,s_slide,v0,delta_t,ds);
a_t=a_t(1);

end


function speed = vs_to_vt(v_s, s_slide, delta_t, ds)
s = 0:ds:s_slide;

% 初始化时间和速度数组
t = 0;  % 初始时间
v_t = v_s(1);  % 初始速度对应的速度

% 初始化当前位置
s_current = s(1);  % 初始位置

% 记录时间-速度曲线
% time = [t];
speed = [v_t];

% 估算最大时间（防止无限循环）
max_time = (s_slide / min(abs(v_s(v_s ~= 0))) + 1) * delta_t; % 粗略估算

% 以时间步长推进，直到位移达到或超过 s_slide
while t < max_time && s_current < s(end)
    % 计算新的位移
    s_new = s_current + v_t * delta_t;  % 基于当前速度和时间步长计算新的位移
    
    % 更新时间
    t = t + delta_t;
    
    % 如果新位移超出范围，使用最后一个速度
    if s_new >= s(end)
        s_current = s(end);
        v_new = v_s(end);  % 使用最后一个速度值
    else
        % 查找新的位移对应的索引
        [~, idx] = min(abs(s - s_new));  % 查找离 s_new 最近的位移
        v_new = v_s(idx);  % 获取对应的速度
        
        % 如果新速度不为0，更新位置
        if v_new ~= 0
            s_current = s_new;
        end
        % 注意：如果 v_new == 0，s_current 不变
    end
    
    % 记录新的时间和速度
    % time = [time, t];
    speed = [speed, v_new];
    
    % 更新当前速度
    v_t = v_new;  % 速度更新为新的对应速度
end
% figure(4);
% plot(time, speed);
end

function acceleration = as_to_at(a_s,s_slide,v0, delta_t,ds)
s=0:ds:s_slide;

% 初始化时间、加速度和速度
t = 0;  % 初始时间
a_t = a_s(1);  % 初始加速度
v_t = v0;  % 初始速度

% 初始化当前位置
s_current = s(1);  % 初始位置

% 记录时间-加速度曲线
% time = [t];
acceleration = [a_t];

% 通过固定时间步长计算位移、速度和加速度
while s_current < s(end)  % 直到位移达到最大值
    % 更新速度
    v_t = v_t + a_t * delta_t;  % 基于加速度和时间步长更新速度

    % 计算新的位移
    s_new = s_current + v_t * delta_t;  % 基于当前速度和时间步长更新位移

    % 查找新的加速度对应的索引（位置）
    [~, idx] = min(abs(s - s_new));  % 查找离 s_new 最近的位移
    % a_new = a_s(idx);  % 获取对应的加速度

    % 检查是否遇到零值加速度
    if a_s(idx) == 0
        a_new = 0;  % 如果当前位置对应的加速度为0，则保持为0
    else
        a_new = a_s(idx);  % 否则使用对应的加速度值
    end

    % 更新当前时间
    t = t + delta_t;

    % 更新当前位置
    s_current = s_new;

    % 记录新的时间和加速度
    % time = [time, t];
    acceleration = [acceleration, a_new];

    % 更新加速度
    a_t = a_new;  % 加速度更新为新的对应加速度
end
end