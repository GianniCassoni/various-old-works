%%  Ground resonance Hammond stochastic
close all  
clc 
clear

addpath("C:\Users\Gianni\Desktop\Matlab\magistrale2\TesiFinale")
savepath
%% ---Data---
data_problem= 1; % <================================ chosing the data
switch data_problem
        case 1 % strong non linearity
            Omega0=26.1799; 
        case 2 %weak non linearity
            Omega0= 13.0106;
        case 3
            Omega0= 22.0017;
    otherwise
end

time_dis=2;
% N = 2000; % number of steps
 N = 10000; % number of steps
deltaT = 1e-2; % time step 

switch time_dis
        case 1 % strong non linearity
            dpsi= 2*pi/Omega0;
            t=0:deltaT*dpsi:dpsi*N*deltaT; %meglio per la stabilità numerica
        case 2 %weak non linearity
            t = [0:N]*deltaT; % time 
    otherwise
end
% [x_position,x_velocity,liglag_blade1,liglag_velocity_blade1,.....,liglag_velocity_blade4]
x0=[0,0,0,0,0,0,0,0,0,0,0,0]; % initial condition
%Omega= linspace(200*2*pi/60,300*2*pi/60,100); % Omega
Omega= linspace(10*2*pi/60,400*2*pi/60,100); % Omega
opt.Tolerance = 1e-4;
opt.Rho = .7;
%% ---- Numerical method parameters ----
opt.Tolerance = 1e-4;
opt.Rho = .7;
%%
g=[0,0,0,0,0,0,1,1,1,1,1,1;];
ep=[100:100:10100];
 %%
ll_max=zeros(length(ep),length(Omega));
T=10;
for j=1:length(ep)
    for i=1:length(Omega)
    %     [x_f, xp_f] = ms(@(t, x, xp) LinearHammond_4blade(t, x, xp,Omega(i)), t, x0 ,zeros(1,12), opt);
        [x_f, xp_f] = ms_s(@(t, x, xp) Hammond_4blade(t, x, xp,Omega(i)), t, x0 ,zeros(1,12),ep(j),g, opt);
        dim=2;
        [S, Aout] = pod(x_f(1:T:end,:),dim,deltaT*T);
        xdata =Aout;
        fs=1/(deltaT*T);
        [~,lag] = phaseSpaceReconstruction(xdata,[],dim);
        eRange = 2000/T;
        %2000
        [Lmax]=lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',eRange);
        ll_max(j,i)=Lmax;
    end
    X = sprintf('%d ly: done',j);
    disp(X)
end

% legend('Maximum amplitude $\xi_3$','interpreter','latex')
%%
% tiledlayout(1,1, 'TileSpacing', 'none', 'Padding', 'none');
% nexttile

% imagesc(C,Omega.*60/2/pi,Amax.*180/pi)
imagesc(ep,Omega.*60/2/pi,ll_max')
c = colorbar;
c.Label.String = '$\lambda_{max}$';
c.Label.Interpreter = 'latex';
colormap jet;
xlabel('$\beta$','interpreter','latex');
ylabel(' $\Omega$[rpm]','interpreter','latex');
%%
tiledlayout(1,1, 'TileSpacing', 'none', 'Padding', 'none');
nexttile
surf(ep,Omega(1:80).*60/2/pi,ll_max(:,1:80)')
colormap jet;
xlabel('$\beta$','interpreter','latex');
ylabel(' $\Omega$[rpm]','interpreter','latex');
zlabel(' $\lambda_{max}$[1/s]','interpreter','latex');