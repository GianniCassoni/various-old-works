%%  Ground resonance Inter2Blade
close all  
clc 
clear

addpath("C:\Users\Gianni\Desktop\Matlab\magistrale2\TesiFinale")
savepath
%% ---------- Data ----------
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
N = 25000; % number of steps
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
x0=[0.1,0,0,0,0,0,0,0,0,0,0,0]; % initial condition
%Omega= linspace(200*2*pi/60,300*2*pi/60,100); % Omega
Omega= linspace(10*2*pi/60,400*2*pi/60,100); % Omega
opt.Tolerance = 1e-4;
opt.Rho = .7;
ti=2000; % plot start
tf=N; % plot end
lim_on=1; % <============= same scale 1 : yes , else : no
%% ---------- LCE Non Linear ----------

if time_dis==1
    error('time discretization depend on Omega so change for each LCE')
end

n=12; % order of the system 
ll_f=zeros(n,length(Omega));
for i = 1 : length(Omega)
    [x_f, xp_f] = ms(@(t, x, xp) Hammond_4blade_ib(t, x, xp,Omega(i)), t, x0 ,zeros(1,12), opt);
    Y0 = eye(n);
    ll = zeros(length(t), n);
    ll_curr = zeros(1, n);
    [Qk, Rk] = myqr(Y0); % defintion of the QR method 
    Qt(:,:,1)=Qk;
    Numerical_method= 2; % chosing the numerical method for solving ... 
    switch Numerical_method
        case 1
            use_BDF = 0; 
        case 2
            use_BDF = 1; % numerical method (Backward difference finite)
        otherwise
    end
    Yk = Y0; % defintion of the inital axuliary problem
    R = Rk;
    for j = 2:length(t)
        [f, fdx, fdxp] = Hammond_4blade_ib(t(j), x_f(j,:)', xp_f(j,:)',Omega(i));
        h = t(j) - t(j - 1);
        if (j == 2 || use_BDF == 0)
            % Crank-Nicolson
            F = (fdxp + (h/2)*fdx)\(fdxp - (h/2)*fdx);
            Qkm1 = Qk;
            Qk = F*Qk;
            Ykm1 = Yk;
            Yk = F*Yk;
        else
            % BDF
            E = fdxp;
            A = -fdx;
            F = eye(n) + (E - (h*2/3)*A)\((h*2/3)*A);
		    tmp = F*(4/3*Qk - 1/3*((Rk')\(Qkm1'))');
            Qkm1 = Qk;
            Qk = tmp;
            tmp = F*(4/3*Yk - 1/3*Ykm1);
            Ykm1 = Yk;
            Yk = tmp;
        end
        [Qk, Rk] = myqr(Qk);
        ll_curr = ll_curr + log(diag(Rk)');
        ll(j, :) = ll_curr/t(j);
    end
    disp('ly: done'); 
    ll_f(:,i)=ll(end,:);
end
%%
tiledlayout(1,1, 'TileSpacing', 'none', 'Padding', 'none');
nexttile
b1=plot(Omega.*60/2/pi,ll_f,'*k');
grid on
xlabel('$\Omega$[rpm]','interpreter','latex');
ylabel(' LCE [1/s]','interpreter','latex');
hold on 
plot(Omega.*60/2/pi,zeros(length(Omega),1),'k--');
% legend(b1,'LCE ad-hoc','interpreter','latex')
%%
C=linspace(100,(4067.5)/4,100); % N*m*s/rad
n=12; % order of the system 
ll_max=zeros(length(C),length(Omega),n);
for k=1:length(C)
    ll_f=zeros(n,length(Omega));
    for i = 1 : length(Omega)
        [x_f, xp_f] = ms(@(t, x, xp) Hammond_4blade_i2b_Cvar(t, x, xp,Omega(i),C(k)), t, x0 ,zeros(1,12), opt);
        Y0 = eye(n);
        ll = zeros(length(t), n);
        ll_curr = zeros(1, n);
        [Qk, Rk] = myqr(Y0); % defintion of the QR method 
        Numerical_method= 2; % chosing the numerical method for solving ... 
        switch Numerical_method
            case 1
                use_BDF = 0; 
            case 2
                use_BDF = 1; % numerical method (Backward difference finite)
            otherwise
        end
        Yk = Y0; % defintion of the inital axuliary problem
        R = Rk;
        for j = 2:length(t)
            [f, fdx, fdxp] = Hammond_4blade_ib(t(j), x_f(j,:)', xp_f(j,:)',Omega(i));
            h = t(j) - t(j - 1);
            if (j == 2 || use_BDF == 0)
                % Crank-Nicolson
                F = (fdxp + (h/2)*fdx)\(fdxp - (h/2)*fdx);
                Qkm1 = Qk;
                Qk = F*Qk;
                Ykm1 = Yk;
                Yk = F*Yk;
            else
                % BDF
                E = fdxp;
                A = -fdx;
                F = eye(n) + (E - (h*2/3)*A)\((h*2/3)*A);
                tmp = F*(4/3*Qk - 1/3*((Rk')\(Qkm1'))');
                Qkm1 = Qk;
                Qk = tmp;
                tmp = F*(4/3*Yk - 1/3*Ykm1);
                Ykm1 = Yk;
                Yk = tmp;
            end
            [Qk, Rk] = myqr(Qk);
            ll_curr = ll_curr + log(diag(Rk)');
            ll(j, :) = ll_curr/t(j);
        end
        ll_f(:,i)=ll(end,:);
        ll_max(k,i,:)=ll_f(:,i);
        Disp2 = sprintf('%d ly: done',i);
        disp(Disp2)
    end
    Displ = sprintf('%d ----------ly: done',k);
    disp(Displ)
end
%%
% tiledlayout(1,1, 'TileSpacing', 'none', 'Padding', 'none');
% nexttile
imagesc(C,Omega.*60/2/pi,ll_max(:,:,4)')
c = colorbar;
c.Label.String = '$\lambda_{5}$';
c.Label.Interpreter = 'latex';
colormap jet;
xlabel('$c_i$ [$m \times N \times s \times rad^{-1}$]','interpreter','latex');
ylabel(' $\Omega$[rpm]','interpreter','latex');


%%
imagesc(C,Omega.*60/2/pi,ll_max')
c = colorbar;
c.Label.String = '$\lambda_{max}$';
c.Label.Interpreter = 'latex';
colormap jet;
xlabel('$\beta$','interpreter','latex');
ylabel(' $\Omega$[rpm]','interpreter','latex');
%% ---------- Time Plot ----------
[x_f, xp_f] = ms(@(t, x, xp) Hammond_4blade_i2b(t, x, xp,Omega0), t, x0 ,zeros(1,12), opt); % risolution of the non linear system 
disp('ms: done');


Colors_v={'[0.8500 0.3250 0.0980]'  '[0, 0.4470, 0.7410]' '[0.9290, 0.6940, 0.1250]' '[0.4660, 0.6740, 0.1880]' ...
    '[0.3010, 0.7450, 0.9330]' '[0.4940, 0.1840, 0.5560]'};
labels_mb={'$x [m]$' '$y[m]$' '$\xi_1 [rad]$' '$\xi_{2} [rad]$' '$\xi_{3} [rad]$' '$\xi_{4} [rad]$' };
labels_dmb={'$\dot{x} [m/s]$' '$\dot{y} [m/s]$' '$\dot{\xi}_1 [rad/s]$' '$\dot{\xi}_{2} [rad/s]$'...
    '$\dot{\xi}_{3} [rad/s]$' '$\dot{\xi}_{4} [rad/s]$' };

figure(1)
tiledlayout(3,2, 'TileSpacing', 'none', 'Padding', 'none');
m=0;
for i=1:2:12
    m=m+1;
    nexttile
    plot(t(ti:tf),x_f(ti:tf,i),'Color',Colors_v{m});
    grid on
    xlabel('$t[s]$','interpreter','latex');
    ylabel(labels_mb{m},'interpreter','latex');
end

figure(2)
tiledlayout(3,2, 'TileSpacing', 'none', 'Padding', 'none');
m=0;
for i=2:2:12
    m=m+1;
    nexttile
    plot(t(ti:tf),x_f(ti:tf,i),'Color',Colors_v{m});
    grid on
    xlabel('$t[s]$','interpreter','latex');
    ylabel(labels_dmb{m},'interpreter','latex');
end

%% ---------- Phase Plot ----------
a=max(max(abs(x_f(:,4))),max(abs(x_f(:,2))));
yylims=-a-a*0.05;
yylimd=a+a*0.05;
a=max(max(abs(x_f(:,1))),max(abs(x_f(:,3))));
xxlims=-a-a*0.05;
xxlimd=a+a*0.05;
a=max([max(abs(x_f(:,6))),max(abs(x_f(:,8))),max(abs(x_f(:,10))),max(abs(x_f(:,12)))]);
xiylims=-a-a*0.05;
xiylimd=a+a*0.05;
a=max([max(abs(x_f(:,5))),max(abs(x_f(:,7))),max(abs(x_f(:,9))),max(abs(x_f(:,11)))]);
xixlims=-a-a*0.05;
xixlimd=a+a*0.05;

figure(3)
tiledlayout(3,2, 'TileSpacing', 'none', 'Padding', 'none');
m=0;
for i=0:2:10
    m=m+1;
    nexttile
    plot(x_f(ti:tf,i+1),x_f(ti:tf,i+2),'Color',Colors_v{m});
    grid on
    if lim_on==1
        if i<4
            ylim([yylims yylimd])
            xlim([xxlims xxlimd])
        else
            ylim([xiylims  xiylimd])
            xlim([xixlims xixlimd])
        end
    end
    xlabel(labels_mb{m},'interpreter','latex');
    ylabel(labels_dmb{m},'interpreter','latex');
end

%% ---------- Poincaré map - Periodic ----------
figure(4)
switch time_dis
        case 1 % strong non linearity
            T=1/deltaT;
        case 2 %weak non linearity
            T=2*pi/Omega0/deltaT;
            T=ceil(T); 
    otherwise
end
tiledlayout(3,2, 'TileSpacing', 'none', 'Padding', 'none');
m=0;
for i=0:2:10
    m=m+1;
    nexttile
    plot(x_f(ti:tf,i+1),x_f(ti:tf,i+2),'Color',Colors_v{m});
    hold on
    for j=ti:T:tf
        plot(x_f(j,i+1),x_f(j,i+2),'ok')
    end
    if lim_on==1
        if i<4
            ylim([yylims yylimd])
            xlim([xxlims xxlimd])
        else
            ylim([xiylims  xiylimd])
            xlim([xixlims xixlimd])
        end
    end
    grid on
    xlabel(labels_mb{m},'interpreter','latex');
    ylabel(labels_dmb{m},'interpreter','latex');
end

