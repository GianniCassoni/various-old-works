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
N = 50000; % number of steps
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
x0=[0.001,0,0,0,0,0,0,0,0,0,0,0]; % initial condition
%Omega= linspace(200*2*pi/60,300*2*pi/60,100); % Omega
Omega= linspace(10*2*pi/60,400*2*pi/60,100); % Omega
opt.Tolerance = 1e-4;
opt.Rho = .7;
ti=2000; % plot start
tf=N; % plot end
lim_on=1; % <============= same scale 1 : yes , else : no
%% ---------- LCE Non Linear ----------

% if time_dis==1
%     error('time discretization depend on Omega so change for each LCE')
% end

n=12; % order of the system 
ll_f=zeros(n,length(Omega));
for i = 1 : length(Omega)
%     switch time_dis
%             case 1 % strong non linearity
%                 dpsi= 2*pi/Omega(i);
%                 t=0:deltaT*dpsi:dpsi*N*deltaT; %meglio per la stabilità numerica
%             case 2 %weak non linearity
%                 t = [0:N]*deltaT; % time 
%         otherwise
%     end
    [x_f, xp_f] = ms(@(t, x, xp) LinearHammond_4blade(t, x, xp,Omega(i)), t, x0 ,zeros(1,12), opt);
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
        [f, fdx, fdxp] = LinearHammond_4blade(t(j), x_f(j,:)', xp_f(j,:)',Omega(i));
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
      Disp2 = sprintf('%d ly: done',i);
        disp(Disp2)
    ll_f(:,i)=ll(end,:);
end

%%
tiledlayout(1,1, 'TileSpacing', 'none', 'Padding', 'none');
nexttile
b1=plot(Omega.*60/2/pi,ll_f,'*k');
grid on
xlabel('$\Omega$[rpm]','interpreter','latex');
ylabel(' Maximum Exponent','interpreter','latex');
%%
for i = 1 : length(Omega)
             [A] = Hammond(Omega(i));
    [Eigen(:,i),Ev] = floquet(A,Omega(i));
end
%%
hold on 
b2=plot(Omega.*60/2/pi,real(Eigen(:,:)),'or');
plot(Omega.*60/2/pi,zeros(length(Omega),1),'k--')
legend([b1 b2],'$Maximum$ $LCE$','$Floquet$','interpreter','latex')