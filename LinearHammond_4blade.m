function [f, fdx, fdxp] = LinearHammond_4blade(t, x, xp,Omega)
% Data
Nb = 4;
e  =    0.3048; % m
R  =     7.620; % m
Mb =      94.9; % Kg
Sb =     289.1; % Kg*m
Ib =    1084.7; % Kg*m^2
Cb =    4067.5; % N*m*s/rad
Kb =       0.0; % N*m/rad
nu2=   e*Sb/Ib;

Mx =    8026.6; % Kg
Cx =   51078.7; % N*s/m
Kx = 1240481.8; % N/m

My =    3283.6; % Kg
Cy =   25539.3; % N*s/m
Ky = 1240481.8; % N/m

v0=sqrt(e*Sb/Ib);
w0=sqrt(Kb/Ib);
nu=Cb/Ib;

psi1=Omega*t;
psi2=pi/2+Omega*t;
psi3=pi+Omega*t;
psi4=3/2*pi+Omega*t;

% Spring x
x1=x(1);
x2=x(2);
xp1=xp(1);
xp2=xp(2);
% Spring y
y1=x(3);
y2=x(4);
yp1=xp(3);
yp2=xp(4);
% Blade 1
z1=x(5);
z2=x(6);
zp1=xp(5);
zp2=xp(6);
% Blade 2
z21=x(7);
z22=x(8);
z2p1=xp(7);
z2p2=xp(8);
% Blade 3
z31=x(9);
z32=x(10);
z3p1=xp(9);
z3p2=xp(10);
% Blade 4
z41=x(11);
z42=x(12);
z4p1=xp(11);
z4p2=xp(12);
% stability
i=0;  % if i==1 stable, if i ==0 unstable

% e=e/rho <================= testare
% Noise : 1*normrnd(0,1)*sqrt(1e-2)

% f(x',x,t)=0
f=[xp1-x2;yp1-y2;zp1-z2;z2p1-z22;z3p1-z32;z4p1-z42;...
    (Mx+Nb*Mb)*xp2+Cx*x2+Kx*x1-Sb*((zp2-Omega^2*z1)*sin(psi1)+2*Omega*z2*cos(psi1)+...    % Equilibrium x direction
    (z2p2-Omega^2*z21)*sin(psi2)+2*Omega*z22*cos(psi2)+...
    (z3p2-Omega^2*z31)*sin(psi3)+2*Omega*z32*cos(psi3)+...
    (z4p2-Omega^2*z41)*sin(psi4)+2*Omega*z42*cos(psi4));
    (My+Nb*Mb)*yp2+Cy*y2+Ky*y1+Sb*((zp2-Omega^2*z1)*cos(psi1)-2*Omega*z2*sin(psi1)+...    % Equilibrium y direction
    (z2p2-Omega^2*z21)*cos(psi2)-2*Omega*z22*sin(psi2)+...
    (z3p2-Omega^2*z31)*cos(psi3)-2*Omega*z32*sin(psi3)+...
    (z4p2-Omega^2*z41)*cos(psi4)-2*Omega*z42*sin(psi4));    
    zp2+nu*z2+(w0^2+Omega^2*v0^2)*z1-(v0^2/e)*(xp2*sin(psi1)-yp2*cos(psi1));...           % Blade 1
    z2p2+nu*z22+(w0^2+Omega^2*v0^2)*z21-(v0^2/e)*(xp2*sin(psi2)-yp2*cos(psi2));...     % Blade 2
    z3p2+i*nu*z32+(w0^2+Omega^2*v0^2)*z31-(v0^2/e)*(xp2*sin(psi3)-yp2*cos(psi3));...     % Blade 3
    z4p2+nu*z42+(w0^2+Omega^2*v0^2)*z41-(v0^2/e)*(xp2*sin(psi4)-yp2*cos(psi4));];      % Blade 4
% Jacobian matrix of x
fdx = [0,-1,0,0,0,0,0,0,0,0,0,0;...
    0,0,0,-1,0,0,0,0,0,0,0,0;...
    0,0,0,0,0,-1,0,0,0,0,0,0;...
    0,0,0,0,0,0,0,-1,0,0,0,0;...
    0,0,0,0,0,0,0,0,0,-1,0,0;...
    0,0,0,0,0,0,0,0,0,0,0,-1;...
    Kx,Cx,0,0,Sb*Omega^2*sin(psi1),-Sb*2*Omega*cos(psi1),...
    Sb*Omega^2*sin(psi2),-Sb*2*Omega*cos(psi2),Sb*Omega^2*sin(psi3),-Sb*2*Omega*cos(psi3),...
    Sb*Omega^2*sin(psi4),-Sb*2*Omega*cos(psi4); ...
    0,0,Ky,Cy,-Sb*Omega^2*cos(psi1),-Sb*2*Omega*sin(psi1),...
    -Sb*Omega^2*cos(psi2),-Sb*2*Omega*sin(psi2),...
    -Sb*Omega^2*cos(psi3),-Sb*2*Omega*sin(psi3),...
    -Sb*Omega^2*cos(psi4),-Sb*2*Omega*sin(psi4);...
    0,0,0,0,(w0^2+Omega^2*v0^2),nu,0,0,0,0,0,0;...
    0,0,0,0,0,0,(w0^2+Omega^2*v0^2),nu,0,0,0,0;...
    0,0,0,0,0,0,0,0,(w0^2+Omega^2*v0^2),nu*i,0,0;...
    0,0,0,0,0,0,0,0,0,0,(w0^2+Omega^2*v0^2),nu;];
% Jacobian matrix of x'

fdxp = [1,0,0,0,0,0,0,0,0,0,0,0;...
    0,0,1,0,0,0,0,0,0,0,0,0;...
    0,0,0,0,1,0,0,0,0,0,0,0;...
    0,0,0,0,0,0,1,0,0,0,0,0;...
    0,0,0,0,0,0,0,0,1,0,0,0;...
    0,0,0,0,0,0,0,0,0,0,1,0;...
    0,(Mx+Nb*Mb),0,0,0,-Sb*sin(psi1),0,-Sb*sin(psi2),0,-Sb*sin(psi3),0,-Sb*sin(psi4);...
    0,0,0,(My+Nb*Mb),0,Sb*cos(psi1),0,Sb*cos(psi2),0,Sb*cos(psi3),0,Sb*cos(psi4);...
    0,-(v0^2/e)*sin(psi1),0,(v0^2/e)*cos(psi1),0,1,0,0,0,0,0,0;...
    0,-(v0^2/e)*sin(psi2),0,(v0^2/e)*cos(psi2),0,0,0,1,0,0,0,0;...
    0,-(v0^2/e)*sin(psi3),0,(v0^2/e)*cos(psi3),0,0,0,0,0,1,0,0;...
    0,-(v0^2/e)*sin(psi4),0,(v0^2/e)*cos(psi4),0,0,0,0,0,0,0,1;];


end

