% *********************************************************** %
% Ground Resonance with one damper inoperative: Hammond Case  %
% *********************************************************** %

% *************************************
% Author: Vincenzo Muscarello
% E-MAIL: vincenzo.muscarello@polimi.it
% Date:   25 - 04 - 2019
% Politecnico di Milano
% *************************************

function [AA] = Hammond(Omega)
%% DISCRETIZATION
N      = 72*3;                      % Number of stations for discretization;
T      = 2*pi/Omega;                % Final Time - One Period;
dpsi   = 2*pi/N;                    % Angular step;                 
dt     = dpsi/Omega;                % Time step;

t      = [0 : dt : T];
psi    = [0 : dpsi : 2*pi];

%% HAMMOND DATA
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

%% MATRICES (SECOND ORDER)
% Mass 
D2 = [ Nb*Ib ,     0   ,     0   ,   0   ,     0      ,      0       ;           
         0   , Nb/2*Ib ,     0   ,   0   ,     0      , -Nb/2*Sb     ;
         0   ,     0   , Nb/2*Ib ,   0   , Nb/2*Sb    ,      0       ;
         0   ,     0   ,     0   , Nb*Ib ,     0      ,      0       ; 
         0   ,     0   , Nb/2*Sb ,   0   ,  Mx + Nb*Mb,      0       ;
         0   ,-Nb/2*Sb ,     0   ,   0   ,     0      ,   My + Nb*Mb];
% Damping
D1 = zeros(6,6);
D1(5,5) = Cx;
D1(6,6) = Cy;
D1(2,3) = Nb/2 * 2 * Omega * Ib;
D1(3,2) =-Nb/2 * 2 * Omega * Ib;
% Stiffness
D0 = zeros(6,6);
D0(5,5) = Kx;
D0(6,6) = Ky;
D0(1,1) = Nb   * Omega^2 *  nu2      * Ib;
D0(2,2) = Nb/2 * Omega^2 * (nu2 - 1) * Ib;
D0(3,3) = Nb/2 * Omega^2 * (nu2 - 1) * Ib;
D0(4,4) = Nb   * Omega^2 *  nu2      * Ib;
% Input
Inp = [Nb*eye(4,4) ; zeros(2,4)];
Inp(2,2) = Inp(2,2)/2;
Inp(3,3) = Inp(3,3)/2;
Out = [   eye(4,4) ; zeros(2,4)];
 
%% MATRICES (FIRST ORDER)
A = [zeros(6,6), eye(6,6)   ;
     -D2\D0    , -D2\D1    ];
B = [zeros(6,4);  D2\Inp   ];
C = [ Out'     , zeros(4,6) ;
     zeros(4,6),  Out'     ];

%% DAMPER
% Nominal dampers
CR = Cb*eye(4,4);
KR = Kb*eye(4,4);
% One damper Inoperative
CR(1,1) = 0;
KR(1,1) = 0;

% Interblade (4 blade)
% CR = diag(2*C*ones(4,1),0) + diag(-C*ones(3,1),-1) + diag(-C*ones(3,1),1) + diag(-C,-3)  + diag(-C,+3);
% KR = diag(2*K*ones(4,1),0) + diag(-K*ones(3,1),-1) + diag(-K*ones(3,1),1) + diag(-K,-3)  + diag(-K,+3);

%% TRANSFORMATION MATRIX
Psi1 = Omega*t + 1*2*pi/4;
Psi2 = Omega*t + 2*2*pi/4;
Psi3 = Omega*t + 3*2*pi/4;
Psi4 = Omega*t + 4*2*pi/4;
T    = zeros( 4, 4,N+1);
Td   = zeros( 4, 4,N+1);
AA   = zeros(12,12,N+1);
G    = zeros( 4, 8,N+1);

for k = 1 : N+1
    T(:,:,k)  = [ 1 , cos(Psi1(k)) , sin(Psi1(k)) , -1 ;
                  1 , cos(Psi2(k)) , sin(Psi2(k)) , +1 ;
                  1 , cos(Psi3(k)) , sin(Psi3(k)) , -1 ;
                  1 , cos(Psi4(k)) , sin(Psi4(k)) , +1];
    
    Td(:,:,k) = [ 0 , -Omega*sin(Psi1(k)) , +Omega*cos(Psi1(k)) , 0 ;
                  0 , -Omega*sin(Psi2(k)) , +Omega*cos(Psi2(k)) , 0 ;
                  0 , -Omega*sin(Psi3(k)) , +Omega*cos(Psi3(k)) , 0 ;
                  0 , -Omega*sin(Psi4(k)) , +Omega*cos(Psi4(k)) , 0];

    G(:,1:4,k)= T(:,:,k)\(KR*T(:,:,k) + CR*Td(:,:,k)); 
    G(:,5:8,k)= T(:,:,k)\CR*T(:,:,k);
   AA(:, : ,k)= A - B*G(:,:,k)*C;
end