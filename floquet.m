%FLOQUET - LYAPUNOV STABILITY FOR PERIODIC SYSTEMS         
% [Eigen] = floquet (A,Omega)
% 
% Solve the system:
% 
% dQ/dt = A(t)*Q
% Q(0)  = I
% 
% A(t)  is the State Space Periodic Matrix A(t) = A(t + T) 
% Q     is the Transition Matrix
% Omega are the RPM in rad/sec 
%
% Author: Vincenzo Muscarello
% E-MAIL: muscarello@aero.polimi.it
% Date:   17 - 11 - 2009
% Politecnico di Milano

function [Eigen,Ev] = floquet (A,Omega)

N      = size(A,3) - 1;             % Number of azimuth steps during one revolution;
M      = size(A,1);                 % Matrix size;
T      = 2*pi/Omega;                % Final Time - One revolution;
dpsi   = 2*pi/N;                    % Azimuth step;                 
dt     = dpsi/Omega;                % Time step;
h      = 2*dt;                      % Time step for Runge - Kutta; 

% Initial Condition
Q(:,:,1) = eye(M);

% Numerical Integration method for periodic coefficient analysis
% Method 1 - Trapezoidal Method - Crank Nicholson;
% Method 2 - Runge Kutta fourth order; 

method = 2;

if method == 1
    % Crank - Nicholson
    for k = 1 : N
        Q(:,:,k + 1) = (eye(M) - dt/2*A(:,:,k + 1))\(eye(M) + dt/2*A(:,:,k))*Q(:,:,k);
    end
else    
    % Runge - Kutta fourth order
    j = 1;
    for k = 1 : N/2
         K1(:,:,k) = A(:,:,j)*Q(:,:,k);
         K2(:,:,k) = A(:,:,j + 1)*(Q(:,:,k) + h/2*K1(:,:,k));
         K3(:,:,k) = A(:,:,j + 1)*(Q(:,:,k) + h/2*K2(:,:,k));
         K4(:,:,k) = A(:,:,j + 2)*(Q(:,:,k) + h*K3(:,:,k));
         Q(:,:,k + 1) = Q(:,:,k) + h/6*(K1(:,:,k) + 2*K2(:,:,k) + 2*K3(:,:,k) + K4(:,:,k));
         j = j + 2;
    end
end

Eigen_k = eig(Q(:,:,end));
for k=1:N/2
    Ev(:,k) = eig(Q(:,:,k));
end
Eigen = 1/T*log(abs(Eigen_k)) + 1i*1/T*atan2(imag(Eigen_k),real(Eigen_k)); 
% Eigen = eig(1/T*logm(Q(:,:,end)));



