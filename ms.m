% Copyright (C) 2010
% 
% Pierangelo Masarati	<masarati@aero.polimi.it>
% 
% Dipartimento di Ingegneria Aerospaziale - Politecnico di Milano
% via La Masa, 34 - 20156 Milano, Italy
% http://www.aero.polimi.it
% 
% Changing this copyright notice is forbidden.
% 
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation (version 2 of the License).
% 
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%
%function [out, outp] = ms(fun, t, x0, xp0, opt);
function [out, outp] = ms(vfun, vt, vinit, vinitp, opt);

% function [out, outp] = ms(fun, t, x0, xp0, opt);
%
% Solves initial value problems of the form `f(x, x', t) = 0' by calling
% function `vfun', using an implicit integration scheme derived
% from 2nd order BDF.  `f' can be DAE.
% 
%     fun: user-provided function that computes f, fdx, fdxp
%          as functions of t, x(t), x'(t)
%
%         [f, fdx, fdxp] = vfun(t, x, xp);
%
%     t: time vector; set t = [0:N]/N*T
%
%     x0: x(t(1)) [column vector, n x 1]
%
%     xp0: x'(t(1)) [column vector, n x 1]
%
%     opt: parameters
%
%         opt.Rho: spectral radius [mandatory]
%
%         opt.Tolerance: tolerance [mandatory]
%                 (convergence test is |f| <= opt.Tolerance)
%
%         opt.MaxIter: maximum iterations number [optional]
%                 ( < 0: silently converge instead of bailing out)
%

[nt0, nx] = size(vinit);
nt = length(vt);

out = zeros(nt, nx);
outp = zeros(nt, nx);

out(1:nt0, :) = vinit;
outp(1:nt0, :) = vinitp;

Rho = opt.Rho;
Tol = opt.Tolerance;
eval('MaxIter = opt.MaxIter;', 'MaxIter = 0;');

%disp(sprintf('MaxIter=%d', MaxIter));
%fflush(stdout);

for i = 1+nt0:nt,
	% predict
%    i
	h = vt(i) - vt(i - 1);
	xm1 = out(i - 1, :);
	xpm1 = outp(i - 1, :);
	if (i == 2),
		xp = xpm1;
		x = xm1 + xpm1*h;

		b0 = h/2;
	else
		xm2 = out(i - 2, :);
		xpm2 = outp(i - 2, :);

		% current step/previous step
		Alpha = h/(vt(i - 1) - vt(i - 2));

	        mp0 = -6.*Alpha*(1. + Alpha)/h;
		mp1 = -mp0;
		np0 = 1. + 4.*Alpha + 3.*Alpha*Alpha;
		np1 = Alpha*(2. + 3.*Alpha);

		xp = mp0*xm1 + mp1*xm2 + np0*xpm1 + np1*xpm2;

		Den = 2.*(1. + Alpha) - (1. - Rho)*(1. - Rho);
		Beta = Alpha*((1. - Rho)*(1. - Rho)*(2. + Alpha) ...
			+ 2.*(2.*Rho - 1.)*(1. + Alpha))/Den;
		Delta = .5*Alpha*Alpha*(1. - Rho)*(1. - Rho)/Den;

		a1 = 1. - Beta;
		a2 = Beta;
		b0 = h*(Delta/Alpha + Alpha/2);
		b1 = h*(Beta/2. + Alpha/2. - Delta/Alpha*(1. + Alpha));
		b2 = h*(Beta/2. + Delta);

		x = a1*xm1 + a2*xm2 + b0*xp + b1*xpm1 + b2*xpm2;
	end

	% correct
	Err = 1.e+18;
	Iter = 0;
	while (1),
		[f, fdx, fdxp] = feval(vfun, vt(i), x', xp');
		Err = norm(f);
		if (Err < Tol),
			break;
		end

		Iter = Iter + 1;
		if (MaxIter && Iter >= abs(MaxIter)),
			disp(sprintf('MaxIter=%d exceeded at time=%e, Err=%e', MaxIter, vt(i), Err));
			if (MaxIter > 0),
				return;
			end
			break;
		end

		J = b0*fdx + fdxp;
		dxp = -J\f;
		xp = xp + dxp';
		x = x + b0*dxp';
	end

	out(i, :) = x;
	outp(i, :) = xp;
end

