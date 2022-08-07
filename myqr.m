function [Q, R] = myqr(A);

[Q, R] = qr(A);

[nr, nc] = size(A);
if nr > nc,
	nn = nc;
else
	nn = nr;
end

Q = Q(:, 1:nc);
R = R(1:nc, :);

for ii = 1:nn,
	if (R(ii, ii) < 0),
		Q(:, ii) = -Q(:, ii);
		R(ii, ii:end) = -R(ii, ii:end);
	end
end

