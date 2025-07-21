function [u, v] = affine(u, v, x, y)
% This function calculates the affined optical flow using feature points 
% 
% input:
% --[u v]:  optical flow
% --p:      feature points: m x n. m = 2, n = 66.
%
% output:   affined optical flow

% size of image
[m ,n]      = size(u);             

%A*H = B 
M = 13; N = 3;                                                          %dimension of affine matrix
A = zeros(M, N); B = zeros(M, N); H = zeros(N, N);                      %initilization of affine matrix
index = [1, 2, 3, 4, 5, 6,  28,  12, 13, 14, 15, 16, 17];    %point set
for i = 1:M
    px = x(index(i)); py = y(index(i));
    B(i, 1) = px; B(i, 2) = py; B(i, 3) = 1.0;
    A(i, 1) = px + u(py, px); A(i, 2) = py + v(py, px); A(i, 3) = 1.0;
end

H = A\B;
%H=[1.00047,0.00111525,6.12078e-010;-0.000288531,1.0008,1.06545e-009;0.0981041,-0.3022,0.999999];

for i = 1:m
   for j = 1:n
       x = u(i, j) + j;
       y = v(i, j) + i;
       
       u(i, j) = x * H(1,1) + y * H(2,1) + 1.0 * H(3,1) - j;
       v(i, j) = x * H(1,2) + y * H(2,2) + 1.0 * H(3,2) - i;
   end
end

end

