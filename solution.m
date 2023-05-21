clear;
clc;

% Problem 1
B = [1.8, 2.8, 0.3; randsample(0.1:0.1:0.9,1), -0.9,0.0;-0.1,1.0,2.3];
A = [5,-4,2;1,5,4;-6,-1,-3];
C = [3,6;4,-5;5,0];

% 1. B+A

ans1 = B+A;

% 2. AB
ans2 = A*B;

% 3. BA
ans3 = B*A;

% 4. 4+C
ans4 = 4 + C;

% 5. A+C
%ans5 = A + C;

% 6. CA
%ans6 = C*A

% 7. AC
ans7 = A*C;

% 8. 5(A+B)

ans8 = 5*(A+B);

% 9. A+B

ans9 = A+B;

% 10. 5A + 5B

ans10 = 5*A + 5*B;


%% Problem2

clear;
clc;

A = [-2,1;-4,2];
B = [-1,-2;-3,-6];
C = [-2,-6;1,3];

% 1. A(B+C) = AB + AC
LHS1 = A*(B+C);
RHS1 = A*B + A*C;
ans1 = LHS1==RHS1

% 2. A(B+C) = BA + CA
LHS2 = A*(B+C);
RHS2 = B*A + C*A;
ans2 = LHS2 == RHS2

% 3. (A+B)^2 = A^2 + 2AB + B^2

LHS3 = (A+B)^2;
RHS3 = A^2 + 2*A*B + B^2;
ans3 = LHS3==RHS3;


% 4. (AB)^2 = A^2B^2

LHS4 = (A*B)^2;
RHS4 = A^2*B^2;
ans4 = LHS4==RHS4;

% 5. (A-B)(A+B) = A^2 - B^2
LHS5 = (A-B)*(A+B);
RHS5 = A^2 - B^2;
ans5 = LHS5==RHS5;

% 6.
ans6 = A^2;
ans7 = B*C;

%% Problem 3
clear;
clc;

B = [-3 1 ; 1 2];
C = [3 0 -1; 1 5 -1];
A = [randsample(-7:7,1) -3;3 -6];

% 1. B'A'

ans1 = B.' * A.';

% 2. A'B'
ans2 = A.' * B.';

% 3. B'

ans3 = B.';

% 4. A' '

ans4 = (A.').';

% 5. C'A

ans5 = C.'*A;

% 6. (AB)'

ans6 = (A*B).';

% 7. AC'


ans7 = A*C.';


%% Problem4
clear;
clc;

R = 10*rand(3), S=10*rand(3)

% problem 1
s1 = S(:, 1);
s2 = S(:, 2);
s3 = S(:, 3);
c1 = R*s1;
c2 = R*s2;
c3 = R*s3;
C = [c1 c2 c3]
% problem 2
r1 = R(1, :);
r2 = R(2, :);
r3 = R(3, :);
m1 = r1*S;
m2 = r2*S;
m3 = r3*S;

M = [m1 ; m2 ; m3]

Q = R*S


%% Problem 5
clear;
clc;

a=randsample(2:9,1);
b=randsample(-10:-2,1);
c = b+1;

% Eye
M = a*eye(3)

% triu
N = triu(b*ones(3))

%diag
P = diag([2 3 4])

% ones 
Q = ones(3,2)*c


%% Problem 6
clear;
clc;

B = [-3 1; randsample([-9:-2, 2:9],1) 2];
A = [-3 -3 ; 3 -6];
C = [3 0 -1 ; 1 5 -1];

G = vertcat(horzcat(eye(2), C, A), horzcat(B, zeros(size(C)), eye(2)))

%% Problem 7
clear;
clc;

G = 10*rand(4,7);

H = G(1:3, 3:5);
E = H;
E(1, 3)=100

F = H
F(:,2)=[]

%% Problem 8
clear;
clc;
format rat

A = [5,3,-9;-10,-7,0;15,8,14];
A(1, :) = A(1, :) / A(1, 1)


A_new1 = A;

A_new1(2, :) = A(2, :) - A(2,1) * A(1,:)

