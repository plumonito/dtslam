% [s R T error] = absoluteOrientationQuaternion( A, B, doScale)
%
% Computes the orientation and position (and optionally the uniform scale 
% factor) for the transformation between two corresponding 3D point sets Ai 
% and Bi such as they are related by:
%
%     Bi = sR*Ai+T
%
% Implementation is based on the paper by Berthold K.P. Horn:
% "Closed-from solution of absolute orientation using unit quaternions" 
% The paper can be downloaded here:
% http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
%
% Authors:      Dr. Christian Wengert, Dr. Gerald Bianchi
% Copyright:    ETH Zurich, Computer Vision Laboratory, Switzerland
%
% Parameters:   A           3xN matrix representing the N 3D points
%               B           3xN matrix representing the N 3D points
%               doScale     Flag indicating whether to estimate the 
%                           uniform scale factor as well [default=0]
%
% Return:       s           The scale factor
%               R           The 3x3 rotation matrix
%               T           The 3x1 translation vector
%               err         Residual error    
%
% Notes: Minimum 3D point number is N > 4
function [s R T err] = absoluteOrientationQuaternion( A, B, doScale)


%Argument check
if(nargin<3)
    doScale=1;
end
%Return argument check
if(nargout<1)
    usage()
    error('Specify at least 1 return arguments.');
end
%Test size of point sets
[c1 r1] = size(A);
[c2 r2] = size(B);
if(r1~=r2)
    usage()
    error('Point sets need to have same size.');
end
if(c1~=3 | c2~=3)
    usage()
    error('Need points of dimension 3');
end
if(r1<4)
    usage()
    error('Need at least 4 point pairs');
end

%Number of points
Na = r1;    
    
%Compute the centroid of each point set
Ca = mean(A,2);
Cb = mean(B,2);

%Remove the centroid
An = A - repmat(Ca,1,Na);
Bn = B - repmat(Cb,1,Na);

%Compute the quaternions
M = zeros(4,4);
for i=1:Na 
    %Shortcuts
    a = [0;An(:,i)];
    b = [0;Bn(:,i)];    
    %Crossproducts
    Ma = [  a(1) -a(2) -a(3) -a(4) ; 
            a(2)  a(1)  a(4) -a(3) ; 
            a(3) -a(4)  a(1)  a(2) ; 
            a(4)  a(3) -a(2)  a(1)  ];
    Mb = [  b(1) -b(2) -b(3) -b(4) ; 
            b(2)  b(1) -b(4)  b(3) ; 
            b(3)  b(4)  b(1) -b(2) ; 
            b(4) -b(3)  b(2)  b(1)  ];
    %Add up
    M = M + Ma'*Mb;
end

%Compute eigenvalues
[E D] = eig(M);

%Compute the rotation matrix
e = E(:,4);
M1 = [  e(1) -e(2) -e(3) -e(4) ; 
        e(2)  e(1)  e(4) -e(3) ; 
        e(3) -e(4)  e(1)  e(2) ; 
        e(4)  e(3) -e(2)  e(1)  ];
M2 = [  e(1) -e(2) -e(3) -e(4) ; 
        e(2)  e(1) -e(4)  e(3) ; 
        e(3)  e(4)  e(1) -e(2) ; 
        e(4) -e(3)  e(2)  e(1)  ];


R = M1'*M2;

%Retrieve the 3x3 rotation matrix
R = R(2:4,2:4);
%Compute the scale factor if necessary
if(doScale)    
    a =0; b=0;
    for i=1:Na
        a = a + Bn(:,i)'*R*An(:,i);
        b = b + Bn(:,i)'*Bn(:,i);
    end 
    s = b/a;    
else
    s = 1;
end

%Compute the final translation
T = Cb - s*R*Ca;

%Compute the residual error
if(nargout>3)    
    err =0;
    for i=1:Na
        d = (B(:,i) - (s*R*A(:,i) + T));
        err = err + norm(d);
    end      
end


%Displayed if an error occurs
function usage()
disp('Usage:')
disp('[s R T error] = absoluteOrientationQuaternion( A, B, doScale)')
disp(' ')
disp('Return values:')
disp('s         The scale factor')
disp('R         The 3x3 rotation matrix')
disp('T         The 3x1 translation vector')
disp('err       Residual error (optional)')
disp(' ')
disp('Input arguments:')
disp('A         3xN matrix representing the N 3D points')
disp('B         3xN matrix representing the N 3D points')
disp('doScale   Optional flag indicating whether to estimate the uniform scale factor as well [default=0]')
disp(' ')

