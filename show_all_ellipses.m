function show_all_ellipses(I, cx, cy, a, b, phi, color, ln_wid)
%% I: image on top of which you want to display the circles
%% cx, cy: column vectors with x and y coordinates of circle centers
%% a, b, phi: ellipse's param. 
%% The sizes of cx, cy, and rad must all be the same
%% color: optional parameter specifying the color of the circles
%%        to be displayed (red by default)
%% ln_wid: line width of circles (optional, 1.5 by default

if nargin < 7
    color = 'r';
end

if nargin < 8
   ln_wid = 1.5;
end

imshow(I); hold on;

theta = 0:0.1:(2*pi+0.1);
cx1 = cx(:,ones(size(theta)));
cy1 = cy(:,ones(size(theta)));
a1 = a(:,ones(size(theta)));
b1 = b(:,ones(size(theta)));
phi1 = phi(:,ones(size(theta)));
theta = theta(ones(size(cx1,1),1),:);
X = cx1 + a1.*cos(theta).*cos(phi1) - b1.*sin(theta).*sin(phi1);
Y = cy1 - a1.*cos(theta).*sin(phi1) - b1.*sin(theta).*cos(phi1);
line(X', Y', 'Color', color, 'LineWidth', ln_wid);

title(sprintf('%d circles', size(cx,1)));
