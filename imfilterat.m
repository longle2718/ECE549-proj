function fval = imfilterat(x, y, r, im, h)
% Filter an image using 'h' at a particular location 'x' and 'y' with window of 
% half-size 'r' in the image 'im' 
%
impad = padarray(im, [r r], 'symmetric');
fval = impad(y:y+2*r,x:x+2*r).*h;
fval = sum(fval(:));

end