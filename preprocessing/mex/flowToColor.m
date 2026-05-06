function img = flowToColor(flow)
%FLOWTOCOLOR Convert a two-channel flow field to an RGB visualization.
u = flow(:, :, 1);
v = flow(:, :, 2);

rad = sqrt(u.^2 + v.^2);
ang = atan2(-v, -u) / pi;

hue = (ang + 1) / 2;
saturation = ones(size(hue));
value = rad ./ (max(rad(:)) + eps);

img = hsv2rgb(cat(3, hue, saturation, value));
end
