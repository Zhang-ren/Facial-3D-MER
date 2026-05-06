function frame2gif(volume, filename)
%FRAME2GIF Save a 4-D image volume as an animated GIF.
for frame = 1:size(volume, 4)
    [indexed, cmap] = rgb2ind(im2uint8(volume(:, :, :, frame)), 256);
    if frame == 1
        imwrite(indexed, cmap, filename, 'gif', 'LoopCount', inf, 'DelayTime', 0.5);
    else
        imwrite(indexed, cmap, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.5);
    end
end
end
