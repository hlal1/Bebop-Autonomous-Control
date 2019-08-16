close all
clear
clc

video_name = 'Bebop2_20180414163256-0700';

videoReader = vision.VideoFileReader([video_name,'.mp4']);
frame_count = 0;

while ~isDone(videoReader)
    frame = step(videoReader);
    if mod(frame_count,15)==0
        % for vertical camera video
        im = im2uint8(frame);
        imwrite(im,[video_name,'/',video_name,'_frame',sprintf('%05d',frame_count),'.jpg'],'jpg');
    end
    frame_count = frame_count + 1;
end 
release(videoReader);