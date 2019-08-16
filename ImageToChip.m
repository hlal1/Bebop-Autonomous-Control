clear
clc

cd 'C:\Users\Owner\Desktop\folder'
new_dir_name = 'sat_cut/';

file_name = 'Scottsdale_upscaled.jpg';

full_img_data = imread(file_name);
img_size = size(full_img_data);
%image_size=[Y=9650,X=9434,3]
ymax = img_size(1);
xmax = img_size(2);

dim = 80;

max_M = floor(ymax/dim);
max_N = floor(xmax/dim);

%loop through sub sections
for M = 1:max_M % Y axis
    for N = 1:max_N % X axis

        cut = full_img_data(((M-1)*dim+1):(M*dim),...
                            ((N-1)*dim+1):(N*dim),:);

        imwrite(cut, strcat(new_dir_name,file_name(1:end-4),'/',file_name(1:end-4),'_',int2str(dim),'_A_',int2str(M),'_',int2str(N),'.jpg'))
    end
end


%50% X and Y offset to create
full_img_data = full_img_data((dim/2):end,(dim/2):end,:);

for M = 1:(max_M-1) % Y axis
    for N = 1:(max_N-1) % X axis

        cut = full_img_data(((M-1)*dim+1):(M*dim),...
                            ((N-1)*dim+1):(N*dim),:);

        imwrite(cut, strcat(new_dir_name,file_name(1:end-4),'/',file_name(1:end-4),'_',int2str(dim),'_B_',int2str(M),'_',int2str(N),'.jpg'))
    end
end
