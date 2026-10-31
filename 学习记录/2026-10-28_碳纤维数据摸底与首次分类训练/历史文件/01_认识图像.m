%% 第一次练习：认识图像，不训练模型
% 需要 MATLAB 和 Image Processing Toolbox。
% 本练习尚未在本机 MATLAB 执行；如遇报错，请保留完整错误信息。
rootDir = fileparts(mfilename('fullpath'));
sampleDir = fullfile(rootDir, '学习样图');
names = {'normal.jpg', 'recoverable.jpg', 'non_recoverable.jpg'};
labels = {'无缺陷', '可修复缺陷', '不可修复缺陷'};
figure('Name', '图像、灰度和直方图');
for k = 1:numel(names)
    I = imread(fullfile(sampleDir, names{k}));
    G = rgb2gray(I);
    fprintf('%s: size=%s, type=%s, range=[%g,%g]\n', ...
        names{k}, mat2str(size(I)), class(I), double(min(I(:))), double(max(I(:))));
    subplot(3, 3, (k-1)*3+1); imshow(I); title(labels{k});
    subplot(3, 3, (k-1)*3+2); imshow(G); title('灰度图');
    subplot(3, 3, (k-1)*3+3); imhist(G); title('灰度直方图');
end
% 思考：灰度转换保留和丢失了什么？直方图能告诉你缺陷的位置吗？
% 下一步可手写一个 3×3 均值滤波，观察微小缺陷是否被一起抹平。
