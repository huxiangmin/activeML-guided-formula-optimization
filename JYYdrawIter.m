%% mesh配方的生成
PFm=zeros(910,4);
ij=0;

for j=1:13
    for k=2:14
        for i=0:9
            if k>j
                ij=ij+1;
                PFm(ij,:)=[j,k,10*i,10*(10-i)];
            end
        end
    end
end
%% 做结果演进图 【正式版本，读取表格数据  -->  画图】
fPathIn="D:\uProjects\AI\PJHjyy1\1.传感课题执行\PJscripts\DataIn";
fPathOut="D:\uProjects\AI\PJHjyy1\1.传感课题执行\PJscripts\DataOut";

load('CustomColormap.mat');%使用colormapeditor编辑后，导出到工作空间并保存

filenameMesh=fullfile(fPathIn,"mesh_表格.xlsx");
PFmesh = xlsread(filenameMesh, 'sheet1');
[X,Y]=meshgrid(1:91, 0:10:90);

iterFIn={'0';'01';'012';'0123';'01234'};
for iter=1:5
    filename=fullfile(fPathIn,strcat(iterFIn{iter},"_结果表格.xlsx"));
    PFm = xlsread(filename, 'sheet1');

    filename=fullfile(fPathIn,strcat(num2str(iter),"_结果表格.xlsx"));
    PFm1 = xlsread(filename, 'sheet1');

    filename0=fullfile(fPathOut,"R2iters.xlsx");
    PFy = xlsread(filename0, ['sheet' num2str(iter)]);
    
    face0=reshape(PFy(:,5),10,91)';
    %PF0=[] %【需要从结果表格拷贝溶剂1、溶剂2、比例、稳定性】
    for i=1:size(PFm,1)
        for j=1:91
            k=10*j-9;                                                                                                                
            if (PFmesh(k,1)==PFm(i,1)) && (PFmesh(k,2)==PFm(i,2))  %【是否减1需要检查一下】
                PFm(i,5)=j;
                continue;
            end
        end
    end

    for i=1:size(PFm1,1)
        for j=1:91
            k=10*j-9;                                                                                                                
            if (PFmesh(k,1)==PFm1(i,1)) && (PFmesh(k,2)==PFm1(i,2))  %【是否减1需要检查一下】
                PFm1(i,5)=j;
                continue;
            end
        end
    end

    figure(12);subplot(5,1,iter);
    contourf(X,Y,face0','LineStyle','none','LevelStep',3);colormap(CustomColormap);

    hold on;
    Face0point=PFm;
    colors = CustomColormap; % 使用 colormapeditor 编辑后，导出到工作空间
    c = round((Face0point(:,6)/40) * (length(colors)-1)) + 1; % 计算颜色索引
    c(c>40)=40;c(c<1)=1;
    scatter(Face0point(:,5),Face0point(:,3), 36, colors(c,:), 'filled','MarkerEdgeColor','green') % 创建散点图，大小为36，填充颜

    %plot(PFm(:,5),PFm(:,3),'dr','MarkerSize',7,'MarkerFaceColor','red')
    plot(PFm1(:,5),PFm1(:,3),'dr','MarkerSize',12,'MarkerFaceColor','green')
end

%% 三维结果面
surf(face0');hold on;colormap(CustomColormap);
colors = CustomColormap; % 使用colormapeditor编辑后，导出到工作空间
c = round((Face0point(:,3)/40) * (length(colors)-1)) + 1; % 计算颜色索引
c(c>40)=40;c(c<1)=1;
scatter3(Face0point(:,1),Face0point(:,2), Face0point(:,3),36, colors(c,:), 'filled') % 创建散点图，大小为36，填充颜色
%% 二维伪彩色图
contourf(face0');hold on;colormap(CustomColormap);
colors = CustomColormap; % 使用 colormapeditor 编辑后，导出到工作空间
c = round((Face0point(:,3)/40) * (length(colors)-1)) + 1; % 计算颜色索引
c(c>40)=40;c(c<1)=1;
scatter(Face0point(:,1),Face0point(:,2), 36, colors(c,:), 'filled') % 创建散点图，大小为36，填充颜色
% 需要手工调整点的边框啥的