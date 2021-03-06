[num1, txt1, t_train] = xlsread('./problem2/t_train.csv');
[num2, txt2, x_train] = xlsread('./problem2/x_train.csv');
c = 1000;
tol = 0.001;
train_num = 150;
row = horzcat(x_train,t_train);
row = cell2mat(row);
[w12,b12,sv12] = svm(row(1:100,1:2),[ones(50,1);-1*ones(50,1)],c,tol);
[w23,b23,sv23] = svm(row(51:end,1:2),[ones(50,1);-1*ones(50,1)],c,tol);
[w13,b13,sv13] = svm([row(1:50,1:2);row(101:end,1:2)],[ones(50,1);-1*ones(50,1)],c,tol);
%train_y
y12_train = row(:,1:2)*w12 + b12;
y23_train = row(:,1:2)*w23 + b23;
y13_train = row(:,1:2)*w13 + b13;
vote_train = zeros(train_num,3);
%voting
vote_train(:,1) = (y12_train>0)+(y13_train>0);
vote_train(:,2) = (y12_train<0)+(y23_train>0);
vote_train(:,3) = (y13_train<0)+(y23_train<0);
%plot_feature
x_min = -1;x_max = 1;y_min = -1;y_max = 1;
plot_num = 300;
w1_plot = linspace(x_min,x_max,plot_num)';
w2_plot = linspace(y_min,y_max,plot_num)';
plot_feature = [];
for i = 1:plot_num
    plot_feature = [plot_feature;repmat(w1_plot(i),plot_num,1),w2_plot];
end
y12_feature = plot_feature*w12 + b12;
y23_feature = plot_feature*w23 + b23;
y13_feature = plot_feature*w13 + b13;
vote_feature = zeros(plot_num*plot_num,3);
vote_feature(:,1) = (y12_feature>0) + (y13_feature>0);
vote_feature(:,2) = (y12_feature<0) + (y23_feature>0);
vote_feature(:,3) = (y13_feature<0) + (y23_feature<0);
[garbage, index_plot] = max(vote_feature,[],2);
%plot
h = zeros(1,9);
h(1) = plot(plot_feature(find(index_plot==1),1),plot_feature(find(index_plot==1),2),'.','Color',[1 0.7 0.7]); hold on;
h(2) = plot(plot_feature(find(index_plot==2),1),plot_feature(find(index_plot==2),2),'.','Color',[0.7 1 0.7]);
h(3) = plot(plot_feature(find(index_plot==3),1),plot_feature(find(index_plot==3),2),'.','Color',[0.7 0.7 1]);
train_class1 = row(1:50,1:2);
train_class2 = row(51:100,1:2);
train_class3 = row(101:150,1:2);
h(4) = plot(train_class1(:,1),train_class1(:,2),'rx','DisplayName','Class 1');
h(5) = plot(train_class2(:,1),train_class2(:,2),'gx','DisplayName','Class 2');
h(6) = plot(train_class3(:,1),train_class3(:,2),'bx','DisplayName','Class 3');
h(7) = plot(sv12(:,1),sv12(:,2),'ko','DisplayName','Support vector');
h(8) = plot(sv23(:,1),sv23(:,2),'ko');
h(9) = plot(sv13(:,1),sv13(:,2),'ko');
axis([x_min,x_max,y_min,y_max]);
legend(h(4:7));
title('Linear kernel(one-versus-one)')
