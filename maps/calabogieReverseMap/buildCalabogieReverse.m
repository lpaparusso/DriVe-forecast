clear all
close all

set(0,'DefaultFigureWindowStyle','docked');

% a=load('calabogie1.mat');
% b=load('calabogie2.mat');
% c=load('calabogie3.mat');
% d=load('calabogie4.mat');
% e=load('calabogie5.mat');
% f=load('calabogie6.mat');
% g=load('calabogie7.mat');
% h=load('calabogie8.mat');
% 
% figure(1)
% hold on
% plot(g.x_value(end-400:end),g.y_value(end-400:end),'Linewidth',1.3);
% plot(b.x_value(1:end-1100),b.y_value(1:end-1100),'Linewidth',1.3);
% plot(e.x_value(950:end),e.y_value(950:end),'Linewidth',1.3);
% plot(f.x_value(1:1130),f.y_value(1:1130),'Linewidth',1.3);
% xlabel('X [m]')
% ylabel('Y [m]')
% legend('g','b','e','f')
% axis equal
% sgtitle('Rough data')
% grid on

roadProperties =  readtable('Calabogie.csv', 'PreserveVariableNames', true);

centerline = [roadProperties{:,'x-coord'}.'; roadProperties{:,'y-coord'}.'; roadProperties{:,'z-coord'}.'];
roadWidth = roadProperties{:,'width'}.';
banking = rad2deg(roadProperties{:,'lat.inc'}).';

%add three more points more between beginning and end of the road
centerline = [centerline interp1([0 1],[centerline(:,end) centerline(:,1)].',[1/3, 2/3 0.999]).'];
roadWidth = [roadWidth interp1([0 1],[roadWidth(:,end) roadWidth(:,1)].',[1/3, 2/3 0.999])];
banking = [banking interp1([0 1],[banking(:,end) banking(:,1)].',[1/3, 2/3 0.999])];

% Compose the left and right margins
% leftMargin = [b.x_value(1:end-1100).' g.x_value(end-400:end).';
%     b.y_value(1:end-1100).' g.y_value(end-400:end).';
%     b.z_value(1:end-1100).' g.z_value(end-400:end).'];
% 
% rightMargin = [f.x_value(1:1130).' e.x_value(950:end).';
%     f.y_value(1:1130).' e.y_value(950:end).';
%     f.z_value(1:1130).' e.z_value(950:end).'];

scenario = drivingScenario;
road(scenario, centerline.', 2, banking);
rbdry = roadBoundaries(scenario);
margin = rbdry{1,1}.';
rightMargin_unit = margin(:,3:2462);
leftMargin_unit = [margin(:,2), margin(:, end:-1:2463)];

indices_width = zeros(1,size(rightMargin_unit,2));
for i=1:length(indices_width)
    [~, indexMin] = min(vecnorm(centerline-rightMargin_unit(:,i),2,1));
    indices_width(i) = indexMin;
end

rightMargin = rightMargin_unit;
leftMargin = leftMargin_unit;
for i=1:length(indices_width)
    rightMargin(:,i) = centerline(:,indices_width(i))*(1-roadWidth(indices_width(i))/2) + rightMargin(:,i) * roadWidth(indices_width(i))/2;
    leftMargin(:,i) = centerline(:,indices_width(i))*(1-roadWidth(indices_width(i))/2) + leftMargin(:,i) * roadWidth(indices_width(i))/2;
end

figure
plot3(centerline(1,:), centerline(2,:), centerline(3,:))
hold on
plot3(rightMargin(1,:), rightMargin(2,:), rightMargin(3,:))
plot3(leftMargin(1,:), leftMargin(2,:), leftMargin(3,:))
legend('center','right','left')
grid on
axis equal

% reverse calabogie
rightMargin_m = [rightMargin(:,2:end) rightMargin(:,1)]; % first I add the first point as last point
leftMargin_m = [leftMargin(:,2:end) leftMargin(:,1)];
rightMargin = leftMargin_m(:,end:-1:1); % then I invert Calabogie
leftMargin = rightMargin_m(:,end:-1:1);

%compute curvilinear abscissa for resampling
rightAbscissa = [ 0 cumsum(vecnorm(diff(rightMargin.').'))];
leftAbscissa = [ 0 cumsum(vecnorm(diff(leftMargin.').'))];

%create spline for left and right margin
Knot = 1500;
sp_left = spap2(Knot,5,leftAbscissa,leftMargin);
sp_right = spap2(Knot,5,rightAbscissa,rightMargin);


%find short and long margin
global sp_right_2d sp_left_2d short_margin long_margin S D_short left_line right_line s_long_search_centerline %ATTENTION!!!!  

%create spline 2d for intersections
sp_left_2d = spap2(Knot,5,leftAbscissa,leftMargin(1:2,:));
sp_right_2d = spap2(Knot,5,rightAbscissa,rightMargin(1:2,:));

if leftAbscissa(end)>= rightAbscissa(end)
    short_margin = sp_right_2d;
    long_margin = sp_left_2d;
    pp_short_margin = sp_right;
    pp_long_margin = sp_left;
    short_abscissa_end = rightAbscissa(end);
    S0=90;%initialization fsolve
else
    short_margin = sp_left_2d;
    long_margin = sp_right_2d;
    pp_short_margin = sp_left;
    pp_long_margin = sp_right;
    short_abscissa_end = leftAbscissa(end);
    S0=-90;%initialization fsolve
end
D_short = fnder(short_margin,1);

%find centerline 
i=1;
center_line= zeros(floor(short_abscissa_end),3);

%plot centerline
figure()
fnplt(sp_right_2d)
hold on
fnplt(sp_left_2d)
axis equal
k=1;
legend('right margin','left margin')
title('centerline search')
xlabel('X[m]')
ylabel('Y[m]')

%change margin to start with centerline search must be the shortest!!!
for S= 0 : 1 : short_abscissa_end 
    
int = @intersezione;
s = fminsearch(int,S0);%find intersection left margin with perpendicular vector
center_line(i,:) = mean([fnval(pp_short_margin,S), fnval(pp_long_margin,s_long_search_centerline) ],2);%find centerline point
i=i+1;

fprintf(['computing abscissa curvilinear: ', num2str(S) , ' of ',num2str(round(short_abscissa_end)), ' angle: ',num2str(s),'\n'])

%plot serch centerline
if k==5
longp= fnval( pp_long_margin,s_long_search_centerline);
shortp=fnval(pp_short_margin,S);
plot([longp(1),shortp(1)],[longp(2),shortp(2)],'HandleVisibility','off' )
scatter(center_line(i-1,1),center_line(i-1,2),'HandleVisibility','off')
k=1;
end
k=k+1;
end

%create spline centerline

Abscissa_centerline=[ 0 cumsum(vecnorm(diff(center_line).'))];
order=5;
break_position = Abscissa_centerline(1,3:  3  :end-3);%avoid zero and last, put in knots
knot_centerline=[zeros(1,order), break_position  ,  Abscissa_centerline(end)*ones(1,order)];
sp_centerline = spap2(knot_centerline , order ,Abscissa_centerline,center_line');

%plot spline centerline

figure()
scatter(center_line(:,1),center_line(:,2))
hold on
index_testo=1:10:length(Abscissa_centerline);
for index=1:length(index_testo)
 text(center_line(index_testo(index),1),center_line(index_testo(index),2),num2str(Abscissa_centerline(index_testo(index))))
 end
fnplt(sp_centerline)
xlabel('X[m]')
ylabel('Y[m]')
legend('centerline scatter','spline centerline')
title('centerline comparison with curvilinear abscissa notation')
grid on

%plot parametrized curve
figure()
plot(Abscissa_centerline,center_line(:,1));
hold on
plot(Abscissa_centerline,center_line(:,2));
plot(Abscissa_centerline,center_line(:,3));
title('spline centerline - parametrized curve')
xlabel('Curvilinear abscissa [m]')
ylabel('[m]')
legend('x','y','z')

%plot first derivative
figure()
interval=(0:0.1:Abscissa_centerline(end));
d_abs=vecnorm(fnval((fnder(sp_centerline,1)),interval));
d=fnval((fnder(sp_centerline,1)),interval);
plot(interval,d_abs)
hold on
plot(interval,d(1,:))
plot(interval,d(2,:))
plot(interval,d(3,:))
legend('norm 1st derivative','derX','derY','derZ')
grid on
title('1st derivative of centerline')
xlabel('Curvilinear Abscissa[m]')
ylabel('1st derivative [m/s]')


%plot second derivative

figure()
dd_abs=vecnorm(fnval((fnder(sp_centerline,2)),interval));
dd=fnval((fnder(sp_centerline,2)),interval);
plot(interval,dd_abs)
hold on
plot(interval,dd(1,:))
plot(interval,dd(2,:))
plot(interval,dd(3,:))
legend('norm 2nd derivative','derX','derY','derZ')
grid on
title('2nd derivative of centerline')
xlabel('Curvilinear Abscissa[m]')
ylabel('2nd derivative [m/s^2]')

%plot all margin and centerline spline and scatter data

figure()
scatter(leftMargin(1,:),leftMargin(2,:),'Linewidth',1.3)
hold on
scatter(rightMargin(1,:),rightMargin(2,:),'Linewidth',1.3)
scatter(center_line(:,1),center_line(:,2),'Linewidth',1.3)
xlabel('X [m]')
ylabel('Y [m]')
fnplt(sp_left) 
fnplt(sp_right)
fnplt(sp_centerline)
legend('Left margin','Right margin','Center line','spline left','spline right','spline centerline')
grid on
axis equal
title('spline with scatter data')

%calculate alla necessary data along centerline
%initialization
%%
curvilinearAb = 0: 0.1 : Abscissa_centerline(end); %sampling along the curvilinear abscissa

tangent_point=zeros(length(curvilinearAb),3);
curvature_sign=zeros(length(curvilinearAb),1);
curvature=zeros(length(curvilinearAb),1);
Norm_vector=zeros(length(curvilinearAb),3);
center_point=zeros(length(curvilinearAb),3);
leftMargin_point =zeros(length(curvilinearAb),3);
rightMargin_point = zeros(length(curvilinearAb),3);
DDx = zeros(length(curvilinearAb),1);
DDy = zeros(length(curvilinearAb),1);
DDz = zeros(length(curvilinearAb),1);
Dx = zeros(length(curvilinearAb),1);
Dy = zeros(length(curvilinearAb),1);
Dz = zeros(length(curvilinearAb),1);
curvature2d = zeros(1,length(curvilinearAb));

%derivative
D=fnder(sp_centerline,1);
DD=fnder(sp_centerline,2);
options = optimoptions('fsolve','Display','none','Algorithm','levenberg-marquardt');

for i = 1:length(curvilinearAb)  
    
x = curvilinearAb(i);
tangent_point(i,:)= fnval(D,x);
curvature(i)= vecnorm(fnval(DD,x)');% for definition this is the curvature 
Norm_vector(i,:)= fnval(DD,x) /curvature(i);%this is the versor of the normal vector
angle_norm2D = atan2(tangent_point(i,2),tangent_point(i,1))+( pi/2);
normV_2D = [cos(angle_norm2D), sin(angle_norm2D)];

%retive road point
center_point(i,:) = fnval(sp_centerline,x);

%left margin

left_line=interp1([0 10],[center_point(i,1),center_point(i,1)+normV_2D(1)*10;center_point(i,2), center_point(i,2)+ normV_2D(2)*10 ]','linear','pp');
int_left = @intersezione_margin_left;
if x >= leftAbscissa(end)
S0_left=[leftAbscissa(end)-2; 8]; 
else 
S0_left=[x; 8];%initialization fsolve
end
s_left = fsolve(int_left,S0_left,options);%find intersection left margin with perpendicular vector
leftMargin_point(i,:) = fnval(sp_left,s_left(1));

%right margin
right_line=interp1([0 10],[center_point(i,1),center_point(i,1)-normV_2D(1)*10;center_point(i,2), center_point(i,2)- normV_2D(2)*10 ]','linear','pp');
int_right = @intersezione_margin_right;
if x >= rightAbscissa(end)
S0_right=[rightAbscissa(end)-2; 8];    
else 
S0_right=[x; 8];%initialization fsolve
end
s_right = fsolve(int_right,S0_right,options);%find intersection right margin with perpendicular vector
rightMargin_point(i,:) = fnval(sp_right,s_right(1));

%sign curvature 2 dimension
dd=fnval(DD,x);
d=fnval(D,x);
DDx(i)=dd(1);
DDy(i)=dd(2);
DDz(i)=dd(3);
Dx(i)=d(1);
Dy(i)=d(2);
Dz(i)=d(3);

curvature2d(i) = (Dx(i)*DDy(i)-Dy(i)*DDx(i))/((Dx(i)^2+Dy(i)^2)^(3/2));%curvature with sign in 2d space
curvature_sign(i) = curvature(i)*sign(curvature2d(i));%curvature 3D with the sign of the 2d(left and right turn - not z bending)

fprintf(['computing abscissa curvilinear: ', num2str(x) , ' of ',num2str(curvilinearAb(end)),'\n'])
end

%plot second derivative
figure()
plot(curvilinearAb,DDx)
hold on
plot(curvilinearAb,DDy)
plot(curvilinearAb,DDz)
legend('second derivative along x','second derivative along y')
title('Second derivative')
xlabel('Curvilinear Abscissa')
ylabel('second derivative')
grid on

%plot curvature
figure()
plot(curvilinearAb,curvature2d)
hold on 
plot(curvilinearAb,curvature_sign)
plot(curvilinearAb,curvature)
legend('curvature 2D','curvature 3D signedd','curvature 3D')
title('curvature')
xlabel('Curvilinear Abscissa')
ylabel('curvature')
grid on

%plot 

figure()
scatter(center_point(:,1),center_point(:,2))
hold on
scatter(rightMargin_point(:,1),rightMargin_point(:,2))
scatter(leftMargin_point(:,1),leftMargin_point(:,2))
legend('Centerline','right margin','left margin')
title('exported data')
xlabel('X [m]')
ylabel('Y [m]')
grid on

figure()
plot(curvilinearAb,center_point(:,3))
hold on
plot(curvilinearAb,Dz)
plot(curvilinearAb,DDz)
title('centerline spline Z')
legend('spline z','first derivative','second derivative')
xlabel('Curvilinear abscissa [m]')
ylabel('magnitude of curve, 1st derivative and second derivative ')
grid on

%Consideration on banking and road width
roadWidth = vecnorm((leftMargin_point-rightMargin_point)');
Height_diff = rightMargin_point(:,3) - leftMargin_point(:,3);%difference on the 2 margin point on z ALL WRT RIGHT MARGIN
Angle_banking = asind(Height_diff./roadWidth');%angle banking in degree

figure()
plot(curvilinearAb,Angle_banking)
title('Angle of banking wrt curvilinear abscissa')
xlabel('Curvilinear abscissa [m]')
ylabel('Banking angle [deg]')
grid on

%create spline Banking
order=5;
break_position_banking = curvilinearAb(1, 3:  3  :end-3);%avoid zero and last, put in knots
knot_banking=[zeros(1,order), break_position_banking  ,  curvilinearAb(end)*ones(1,order)];
sp_banking = spap2(knot_banking , order ,curvilinearAb,Height_diff');
figure()
fnplt(sp_banking)
title('Banking spline')
xlabel('Curvilinear abscissa [m]')
ylabel('difference in height right-left margin [m]')
grid on

%save data
centerlineX = center_point(:,1)';
centerlineY = center_point(:,2)';
centerlineZ = center_point(:,3)';
tangentX = tangent_point(:,1)';
tangentY = tangent_point(:,2)';
tangentZ = tangent_point(:,3)';
normalX = Norm_vector(:,1)';
normalY = Norm_vector(:,2)';
normalZ = Norm_vector(:,3)';
leftMarginX = leftMargin_point(:,1)';
leftMarginY = leftMargin_point(:,2)';
leftMarginZ = leftMargin_point(:,3)';
rightMarginX = rightMargin_point(:,1)';
rightMarginY = rightMargin_point(:,2)';
rightMarginZ = rightMargin_point(:,3)';
curvature= curvature';
SecondDerZ=DDz';
FirstDerZ=Dz';
Angle_banking = Angle_banking.';

%Save data
save('calabogieReverseData','centerlineX','centerlineY','centerlineZ','curvature','curvature2d','curvilinearAb','leftMarginX','leftMarginY','leftMarginZ','rightMarginX','rightMarginY','rightMarginZ','normalX','normalY','normalZ','roadWidth','tangentX','tangentY','tangentZ','SecondDerZ','FirstDerZ', 'Angle_banking');

%function for intersection

function min_lenght = intersezione(X0)
global short_margin S D_short line s_long_search_centerline

Starting_point = fnval(short_margin ,S);
tangent= fnval(D_short ,S);
angle=atan2d(tangent(2),tangent(1))+ X0;
V_dir = [cosd(angle), sind(angle)];
line= interp1([0 30],[Starting_point(1),Starting_point(1)+V_dir(1)*30;Starting_point(2), Starting_point(2)+ V_dir(2)*30 ]','linear','pp');%perpendicular vetor to the right margin 

S0=[S,10];
fun=@intersezione_crf;%find for minimum distance between starting margin and other
options = optimoptions('fsolve','Display','none','Algorithm','levenberg-marquardt');
int = fsolve(fun,S0,options);

s_long_search_centerline=int(1);
min_lenght =  int(2);

end
function crossing = intersezione_crf(Ascisse)
global long_margin line
v1=fnval(long_margin, Ascisse(1));
v2=fnval(line,Ascisse(2));
crossing =vecnorm( v1-v2 );
end
function crossing = intersezione_margin_left(Ascisse)
global sp_left_2d  left_line

v1=fnval(sp_left_2d,Ascisse(1));
v2=fnval(left_line,Ascisse(2));
crossing =vecnorm( v1-v2 );

end
function crossing = intersezione_margin_right(Ascisse)
global sp_right_2d  right_line

v1=fnval(sp_right_2d,Ascisse(1));
v2=fnval(right_line,Ascisse(2));
crossing =vecnorm( v1-v2 );

end