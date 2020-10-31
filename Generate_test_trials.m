clear all;
close all;
clc;

addpath('HRTF');
addpath('GASSOM');
load('HRTF_Data/HRTF_2.mat');

HRTFl=HRTFl{1};
HRTFr=HRTFr{1};
select=~(logical(((azm<-90 | azm>90)|(elv<-90 | elv>90))));
azm=azm(select);
elv=elv(select);
lat=real(asind(sin(azm*pi/180).*cos(elv*pi/180)));
HRTFl=HRTFl(select,:);
HRTFr=HRTFr(select,:);
range=[-80 80 80];

file_names={'Subject_2_trail_1_sigma_10'};        

H=[];
for f=1:length(file_names) 
    
    %load pre-generated trajectories
    load([file_names{f} '/all_trajectories_final_policy.mat']);
    
    %load new trajectories for the given policy
    %load([file_names{f} '/learned_policy.mat']);
    %model=Results{1};    
    %[source_pos_final_azim,source_pos_final_elev,lat_loc,elev_loc,Calls,source_cng,Test_traj] = generate_trajectories(file_names{f},model,HRTFl,HRTFr,azm,elv,range);
    
    %select a call duration to plot
    for cl=[10]
        disp(cl);
        azim_call_trajs=source_pos_final_azim(Test_traj(:,3)==Calls(cl));
        elev_call_trajs=source_pos_final_elev(Test_traj(:,3)==Calls(cl));
        spatial_dir=[Test_traj((Test_traj(:,3)==Calls(cl)),1) Test_traj((Test_traj(:,3)==Calls(cl)),2)];
        azim_diff_init=reshape(azim_call_trajs,[source_cng length(lat_loc)]);
        elev_diff_init=reshape(elev_call_trajs,[source_cng length(lat_loc)]);
        %select initial directions to plot
        for j=[1 5 9 17 25 33 37 41]
            azim_path=azim_diff_init(1:20,j);
            elev_path=elev_diff_init(1:20,j);
            lat_path=real(asind(sin(azim_path*pi/180).*cos(elev_path*pi/180)));
            disp([lat_path(1) elev_path(1)]);
            [h1]=plot_dir(lat_path,elev_path,[0 0 1]); hold on;
            h2=scatter(lat_path(end),elev_path(end),'r','filled');
            H=[H h1 h2];
            axis([-80 80 -80 80]);
            set(gca,'XTick',-80:10:80);
            set(gca,'YTick',-80:10:80);
            xlabel('Horizontal(deg)');
            ylabel('Elevation(deg)');
            set(gca,'Fontsize',10);
            axis square;
            hold on;
        end
    end

end



