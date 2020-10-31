function [source_pos_final_azim,source_pos_final_elev,lat_loc,elev_loc,Calls,source_cng,Test_traj] = gennerate_trajectories(folder,model,HRTFl,HRTFr,azm,elv,learn_range)

source_cng=20;
Nfft=8192;
Fs=500000;
space=source_cng;
lat_range=-80:20:80;
elev_range=-80:20:80;
lat=real(asind(sin(azm*pi/180).*cos(elv*pi/180)));
[lat_loc,elev_loc]=meshgrid(lat_range,elev_range);
y_min=learn_range(1)*(1-abs(lat_loc(:))/learn_range(3));
y_max=learn_range(2)*(1-abs(lat_loc(:))/learn_range(3));
select_rng=(elev_loc(:)>=y_min)&(elev_loc(:)<=y_max);
elev_loc=elev_loc(select_rng);
lat_loc=lat_loc(select_rng);
azim_loc=asind(sind(lat_loc)./cosd(elev_loc));


%Calls=[linspace(0.4,0.5+3*(0.25/2),10) linspace(2.25-3*(0.75/2),2.25+3*(0.75/2),10) linspace(10-3*(3/2),10+3*(3/2),10)]*1e-3;
Calls=[linspace(1,5,30)]*1e-3;
N_calls=length(Calls);
indx=1:length(lat_loc);
[Index,C]=meshgrid(indx,Calls);
Test_traj=zeros(length(Index(:))*space,2);
A_extend=(repmat(azim_loc(Index(:)),[1 space]))';
E_extend=(repmat(elev_loc(Index(:)),[1 space]))';
C_extend=(repmat(C(:),[1 space]))';
Test_traj(:,1)=A_extend(:);
Test_traj(:,2)=E_extend(:);
Test_traj(:,3)=C_extend(:);

%%
N_total=size(Test_traj,1);
reset=false;
window=50;
stride=5;
offset=1600;
combined_source_azim=zeros(N_total,1);
combined_source_elev=zeros(N_total,1);
interm_results_azim=zeros(51,1);
interm_results_elev=zeros(51,1);
interm_results_azim_std=zeros(51,1);
interm_results_elev_std=zeros(51,1);
Hist_angles=zeros(N_total,1);

source_pos_azim=zeros(N_total,1);
source_pos_elev=zeros(N_total,1);
head_pos_azim=zeros(N_total,1);
head_pos_elev=zeros(N_total,1);
rel_azim=zeros(N_total,1);
rel_elev=zeros(N_total,1);
action_record=zeros(N_total,2);
reset_record=zeros(N_total,1);

for i=1:N_total
    disp(i);
    if(mod(i,source_cng)==1)
        duration=Test_traj(i,3);

        X=gen_random_fm_sweep_duration(duration);
        X_non_zero=X;
        source_azim=Test_traj(i,1);
        source_elev=Test_traj(i,2);
        source_lat=asind(sin(source_azim*pi/180).*cos(source_elev*pi/180));
        [x_s,y_s,z_s]=sph2cart(source_azim*pi/180,source_elev*pi/180,1);
        source_vec=[x_s;y_s;z_s];
    end

    source_pos_azim(i,1)=source_azim;
    source_pos_elev(i,1)=source_elev;

    [HRTFl_interp, HRTFr_interp] = HRTF_gauss( HRTFl, HRTFr, lat, elv ,source_lat, source_elev );

    Dr = HRTFr_interp;
    Dl = HRTFl_interp;

    DTFs{1}=Dr;
    DTFs{2}=Dl; 

    [IR_min_phase, IR] = Imp_Res(DTFs);   
    x_r=conv(X_non_zero,IR{1});
    x_l=conv(X_non_zero,IR{2});
    x_r=x_r(offset:end);
    x_l=x_l(offset:end);
    T=length(x_r);
            
    joint_norm=[x_r x_l];
    x_r_norm = ( x_r );
    x_l_norm = ( x_l );
    x_r=joint_norm(1:length(x_r));
    x_l=joint_norm(length(x_r)+1:end);  

    fine_index=bsxfun(@plus, (0:stride:T-2*window)', 1:window);
    coarse_index=bsxfun(@plus, (0:stride:T-2*window)', 1:2*window);

    x_l_win=(x_l(fine_index))';
    x_r_win=(x_r(fine_index))';
    x_l_win_norm=(x_l_norm(fine_index))';
    x_r_win_norm=(x_r_norm(fine_index))';
    x_l_win_c=resample((x_l(coarse_index))',1,2);
    x_r_win_c=resample((x_r(coarse_index))',1,2);
    x_l_win_c_norm=resample((x_l_norm(coarse_index))',1,2);
    x_r_win_c_norm=resample((x_r_norm(coarse_index))',1,2);

    [feature_azim,reward_azim]=model.gassom_encode(x_l_win,x_r_win,x_l_win_c,x_r_win_c,0);
    [feature_elev,reward_elev]=model.gassom_encode_elev(x_l_win_norm,x_r_win_norm,x_l_win_c_norm,x_r_win_c_norm,0);


    feature=[feature_azim;feature_elev];
    feature=(feature-mean(feature))/std(feature);
    reward=(reward_azim+reward_elev)/3;
    command=model.rl_test_gauss(feature,reward,0);

    Re_command=[cosd(-command(2)) 0 sind(-command(2));0 1 0;-sind(-command(2)) 0 cosd(-command(2))];
    Ra_command=[cosd(command(1)) -sind(command(1)) 0;sind(command(1)) cosd(command(1)) 0;0 0 1];
    source_vec=Re_command*Ra_command*source_vec;
    [source_azim,source_elev,r] = cart2sph(source_vec(1),source_vec(2),source_vec(3));
    source_azim=source_azim*180/pi;
    source_elev=source_elev*180/pi;  
    source_lat=asind(sin(source_azim*pi/180).*cos(source_elev*pi/180));

    action_record(i,:)=command';
    reset_record(i,1)=double(reset);
end
source_pos_final_azim=source_pos_azim;
source_pos_final_elev=source_pos_elev;

save([folder '/all_trajectories_final_policy.mat'],'Test_traj','source_cng','source_pos_final_azim','source_pos_final_elev','lat_loc','elev_loc','Calls'); 


end

