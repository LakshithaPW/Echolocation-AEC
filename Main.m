clear all;
close all;
clc;

addpath('HRTF');
addpath('GASSOM');

%set up the random seed
random_seeds=1;
rng(random_seeds);
%set head roll standard deviation
sigma=10;
sigma_index=1;
%set trial number
trial=1;

%load HRTF data
load('HRTF_Data/HRTF_2.mat');
HRTFl=HRTFl{1};
HRTFr=HRTFr{1};
select=~(logical(((azm<-90 | azm>90)|(elv<-90 | elv>90))));
azm=azm(select);
elv=elv(select);
lat=real(asind(sin(azm*pi/180).*cos(elv*pi/180)));
HRTFl=HRTFl(select,:);
HRTFr=HRTFr(select,:);

%create folder
folder_name=['Subject_2_trail_' num2str(trial) '_sigma_' num2str(sigma) ''];
mkdir(folder_name);

%parameters
N_total=500000;
interm=10000;
reset=false;
Thresh=0.2;
Fs=500000;
window=50;
stride=5;

%initialize model and variables

InitGassom;

source_pos_azim=zeros(N_total,1);
source_pos_lat=zeros(N_total,1);
source_pos_elev=zeros(N_total,1);

N_check_points=(N_total/interm)+1;
Validation_error=zeros(N_check_points,2);

head_pos_azim=zeros(N_total,1);
head_pos_elev=zeros(N_total,1); 
rel_azim=zeros(N_total,1);
action_record=zeros(N_total,2);
waggle_record=zeros(N_total,1);
head_record=zeros(N_total,3);
reset_record=zeros(N_total,1);
Hist_angles=zeros(N_total,1);
learn_range=[-80 80 80];
source_pos_azim(1)=0;
source_pos_elev(1)=0;
source_pos_lat(1)=0;
roll_angle=0;


for i=1:N_total

    if(i<N_total/4)
        angs=((i-1)*(learn_range-learn_range/4))/(N_total/4-1)+learn_range/4;
    else
        angs=learn_range;
    end
    ss_dur=20;
    if(mod(i,ss_dur)==1)
        a=8.0;b=0.3;
        t_dur=gamrnd(a,b)*1e-3;       
        [ waggle_seq ] = generate_waggles_ar5(sigma_index);
        if(t_dur<0.4*1e-3)
            t_dur=0.4*1e-3;
        end
        
        X=gen_random_fm_sweep_duration(t_dur);
        offset=1600+randi(5);
        while(1)
            source_lat=(2*rand(1)-1)*(angs(3));
            source_elev=(angs(2)-angs(1))*rand(1)+angs(1);        
            y_min=angs(1)*(1-abs(source_lat)/angs(3));
            y_max=angs(2)*(1-abs(source_lat)/angs(3));
            if(source_elev>y_min && source_elev<y_max)
                break
            end
        end 
        
        source_azim=asind(sind(source_lat)./cosd(source_elev));
        [x_s,y_s,z_s]=sph2cart(source_azim*pi/180,source_elev*pi/180,1);
        source_vec=[x_s;y_s;z_s];
        source_vec_world=source_vec;
        R_H_W=eye(3);
    end
    
    source_pos_azim(i,1)=source_azim;
    source_pos_elev(i,1)=source_elev;
    source_pos_lat(i,1)=source_lat;
    
    [HRTFl_interp, HRTFr_interp] = HRTF_gauss( HRTFl, HRTFr, lat, elv ,source_lat, source_elev );
    
    Dr = HRTFr_interp;
    Dl = HRTFl_interp;
    
    DTFs{1}=Dr;
    DTFs{2}=Dl; 
    [IR_min_phase, IR] = Imp_Res(DTFs); 
    x_r=conv(X,IR{1});
    x_l=conv(X,IR{2});
    x_r=x_r(offset:end);
    x_l=x_l(offset:end);
    T=length(x_r);

    joint_norm=[x_r x_l];
    x_r_norm = x_r;
    x_l_norm = x_l;
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

    [feature_azim,reward_azim]=model.gassom_encode(x_l_win,x_r_win,x_l_win_c,x_r_win_c,1);
    [feature_elev,reward_elev]=model.gassom_encode_elev(x_l_win_norm,x_r_win_norm,x_l_win_c_norm,x_r_win_c_norm,1);

    feature=[feature_azim;feature_elev];
    feature=(feature-mean(feature))/std(feature);
    reward=(reward_azim+reward_elev)/6;
   	command=model.rl_command_gauss(feature,reward,1);
    
    Re_command=[cosd(-command(2)) 0 sind(-command(2));0 1 0;-sind(-command(2)) 0 cosd(-command(2))];
    Ra_command=[cosd(command(1)) -sind(command(1)) 0;sind(command(1)) cosd(command(1)) 0;0 0 1];

    R_H_W=Re_command*Ra_command*R_H_W;
    R_W_H=pinv(R_H_W);   
    H_w=R_W_H*[1;0;0];   
    [head_azim,head_elev,~] = cart2sph(H_w(1),H_w(2),H_w(3));    
    gamma_list=(head_azim*head_elev)/2;
    
    seq_indx=mod(i,ss_dur).*(mod(i,ss_dur)~=0)+(mod(i,ss_dur)==0)*ss_dur;
    roll_angle=waggle_seq(seq_indx);
    waggle_record(i)=roll_angle;
    gamma_waggle=roll_angle*pi/180;
    head_record(i,:)=[head_azim head_elev gamma_waggle]*180/pi;
    
    gamma_head=(gamma_list+gamma_waggle); %rad
    
    %new rotation matrix in Fick coordinates
    R_z_theta=[cos(head_azim) -sin(head_azim) 0;sin(head_azim) cos(head_azim) 0;0 0 1];
    R_y_phi=[cos(-head_elev) 0 sin(-head_elev);0 1 0;-sin(-head_elev) 0 cos(-head_elev)];
    R_x_gamma=[1 0 0;0 cos(gamma_head) -sin(gamma_head);0 sin(gamma_head) cos(gamma_head)];
    R_W_H_new=R_z_theta*R_y_phi*R_x_gamma;
    
    R_H_W=pinv(R_W_H_new);
    
    source_vec=R_H_W*source_vec_world;
    
    [source_azim,source_elev,r] = cart2sph(source_vec(1),source_vec(2),source_vec(3));
    source_azim=source_azim*180/pi;
    source_elev=source_elev*180/pi;
    source_lat=asind(sin(source_azim*pi/180).*cos(source_elev*pi/180));
    
    if(abs(source_lat)<angs(3))
        elv_min=angs(1)*(1-abs(source_lat)/angs(3));
        elv_max=angs(2)*(1-abs(source_lat)/angs(3));
        if(source_elev<elv_min || source_elev>elv_max)
            reset=true;
            while(1)
                source_lat=(2*rand(1)-1)*(angs(3));
                source_elev=(angs(2)-angs(1))*rand(1)+angs(1);        
                y_min=angs(1)*(1-abs(source_lat)/angs(3));
                y_max=angs(2)*(1-abs(source_lat)/angs(3));
                if(source_elev>y_min && source_elev<y_max)
                    break
                end
            end
            
            source_azim=asind(sind(source_lat)./cosd(source_elev));
            [x_s,y_s,z_s]=sph2cart(source_azim*pi/180,source_elev*pi/180,1);
            source_vec=[x_s;y_s;z_s];
            source_vec_world=source_vec;
            R_H_W=eye(3);

        end
    else
        reset=true;
        while(1)
            source_lat=(2*rand(1)-1)*(angs(3));
            source_elev=(angs(2)-angs(1))*rand(1)+angs(1);        
            y_min=angs(1)*(1-abs(source_lat)/angs(3));
            y_max=angs(2)*(1-abs(source_lat)/angs(3));
            if(source_elev>y_min && source_elev<y_max)
                break
            end
        end
        source_azim=asind(sind(source_lat)./cosd(source_elev));
        [x_s,y_s,z_s]=sph2cart(source_azim*pi/180,source_elev*pi/180,1);
        source_vec=[x_s;y_s;z_s];
        source_vec_world=source_vec;
        R_H_W=eye(3);

    end
    
    action_record(i,:)=command';
    reset_record(i,1)=double(reset);
    
    if(mod(i,interm)==1 || i==N_total)
        disp(['Complete Percentage:' num2str((i/N_total)*100)]);
        interm_index=floor(i/interm);
        file_name=[folder_name '/interm_policy_' num2str(interm_index) '.mat'];
        disp('Validating...');
        [validation_error1,validation_error2_std] = check_early_stopping(model,HRTFl,HRTFr,azm,elv,learn_range);
        disp('Validation done');
        
        Validation_error(interm_index+1,1)=validation_error1;
        Validation_error(interm_index+1,2)=validation_error2_std;
        %[model] = free_memory_unnc(model);
        Results{1}=model;
        Results{2}=source_pos_azim;
        Results{3}=source_pos_elev;
        Results{4}=action_record;
        Results{5}=head_record;
        Results{6}=source_pos_lat;
        Results{7}=Validation_error;
        Results{8}=waggle_record;
        Results{9}=random_seeds(trial);
        save(file_name,'Results'); 
        
        Validation_select=Validation_error(1:interm_index+1);
        valid_diff=diff(Validation_select);
        Valid_count=(abs(valid_diff)<Thresh);
        if(interm_index>2)
            if(sum(Valid_count(interm_index-2:interm_index))>=3)
                disp('Valid error doesnt change for 3 checkpoints...');
                save([folder_name '/minimum_validation.mat'],'interm_index','Valid_count','Validation_error'); 
                break;
            end
        end        
    end
    
    
end
