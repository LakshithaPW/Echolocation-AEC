classdef Model < handle
    properties
        scmodel;
        scmodel1;
        scmodel2;
        scmodel3;
        scmodel4;
        scmodel5;
        rlmodel;
        rlmodel1;
        rl_critic;
        rl_actor;
        iters;
        RL_norms;
        accumulated_error;
        reward_record;
    end
    methods 
        function obj = Model(PARAM,PARAMRL)
            training_length=500000;
            obj.scmodel = ASSOMOnline(PARAM);
            obj.scmodel1 = ASSOMOnline(PARAM);
            obj.scmodel2 = ASSOMOnline_Mon(PARAM);
            obj.scmodel3 = ASSOMOnline_Mon(PARAM);
            obj.scmodel4 = ASSOMOnline_Mon(PARAM);
            obj.scmodel5 = ASSOMOnline_Mon(PARAM);
            
            obj.rl_critic = CCritic(PARAMRL{1});
            obj.rl_actor = CActorG(PARAMRL{2});
            obj.iters=0;
            obj.RL_norms=zeros(training_length,11);
        end
        function [feature,reward,error_f,error_c]=gassom_encode(this,x_left,x_right,x_left_c,x_right_c,update_flag)
            X=([x_left;x_right]);
            [X]=this.pre_process_ind(X);
            
            X_c=([x_left_c;x_right_c]);
            [X_c]=this.pre_process_ind(X_c);
            [coef_f,error_f] = this.scmodel.sparseEncode(X,false);
            [coef_c,error_c] = this.scmodel1.sparseEncode(X_c,false);
            feature=[mean(coef_f,2);mean(coef_c,2)];

            reward=-(error_f+error_c);    
            if update_flag==1
                this.scmodel.updateBasis(X);
                this.scmodel1.updateBasis(X_c);
            end
            this.iters=this.iters+1;
        end
        function [feature,reward,mon_reward_left,mon_reward_right,xl_error_f,xlc_error_c,xr_error_f,xrc_error_c]=gassom_encode_elev(this,x_left,x_right,x_left_c,x_right_c,update_flag)
            X_l=(x_left);
            X_r=(x_right);
            [X_l]=this.pre_process_ind(X_l);
            [X_r]=this.pre_process_ind(X_r);
            X_l_c=x_left_c;
            X_r_c=x_right_c;
            [X_l_c]=this.pre_process_ind(X_l_c);
            [X_r_c]=this.pre_process_ind(X_r_c);            

            [xl_coef_f,xl_error_f] = this.scmodel2.sparseEncode(X_l,false);
            [xlc_coef_c,xlc_error_c] = this.scmodel3.sparseEncode(X_l_c,false);
            [xr_coef_f,xr_error_f] = this.scmodel4.sparseEncode(X_r,false);
            [xrc_coef_c,xrc_error_c] = this.scmodel5.sparseEncode(X_r_c,false);
            xl_feature=[mean(xl_coef_f,2);mean(xlc_coef_c,2)];
            xl_reward=-(xl_error_f+xlc_error_c);  
            xr_feature=[mean(xr_coef_f,2);mean(xrc_coef_c,2)];
            xr_reward=-(xr_error_f+xrc_error_c); 

            if update_flag==1
                this.scmodel2.updateBasis(X_l);
                this.scmodel3.updateBasis(X_l_c);
                this.scmodel4.updateBasis(X_r);
                this.scmodel5.updateBasis(X_r_c);
            end
            feature=[xl_feature;xr_feature];
            reward=(xl_reward+xr_reward);
            mon_reward_left=xl_reward;
            mon_reward_right=xr_reward;
        end

       function [tune_res]=get_tune_response_fine(this,x_left,x_right,sub)
            X=([x_left;x_right]);
            X=this.pre_process(X);
            [tune_res] = this.scmodel.get_response(X,sub);
        end
        function [tune_res]=get_tune_response_coarse(this,x_left,x_right,sub)
            X=([x_left;x_right]);
            X=this.pre_process(X);
            [tune_res] = this.scmodel1.get_response(X,sub);
        end
        function [tune_res]=get_tune_response_fine_left(this,x_left,sub)
            X=([x_left]);
            [tune_res] = this.scmodel2.get_response(X,sub);
        end
        function [tune_res]=get_tune_response_fine_right(this,x_right,sub)
            X=([x_right]);
            [tune_res] = this.scmodel4.get_response(X,sub);
        end
        function [tune_res]=get_tune_response_coarse_left(this,x_left,sub)
            X=([x_left]);
            [tune_res] = this.scmodel3.get_response(X,sub);
        end
        function [tune_res]=get_tune_response_coarse_right(this,x_right,sub)
            X=([x_right]);
            [tune_res] = this.scmodel5.get_response(X,sub);
        end
        function [tune_res]=get_gassom_response(this,x_left,x_right,sub)
            X=([x_left;x_right]);
            X=this.pre_process(X);
            [coef_f,~] = this.scmodel.sparseEncode(X,false);
            feature=[mean(coef_f,2)];
            tune_res=feature(sub);
        end

        function command=rl_command_gauss(this,feature,reward,update_flag)
            input_rl=[feature;0.001];
            update_flag=logical(update_flag);
            [delta,params_v] = this.rl_critic.train(input_rl,reward,this.iters,update_flag);
            [command,params_p] = this.rl_actor.train(input_rl,delta,this.iters,update_flag);
            
            this.RL_norms(this.iters,1)=params_p(1);
            this.RL_norms(this.iters,2)=params_p(2);
            this.RL_norms(this.iters,3)=params_p(3);
            this.RL_norms(this.iters,4)=params_p(4);
            this.RL_norms(this.iters,5)=params_p(5);
            this.RL_norms(this.iters,6)=params_p(6);
            this.RL_norms(this.iters,9)=params_v(1);
            this.RL_norms(this.iters,10)=params_v(2);
        end
        function value_func=rl_get_value(this,feature)
            input_rl=[feature;0.001];
            value_func=this.rl_critic.forward(input_rl);
        end
        function command=rl_test_gauss(this,feature,reward,update_flag)
            input_rl=[feature;0.001];
            [command]=this.rl_actor.actHard(input_rl);

        end
        function im_out=pre_process(this,im_in)
            X=im_in;
            X = X-ones(size(X,1),1)*mean(X,1);
            Energy_window=sum(X.^2);
            X = bsxfun(@rdivide, X, repmat(sqrt(Energy_window),[size(X,1) 1])+eps);
            im_out=X;
        end
        function [out]=pre_process_ind(this,inp)
            X=[inp];
            N_wins=size(inp,2);
            sum_norm=sum(X.^2);
            norm_coeff=(1/(N_wins))*sum(sum_norm);
            X = bsxfun(@rdivide, X, repmat(sqrt(norm_coeff),[size(X)])+eps);
            out=X;
        end


    end
end