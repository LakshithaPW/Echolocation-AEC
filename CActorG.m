classdef CActorG < handle
    properties        
        alpha_p; alpha_n;
        input_dim; hidden_dim; output_dim;
        w_init_range;
        wp_ji;
        wp_kj;
        wn_ji; 
        g_wp_ji;
        g_wp_kj;
        g_wn_ji;
        norm_record;
        cmd_prev;        
        z_k_prev;
        z_j_prev;
        z_i_prev;
        iters;
        covmat;
        type_hidden;
        param_num;
    end
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = CActorG(PARAM)
            % PARAM = {Action,alpha_p,alpha_n,lambda,tau,dimension_Feature,initialWeightRange};
            obj.alpha_p = PARAM{1};
            obj.alpha_n = PARAM{2};                      
            obj.input_dim = PARAM{3};
            obj.hidden_dim = 500;
            obj.output_dim = 2;
            obj.w_init_range = PARAM{4};
            obj.type_hidden = PARAM{5};
            obj.covmat = 1*[1,0;0,1];
            obj.norm_record=zeros(1,6);
            obj.iters=0;
            %            
            obj.wn_ji = zeros((obj.input_dim)*obj.hidden_dim + (obj.hidden_dim)*obj.output_dim,1);
%             obj.wn_ji = (2*rand((obj.input_dim)*obj.hidden_dim + (obj.hidden_dim)*obj.output_dim,1));
%             obj.wn_ji = (2*rand((obj.input_dim*obj.hidden_dim) + (obj.hidden_dim*obj.output_dim),1)-1)*0.001;
%             obj.wp_ji = (2*rand(obj.hidden_dim,obj.input_dim)-1)*obj.w_init_range; 
%             obj.wp_kj = (2*rand(obj.output_dim,obj.hidden_dim)-1)*obj.w_init_range;
            %obj.wp_ji = (2*rand(obj.hidden_dim,obj.input_dim)-1)*0.00025; 
%             obj.wp_ji = (2*rand(obj.hidden_dim,obj.input_dim)-1)*0.0001; 
%             obj.wp_kj = (2*rand(obj.output_dim,obj.hidden_dim)-1)*0.001;
            obj.wp_ji = (2*rand(obj.hidden_dim,obj.input_dim)-1)*0.001; 
            obj.wp_kj = (2*rand(obj.output_dim,obj.hidden_dim)-1)*0.01;
            obj.g_wp_ji=zeros(size(obj.wp_ji));
            obj.g_wp_kj=zeros(size(obj.wp_kj));
            obj.g_wn_ji=zeros(size(obj.wn_ji));
            obj.param_num = 6;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function cmd = actDist(this,z_i)            
            a_j = this.wp_ji*z_i;
            switch this.type_hidden
                case 'tanh'
                    z_j = tanh(a_j);
                case 'softmax'
                    z_j = softmax(a_j);
            end           
            z_k = (180/pi)*this.wp_kj * z_j;
            %cmd = round(mvnrnd(z_k',this.covmat)); cmd = cmd';
            cmd = (mvnrnd(z_k',this.covmat)); cmd = cmd';
            % save current results
            this.cmd_prev = cmd;
            this.z_i_prev = z_i;            
            this.z_j_prev = z_j;
            this.z_k_prev = z_k;
        end
        function z_k = actHard(this,z_i)
            a_j = this.wp_ji*z_i;
            switch this.type_hidden
                case 'tanh'
                    z_j = tanh(a_j);
                case 'softmax'
                    z_j = softmax(a_j);
            end          
            z_k = (180/pi)*this.wp_kj * z_j;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function update(this,delta)
%             params=zeros(1,6);
            delta_k = (this.cmd_prev - this.z_k_prev);
%             delta_k = (this.cmd_prev - this.z_k_prev);
            switch this.type_hidden
                case 'tanh'
                    delta_j = (1-this.z_j_prev.^2).*(this.wp_kj' * delta_k);
                case 'softmax'
                    Zm = repmat(this.z_j_prev,1,this.hidden_dim);
                    Im = eye(this.hidden_dim);
                    temp = Zm.*(Im-Zm');
                    delta_j = temp' * (this.wp_kj' * delta_k);
            end
            %
            dlogp_vp = delta_k * this.z_j_prev';
            dlogp_wp = delta_j * this.z_i_prev';            
            %
            psi = [dlogp_wp(:);dlogp_vp(:)];
            dwn_ji = delta*psi - (psi'*this.wn_ji)*psi;
            this.wn_ji = this.wn_ji + this.alpha_n * dwn_ji;
            dwv = this.alpha_p * this.wn_ji;
            dwp = reshape(dwv(1:numel(dlogp_wp)),size(this.wp_ji));
            dvp = reshape(dwv(numel(dlogp_wp)+1:end),size(this.wp_kj));
            this.wp_ji = (1-(2*1e-4)*this.alpha_p)*this.wp_ji;
            this.wp_kj = (1-(2*1e-4)*this.alpha_p)*this.wp_kj;
            this.wp_ji = this.wp_ji + dwp;
            this.wp_kj = this.wp_kj + dvp;
            
            
            this.g_wp_ji=dwp;
            this.g_wp_kj=dvp;
            this.g_wn_ji=dwn_ji;
            % save results
%             params(1) = norm(this.wp_ji,'fro'); params(2) = norm(dwp,'fro');
%             params(3) = norm(this.wp_kj,'fro'); params(4) = norm(dvp,'fro');
%             params(5) = psi' * this.wn_ji;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [cmd,params] = train(this,feature,delta,iters,update)
            this.alpha_p=(0.5)*exp(-iters/100000)+0.001;
            sigma=100;%(-75)*exp(-iters/10000)+100;
            this.covmat = sigma*[1,0;0,1];
%             params = zeros(1,this.param_num);
            if(iters>1 && update)
                this.update(delta);
            end
            %record the norms
            this.norm_record(1)=norm(this.wp_ji,'fro');
            this.norm_record(2)=norm(this.wp_kj,'fro');
            this.norm_record(3)=norm(this.wn_ji,'fro');
            this.norm_record(4)=norm(this.g_wp_ji,'fro');
            this.norm_record(5)=norm(this.g_wp_kj,'fro');
            this.norm_record(6)=norm(this.g_wn_ji,'fro');
            
            cmd = this.actDist(feature);
            params=this.norm_record;
        end
    end
end