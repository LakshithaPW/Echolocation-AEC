classdef CCritic < handle
    properties
        alpha_v;
        gamma1;
        gamma2;
        input_dim;
        v_ji;
        v_init_range;
        norm_param;
        z_i_prev;
        z_j_prev;
        J;
    end
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function obj = CCritic(PARAM)
            % PARAM = {alpha_v,gamma1,gamma2,input_dim,v_init_range};
            obj.alpha_v = PARAM{1};
            obj.gamma1 = PARAM{2};
            obj.gamma2 = PARAM{3};
            obj.input_dim = PARAM{4};
            obj.v_init_range = PARAM{5};
            obj.norm_param=zeros(1,2);
            %
            obj.v_ji = (2*rand(1,obj.input_dim)-1)*obj.v_init_range;
            obj.z_j_prev = 0;
            obj.J = 0;
%             obj.J = -40;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function z_j = forward(this,z_i)           
            z_j = this.v_ji * z_i;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [delta,params] = update(this,z_j,reward)
            params = zeros(1,2);            
            this.J = (1-this.gamma1) * this.J + this.gamma1*reward;
            delta = reward - this.J + this.gamma2 * z_j - this.z_j_prev;            
            dv_ji = this.alpha_v * delta * this.z_i_prev';
%             this.v_ji = (1-0.005*this.alpha_v)*this.v_ji;
            this.v_ji = this.v_ji +  dv_ji;
            params(1) = norm(this.v_ji,'fro');
            params(2) = delta;
            this.norm_param(1,1)=norm(this.v_ji,'fro');
            this.norm_param(1,2)=delta;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [delta,params] = train(this,z_i,reward,iters,update)
            delta = 0;
%             params = [0,0];
%             this.alpha_v=(0.2)*exp(-iters/100000)+0.01;
%             this.alpha_v=0.5;
            z_j = this.forward(z_i);
            if(iters>1 && update)
                [delta,~] = this.update(z_j,reward);
            end
            params=this.norm_param;
            this.z_i_prev = z_i;
            this.z_j_prev = z_j;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end