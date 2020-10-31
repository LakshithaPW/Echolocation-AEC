dim_patch_single = [2 50];
num_basis =20;
topo_subspace = [num_basis num_basis];
topo_subspace_elev = [num_basis num_basis];
timeTrain = 500000;
PARAMSC = {dim_patch_single,topo_subspace,timeTrain,[],[],topo_subspace_elev};

alpha_v = 0.00001;
alpha_n = 0.000002;
alpha_p = 0.01; % learning rate to update the policy function
gamma = 0.05; % learning rate to update cumulative value;
lambda = 0.01; % regularizor
xi = 0.3;%0.3
tau = 1; % temperature in softmax function in policy network
dimension_Feature = 6*(num_basis*num_basis)+1;
dimension_Hid = 500;
initialWeightRange = [0.01,0.001]; % initial weight
%  value network {alpha_v,gamma,xi,dim_feature,init_weights_range};
PARAMCritic = {alpha_v,gamma,xi,dimension_Feature,initialWeightRange(1)};
% policy network {Action,alpha_p,alpha_n,lambda,tau,dimension_Feature,initialWeightRange};
actor_hidden_type = 'tanh';
PARAMActor = {alpha_p,alpha_n,dimension_Feature,initialWeightRange(2),actor_hidden_type};

PARAMRL = {PARAMCritic,PARAMActor};
model = Model(PARAMSC,PARAMRL);