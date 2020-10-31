function [ waggle_seq ] = generate_waggles_ar5(k)
    phi=[0.6926 -0.0557 -0.0350 0.0786 -0.1395];
    sigma=k*8;
    length_trial=25;
    x=zeros(length_trial,1);
    mu=10*randn(1);
    for t=5:length_trial-1
        x(t+1)=phi*x(t:-1:t-5+1)+sigma*randn(1);
    end
    waggle_seq=mu+x;
    waggle_seq=waggle_seq(6:end);
end



