function [ xt_sweep ] = gen_random_fm_sweep_duration(T)

    f1=70000;
    f2=15000;
    Fs=500000;
    time_length=T;
    t=0:1/Fs:time_length-1/Fs;
    A=1;
    SLOPE=(f2-f1)./t(end);
    F=f1+SLOPE*(t/2);
    theta=-(2*pi*F(end)*t(end));
    
    Nfft=16384;  
    X=A.*sin(2*pi*F.*t+theta);
    X=[X zeros(1,Nfft-length(X))];
    fshift = (-Nfft/2+1:Nfft/2)*(Fs/Nfft);
    x_fit=fshift(Nfft/2+1:end);
    ep=1.2726;
    omega=3.1082;
    alpha=10.5031;
    shift=0.0048;
    x_bar=(x_fit/10000-ep)/omega;
    y_fit_log=log10((2/omega)*(1/sqrt(2*pi))*exp(-x_bar.^2/2).*((1/2)*(1+erf(alpha*x_bar/sqrt(2))))+shift);  
    y_fit=10.^(y_fit_log);
    y_fit=y_fit-min(y_fit);
    
    Amp=(0.5)*[y_fit conj(fliplr(y_fit))];
    X_f_sweep_amp = fft(X,Nfft).*Amp; 
    xt_sweep=real(ifft(X_f_sweep_amp,Nfft));
    xt_sweep=xt_sweep(1:length(t));


end



