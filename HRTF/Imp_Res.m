function [IR_min_phase, IR] = Imp_Res(DTFs)

for g = 1:2
    [a b] = size(DTFs{g});
    r = find(([a b] == 339) == 0); % 339 is stardard length of the DTFs/HRTFs
    if r == 2 DTFs = DTFs{g}';end; % DTFs are are on different rows
    t = size(DTFs{1},1);

%    if g == 1
%        freq = (42:380)/2048*500e3;
%        qwe = 2048*angle(DTFs{g}(:,end))/(2*pi*41);           %%%% delay in integer
%        for k=1:t phase_shift(k,:) = exp(-i*2*pi*qwe(k)*(41:379)/2048);end
%    end;

 %   DTFs{g} = DTFs{g}.*phase_shift;


    asd = [zeros(t,41) DTFs{g} zeros(t,644)];         %% remove "-" if you have corrected the sign error in DTFs!
    Y1 = [asd zeros(t,1) conj(fliplr(asd(:,2:end)))];
    %Y1=asd;

    asd1 = [zeros(t,41) log(abs(-DTFs{g})) zeros(t,644)];         %% remove "-" if you have corrected the sign error in DTFs!
    asd2 = [zeros(t,41) log(conj(abs(-DTFs{g}))) zeros(t,644)];   %% remove "-" if you have corrected the sign error in DTFs!
    Y2 = [asd1 zeros(t,1) fliplr(asd2(:,2:end))];
    C = abs(Y1).*exp(i*imag(hilbert(abs(-Y2))));             %% Min Phase IR calculation

    for k = 1:t
        IR{g}(k,:) = real(ifft(Y1(k,:)));
%         IR{g}(k,:) = abs(ifft(Y1(k,:)));
        %temp=real(ifft(Y1(k,:)));
        IR_min_phase{g}(k,:) = real(ifft(C(k,:)));
    end;
    IR{g} = [IR{g}(:,1025:end) IR{g}(:,1:1024)];             % change this by removing phase delay at the begining
%     IR{g} = [IR{g}(:,1:1024)];
    IR_min_phase{g} = [IR_min_phase{g}(:,1025:end) IR_min_phase{g}(:,1:1024)];             % change this by removing phase delay at the begining
end;