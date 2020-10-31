function [HRTFl_interp, HRTFr_interp] = HRTF_gauss( HRTFl, HRTFr, azim, elev ,d_azim, d_elev )

    distance = ((azim-d_azim).^2+(elev-d_elev).^2);
    sigma = 2.5;
    weights = exp(-distance/(2*sigma^2));
    HRTFl_interp = weights'*HRTFl/sum(weights);
    HRTFr_interp = weights'*HRTFr/sum(weights);
end



