function imcorner(img::AbstractArray, method::AbstractString="harris", border::AbstractString="replicate", blockSize::Int=3, k::Float64=0.04)
    # Performes corner detection using the Harris method or the Shi-Tomasi method

    (grad_x,grad_y) = imgradients(img, "sobel",border);

    cov_xx = grad_x .* grad_x;
    cov_xy = grad_x .* grad_y;
    cov_yy = grad_y .* grad_y;


    box_filter_kernel = (1/(blockSize*blockSize)) * ones(blockSize,blockSize);

    cov_xx = imfilter(cov_xx,box_filter_kernel);
    cov_xy = imfilter(cov_xy,box_filter_kernel);
    cov_yy = imfilter(cov_yy,box_filter_kernel);

    if method == "harris"
        corner = (cov_xx.*cov_yy - cov_xy.*cov_xy - k*(cov_xx + cov_yy).*(cov_xx + cov_yy));
    elseif method == "shi-tomasi"
        cov_yy = 0.5*cov_yy;
        cov_xx = 0.5*cov_xx;
        corner = ((cov_xx + cov_yy) - sqrt((cov_xx - cov_yy).*(cov_xx - cov_yy) + cov_xy.*cov_xy));
    else
        error("Unknown corner method: $method")
    end

    return corner
end