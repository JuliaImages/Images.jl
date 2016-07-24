"""
```
corners = imcorner(img, threshold, percentile; [method])
```

Performs corner detection using one of the following methods - 

    1. harris
    2. shi_tomasi
    3. kitchen_rosenfeld

The parameters of the individual methods are described in their documentation. If
a threshold is specified, the values of the responses are thresholded to give the 
corner pixels. The threshold is assumed to be a percentile value unless `percentile`
is set to false. 
"""
function imcorner(img::AbstractArray, threshold = 0.99, percentile = true; method::Function = harris, args...)
    img_gray = convert(Array{Gray}, img)
    responses = method(img_gray; args...)
    
    if percentile == true
        threshold = StatsBase.percentile(responses[:], threshold * 100)
    end

    corners = map(i -> i < threshold ? zero(Gray{U8}) : one(Gray{U8}), responses)
    corners
end

"""
```
harris_response = harris(img; [k], [border], [weights])
```

Performs Harris corner detection. The covariances can be taken using either a mean
weighted filter or a gamma kernel.
"""
function harris(img::AbstractArray; k::Float64 = 0.04, args...)
    cov_xx, cov_xy, cov_yy = gradcovs(img, args...)
    corner = map((xx, yy, xy) -> xx * yy - xy ^ 2 - k * (xx + yy) ^ 2, cov_xx, cov_yy, cov_xy)
    corner
end

"""
```
shi_tomasi_response = shi_tomasi(img; [border], [weights])
```

Performs Shi Tomasi corner detection. The covariances can be taken using either a mean
weighted filter or a gamma kernel.
"""
function shi_tomasi(img::AbstractArray; border::AbstractString = "replicate", args...)
    cov_xx, cov_xy, cov_yy = gradcovs(img, border; args...)
    corner = map((xx, yy, xy) -> ((xx + yy) - (sqrt((xx - yy) ^ 2 + 4 * xy ^ 2))) / 2, cov_xx, cov_yy, cov_xy)
    corner
end

"""
```
kitchen_rosenfeld_response = kitchen_rosenfeld(img; [border])
```

Performs Kitchen Rosenfeld corner detection. The covariances can be taken using either a mean
weighted filter or a gamma kernel.
"""
function kitchen_rosenfeld(img::AbstractArray; border::AbstractString = "replicate")
    (grad_x, grad_y) = imgradients(img, "sobel", border)
    (grad_xx, grad_xy) = imgradients(grad_x, "sobel", border)
    (grad_yx, grad_yy) = imgradients(grad_y, "sobel", border)
    numerator = map((x, y, xx, xy, yy) -> xx * (y ^ 2) + yy * (x ^ 2) - 2 * xy * x * y, grad_x, grad_y, grad_xx, grad_xy, grad_yy)
    denominator = map((x, y) -> x ^ 2 + y ^ 2, grad_x, grad_y)
    corner = map((n, d) -> d == 0.0 ? 0.0 : n/d, numerator, denominator)
    corner
end

function gradcovs(img::AbstractArray, border::AbstractString = "replicate"; weights::Function = meancovs, args...)
    (grad_x, grad_y) = imgradients(img, "sobel", border)

    cov_xx = grad_x .* grad_x
    cov_xy = grad_x .* grad_y
    cov_yy = grad_y .* grad_y

    weights(cov_xx, cov_xy, cov_yy, args...)
end

function meancovs(cov_xx, cov_xy, cov_yy, blockSize::Int = 3)

    box_filter_kernel = (1 / (blockSize * blockSize)) * ones(blockSize, blockSize)

    filt_cov_xx = imfilter(cov_xx, box_filter_kernel)
    filt_cov_xy = imfilter(cov_xy, box_filter_kernel)
    filt_cov_yy = imfilter(cov_yy, box_filter_kernel)

    filt_cov_xx, filt_cov_xy, filt_cov_yy
end

function gammacovs(cov_xx, cov_xy, cov_yy, gamma::Float64 = 1.4)

    filt_cov_xx = imfilter_gaussian(cov_xx, [gamma, gamma])
    filt_cov_xy = imfilter_gaussian(cov_xy, [gamma, gamma])
    filt_cov_yy = imfilter_gaussian(cov_yy, [gamma, gamma])

    filt_cov_xx, filt_cov_xy, filt_cov_yy
end