function canny{T<:NumberLike}(img_gray::AbstractMatrix{T}, sigma::Number = 1.4, upperThreshold::Number = 0.90, lowerThreshold::Number = 0.10; percentile::Bool = true)
    depwarn("canny(img, sigma, $upperThreshold, $lowerThreshold; percentile=$percentile) is deprecated.\n Please use canny(img, ($upperThreshold, $lowerThreshold), sigma) or canny(img, (Percentile($(100*upperThreshold)), Percentile($(100*lowerThreshold))), sigma)",:canny)
    if percentile
        canny(img_gray, (Percentile(100*upperThreshold), Percentile(100*lowerThreshold)), sigma)
    else
        canny(img_gray, (upperThreshold, lowerThreshold), sigma)
    end
end

function imcorner(img::AbstractArray, threshold, percentile;
                  method::Function = harris, args...)
    if percentile == true # NB old function didn't require Bool, this ensures conversion
        depwarn("imcorner(img, $threshold, true; ...) is deprecated. Please use imcorner(img, Percentile($(100*threshold)); ...) instead.", :imcorner)
        imcorner(img, Percentile(100*threshold); method=method, args...)
    else
        depwarn("imcorner(img, $threshold, false; ...) is deprecated. Please use imcorner(img, $threshold; ...) instead.", :imcorner)
        imcorner(img, threshold; method=method, args...)
    end
end

function imedge(img::AbstractArray, method::AbstractString, border::AbstractString="replicate")
    f = ImageFiltering.kernelfunc_lookup(method)
    depwarn("`imedge(img, \"$method\", args...)` is deprecated, please use `imedge(img, $f, args...)` instead.", :imedge)
    imedge(img, f, border)
end
