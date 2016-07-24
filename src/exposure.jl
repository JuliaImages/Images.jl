"""
  `imadjustintensity(img [, (minval,maxval)]) -> Image`

   Map intensities over the interval `(minval,maxval)` to the interval
   `[0,1]`. This is equivalent to `map(ScaleMinMax(eltype(img), minval,
   maxval), img)`.  (minval,maxval) defaults to `extrema(img)`.
"""
imadjustintensity{T}(img::AbstractArray{T}, range) = map(ScaleMinMax(T, range...), img)
imadjustintensity{T}(img::AbstractArray{T}) = map(ScaleAutoMinMax(T), img)

function _imstretch{T}(img::AbstractArray{T}, m::Number, slope::Number)
    shareproperties(img, 1./(1 + (m./(data(img) .+ eps(T))).^slope))
end

"""
`imgs = imstretch(img, m, slope)` enhances or reduces (for
slope > 1 or < 1, respectively) the contrast near saturation (0 and 1). This is
essentially a symmetric gamma-correction. For a pixel of brightness `p`, the new
intensity is `1/(1+(m/(p+eps))^slope)`.

This assumes the input `img` has intensities between 0 and 1.
"""
imstretch(img::AbstractArray, m::Number, slope::Number) = _imstretch(float(img), m, slope)

function imcomplement{T}(img::AbstractArray{T})
    reshape([complement(x) for x in img], size(img))
end
imcomplement(img::AbstractImage) = copyproperties(img, imcomplement(data(img)))
complement(x) = one(x)-x
complement(x::TransparentColor) = typeof(x)(complement(color(x)), alpha(x))

imhist{T<:Colorant}(img::AbstractArray{T}, nbins::Integer = 400) = imhist(convert(Array{Gray}, data(img)), nbins)

function imhist{T<:Union{Gray,Number}}(img::AbstractArray{T}, nbins::Integer = 400)
    minval = minfinite(img)
    maxval = maxfinite(img)
    imhist(img, nbins, minval, maxval)
end

"""
```
edges, count = imhist(img, nbins)
edges, count = imhist(img, nbins, minval, maxval)
```

Generates a histogram for the image over nbins spread between `(minval, maxval]`.
If `minval` and `maxval` are not given, then the minimum and
maximum values present in the image are taken.

`edges` is a vector that specifies how the range is divided;
`count[i+1]` is the number of values `x` that satisfy `edges[i] <= x < edges[i+1]`.
`count[1]` is the number satisfying `x < edges[1]`, and
`count[end]` is the number satisfying `x >= edges[end]`. Consequently,
`length(count) == length(edges)+1`.
"""
function imhist(img::AbstractArray, nbins::Integer, minval::Union{Gray,Real}, maxval::Union{Gray,Real})
    edges = StatsBase.histrange([Float64(minval), Float64(maxval)], nbins, :left)
    imhist(img, edges)
end

function imhist(img::AbstractArray, edges::Range)
    histogram = zeros(Int, length(edges)+1)
    o = Base.Order.Forward
    G = graytype(eltype(img))
    for v in img
        val = convert(G, v)
        if val>=edges[end]
            histogram[end] += 1
            continue
        end
        index = searchsortedlast(edges, val, o)
        histogram[index+1] += 1
    end
    edges, histogram
end

function _histeq_pixel_rescale{T<:Union{Gray,Number}}(pixel::T, cdf, minval, maxval)
    n = length(cdf)
    bin_pixel = clamp(ceil(Int, (pixel-minval)*length(cdf)/(maxval-minval)), 1, n)
    rescaled_pixel = minval + ((cdf[bin_pixel]-cdf[1])*(maxval-minval)/(cdf[end]-cdf[1]))
    convert(T, rescaled_pixel)
end
function _histeq_pixel_rescale{C<:Color}(pixel::C, cdf, minval, maxval)
    yiq = convert(YIQ, pixel)
    y = _histeq_pixel_rescale(yiq.y, cdf, minval, maxval)
    convert(C, YIQ(y, yiq.i, yiq.q))
end
function _histeq_pixel_rescale{C<:TransparentColor}(pixel::C, cdf, minval, maxval)
    base_colorant_type(C)(_histeq_pixel_rescale(color(pixel), cdf, minval, maxval), alpha(pixel))
end

"""
```
hist_equalised_img = histeq(img, nbins)
hist_equalised_img = histeq(img, nbins, minval, maxval)
```

Returns a histogram equalised image with a granularity of approximately `nbins`
number of bins.

The `histeq` function can handle a variety of input types. The returned image depends
on the input type. If the input is an `Image` then the resulting image is of the same type
and has the same properties.

For coloured images, the input is converted to YIQ type and the Y channel is equalised. This
is the combined with the I and Q channels and the resulting image converted to the same type
as the input.

If minval and maxval are specified then intensities are equalized to the range
(minval, maxval). The default values are 0 and 1.

"""
function histeq(img::AbstractArray, nbins::Integer, minval::Union{Number,Gray}, maxval::Union{Number,Gray})
    bins, histogram = imhist(img, nbins, minval, maxval)
    cdf = cumsum(histogram[2:end-1])
    img_shape = size(img)
    minval == maxval && return map(identity, img)
    hist_equalised_img = map(p -> _histeq_pixel_rescale(p, cdf, minval, maxval), img)
    hist_equalised_img
end

function histeq(img::AbstractArray, nbins::Integer)
    T = graytype(eltype(img))
    histeq(img, nbins, zero(T), one(T))
end

function histeq(img::AbstractImage, nbins::Integer, minval::Union{Number,Gray}, maxval::Union{Number,Gray})
    newimg = histeq(data(img), nbins, minval, maxval)
    shareproperties(img, newimg)
end

histeq(img::AbstractImage, nbins::Integer) = shareproperties(img, histeq(data(img), nbins))

adjust_gamma(img::AbstractImage, gamma::Number) = shareproperties(img, adjust_gamma(data(img), gamma))

_gamma_pixel_rescale{T<:Union{Gray, Number}}(pixel::T, gamma::Number) = pixel ^ gamma

function _gamma_pixel_rescale{C<:Color}(pixel::C, gamma::Number)
    yiq = convert(YIQ, pixel)
    y = _gamma_pixel_rescale(yiq.y, gamma)
    convert(C, YIQ(y, yiq.i, yiq.q))
end

function _gamma_pixel_rescale{C<:TransparentColor}(pixel::C, gamma::Number)
    base_colorant_type(C)(_gamma_pixel_rescale(color(pixel), gamma), alpha(pixel))
end

function _gamma_pixel_rescale(original_val::Number, gamma::Number, minval::Number, maxval::Number)
    Float32(minval + (maxval - minval) * ((original_val - minval) / (maxval - minval)) ^ gamma)
end

"""
```
gamma_corrected_img = adjust_gamma(img, gamma)
```

Returns a gamma corrected image. 

The `adjust_gamma` function can handle a variety of input types. The returned image depends 
on the input type. If the input is an `Image` then the resulting image is of the same type
and has the same properties. 

For coloured images, the input is converted to YIQ type and the Y channel is gamma corrected. 
This is the combined with the I and Q channels and the resulting image converted to the same 
type as the input.

"""
adjust_gamma(img::AbstractArray, gamma::Number) = map(i -> _gamma_pixel_rescale(i, gamma), img)

function adjust_gamma{T<:FixedPointNumbers.UFixed}(img::AbstractArray{Gray{T}}, gamma::Number)
    raw_type = FixedPointNumbers.rawtype(T)
    gamma_inv = 1.0 / gamma
    table = [T((i / typemax(raw_type)) ^ gamma_inv) for i in zero(raw_type):typemax(raw_type)]
    map(x -> Gray(table[convert(base_colorant_type(typeof(x)){T}, x).val.i + 1]), img)
end

adjust_gamma{T<:Number}(img::AbstractArray{T}, gamma::Number, minval::Number, maxval::Number) = map(i -> _gamma_pixel_rescale(i, gamma, minval, maxval), img)

"""
```
hist_matched_img = histmatch(img, oimg, nbins)
```

Returns a grayscale histogram matched image with a granularity of `nbins` number of bins. `img` is the image to be 
matched and `oimg` is the image having the desired histogram to be matched to. 

"""
histmatch(img::AbstractImage, oimg::AbstractArray, nbins::Integer = 400) = shareproperties(img, histmatch(data(img), oimg, nbins))

_hist_match_pixel{T<:Union{Gray, Number}}(pixel::T, bins, lookup_table) = T(bins[lookup_table[searchsortedlast(bins, pixel)]])

function _hist_match_pixel{T<:Color}(pixel::T, bins, lookup_table)
    yiq = convert(YIQ, pixel)
    y = _hist_match_pixel(yiq.y, bins, lookup_table)
    convert(T, YIQ(y, yiq.i, yiq.q))
end

_hist_match_pixel{T<:TransparentColor}(pixel::T, bins, lookup_table) = base_colorant_type(T)(_hist_match_pixel(color(pixel), bins, lookup_table), alpha(pixel))

function histmatch{T<:Colorant}(img::AbstractArray{T}, oimg::AbstractArray, nbins::Integer = 400)
    el_gray = graytype(eltype(img))
    oedges, ohist = imhist(oimg, nbins, zero(el_gray), one(el_gray))
    _histmatch(img, oedges, ohist)
end

function _histmatch(img::AbstractArray, oedges::Range, ohist::AbstractArray{Int})
    bins, histogram = imhist(img, oedges)
    ohist[1] = 0
    ohist[end] = 0
    histogram[1] = 0 
    histogram[end] = 0
    cdf = cumsum(histogram)
    cdf /= cdf[end]
    ocdf = cumsum(ohist)
    ocdf /= ocdf[end]
    lookup_table = [indmin(abs(ocdf-val)) for val in cdf]
    map(i -> _hist_match_pixel(i, bins, lookup_table), img)
end

