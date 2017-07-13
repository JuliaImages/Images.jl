"""
  `imadjustintensity(img [, (minval,maxval)]) -> Image`

   Map intensities over the interval `(minval,maxval)` to the interval
   `[0,1]`. This is equivalent to `map(ScaleMinMax(eltype(img), minval,
   maxval), img)`.  (minval,maxval) defaults to `extrema(img)`.
"""
imadjustintensity{T}(img::AbstractArray{T}, range::Tuple{Any,Any}) = map(scaleminmax(T, range...), img)
imadjustintensity(img::AbstractArray, range::AbstractArray) = imadjustintensity(img, (range...,))
imadjustintensity(img::AbstractArray) = map(takemap(scaleminmax, img), img)

_imstretch{T}(img::AbstractArray{T}, m::Number, slope::Number) = map(i -> 1 / (1 + (m / (i + eps(T))) ^ slope), img)

"""
`imgs = imstretch(img, m, slope)` enhances or reduces (for
slope > 1 or < 1, respectively) the contrast near saturation (0 and 1). This is
essentially a symmetric gamma-correction. For a pixel of brightness `p`, the new
intensity is `1/(1+(m/(p+eps))^slope)`.

This assumes the input `img` has intensities between 0 and 1.
"""
imstretch(img::AbstractArray, m::Number, slope::Number) = _imstretch(float(img), m, slope)
imstretch(img::ImageMeta, m::Number, slope::Number) = shareproperties(img, imstretch(data(img), m, slope))

"""
    y = complement(x)

Take the complement `1-x` of `x`.  If `x` is a color with an alpha channel,
the alpha channel is left untouched.
"""
complement(x) = one(x)-x
complement(x::TransparentColor) = typeof(x)(complement(color(x)), alpha(x))

imhist{T<:Colorant}(img::AbstractArray{T}, nbins::Integer = 400) = imhist(convert(Array{Gray}, img), nbins)

function imhist{T<:NumberLike}(img::AbstractArray{T}, nbins::Integer = 400)
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
function imhist(img::AbstractArray, nbins::Integer, minval::RealLike, maxval::RealLike)
    edges = StatsBase.histrange([Float64(minval), Float64(maxval)], nbins, :left)
    imhist(img, edges)
end

function imhist(img::AbstractArray, edges::Range)
    histogram = zeros(Int, length(edges) + 1)
    o = Base.Order.Forward
    G = graytype(eltype(img))
    for v in img
        val = convert(G, v)
        if val >= edges[end]
            histogram[end] += 1
            continue
        end
        index = searchsortedlast(edges, val, o)
        histogram[index + 1] += 1
    end
    edges, histogram
end

function _histeq_pixel_rescale{T<:NumberLike}(pixel::T, cdf, minval, maxval)
    n = length(cdf)
    bin_pixel = clamp(ceil(Int, (pixel - minval) * length(cdf) / (maxval - minval)), 1, n)
    rescaled_pixel = minval + ((cdf[bin_pixel] - cdf[1]) * (maxval - minval) / (cdf[end] - cdf[1]))
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
function histeq(img::AbstractArray, nbins::Integer, minval::RealLike, maxval::RealLike)
    bins, histogram = imhist(img, nbins, minval, maxval)
    cdf = cumsum(histogram[2:end-1])
    img_shape = size(img)
    minval == maxval && return map(identity, img)
    # Would like to use `map` here, but see https://github.com/timholy/Images.jl/pull/523#issuecomment-235236460
    hist_equalised_img = similar(img)
    for I in eachindex(img)
        hist_equalised_img[I] = _histeq_pixel_rescale(img[I], cdf, minval, maxval)
    end
    hist_equalised_img
end

function histeq(img::AbstractArray, nbins::Integer)
    T = graytype(eltype(img))
    histeq(img, nbins, zero(T), one(T))
end

function histeq(img::ImageMeta, nbins::Integer, minval::RealLike, maxval::RealLike)
    newimg = histeq(data(img), nbins, minval, maxval)
    shareproperties(img, newimg)
end

histeq(img::ImageMeta, nbins::Integer) = shareproperties(img, histeq(data(img), nbins))

adjust_gamma(img::ImageMeta, gamma::Number) = shareproperties(img, adjust_gamma(data(img), gamma))

_gamma_pixel_rescale{T<:NumberLike}(pixel::T, gamma::Number) = pixel ^ gamma

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
function adjust_gamma{T<:FixedPointNumbers.Normed}(img::AbstractArray{Gray{T}}, gamma::Number)
    raw_type = FixedPointNumbers.rawtype(T)
    gamma_inv = 1.0 / gamma
    table = zeros(T, typemax(raw_type) + 1)
    for i in zero(raw_type):typemax(raw_type)
        table[i + 1] = T((i / typemax(raw_type)) ^ gamma_inv)
    end
    gamma_corrected_img = similar(img)
    for I in eachindex(img)
        gamma_corrected_img[I] = Gray(table[convert(base_colorant_type(typeof(img[I])){T}, img[I]).val.i + 1])
    end
    gamma_corrected_img
end

adjust_gamma{T<:Number}(img::AbstractArray{T}, gamma::Number) = _adjust_gamma(img, gamma, Float64)
adjust_gamma{T<:Colorant}(img::AbstractArray{T}, gamma::Number) = _adjust_gamma(img, gamma, T)

function _adjust_gamma(img::AbstractArray, gamma::Number, C::Type)
    gamma_corrected_img = zeros(C, size(img))
    for I in eachindex(img)
        gamma_corrected_img[I] = _gamma_pixel_rescale(img[I], gamma)
    end
    gamma_corrected_img
end

function adjust_gamma{T<:Number}(img::AbstractArray{T}, gamma::Number, minval::Number, maxval::Number)
    gamma_corrected_img = zeros(Float64, size(img))
    for I in eachindex(img)
        gamma_corrected_img[I] = _gamma_pixel_rescale(img[I], gamma, minval, maxval)
    end
    gamma_corrected_img
end

"""
```
hist_matched_img = histmatch(img, oimg, nbins)
```

Returns a grayscale histogram matched image with a granularity of `nbins` number of bins. `img` is the image to be
matched and `oimg` is the image having the desired histogram to be matched to.

"""
histmatch(img::ImageMeta, oimg::AbstractArray, nbins::Integer = 400) = shareproperties(img, histmatch(data(img), oimg, nbins))

_hist_match_pixel{T<:NumberLike}(pixel::T, bins, lookup_table) = T(bins[lookup_table[searchsortedlast(bins, pixel)]])

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
    ohist[1] = zero(eltype(ohist))
    ohist[end] = zero(eltype(ohist))
    histogram[1] = zero(eltype(histogram))
    histogram[end] = zero(eltype(histogram))
    cdf = cumsum(histogram)
    norm_cdf = cdf / cdf[end]
    ocdf = cumsum(ohist)
    norm_ocdf = ocdf / ocdf[end]
    lookup_table = zeros(Int, length(norm_cdf))
    for I in eachindex(cdf)
        lookup_table[I] = indmin(abs.(norm_ocdf .- norm_cdf[I]))
    end
    hist_matched_img = similar(img)
    for I in eachindex(img)
        hist_matched_img[I] = _hist_match_pixel(img[I], bins, lookup_table)
    end
    hist_matched_img
end


"""
```
hist_equalised_img = clahe(img, nbins, xblocks = 8, yblocks = 8, clip = 3)

```

Performs Contrast Limited Adaptive Histogram Equalisation (CLAHE) on the input image. It differs from ordinary histogram
equalization in the respect that the adaptive method computes several histograms, each corresponding to a distinct section
of the image, and uses them to redistribute the lightness values of the image. It is therefore suitable for improving the
local contrast and enhancing the definitions of edges in each region of an image.

In the straightforward form, CLAHE is done by calculation a histogram of a window around each pixel and using the transformation
function of the equalised histogram to rescale the pixel. Since this is computationally expensive, we use interpolation which gives
a significant rise in efficiency without compromising the result. The image is divided into a grid and equalised histograms are
calculated for each block. Then, each pixel is interpolated using the closest histograms.

The `xblocks` and `yblocks` specify the number of blocks to divide the input image into in each direction. `nbins` specifies
the granularity of histogram calculation of each local region. `clip` specifies the value at which the histogram is clipped.
The excess in the histogram bins with value exceeding `clip` is redistributed among the other bins.

"""
function clahe{C}(img::AbstractArray{C, 2}, nbins::Integer = 100; xblocks::Integer = 8, yblocks::Integer = 8, clip::Number = 3)
    h, w = size(img)
    y_padded = ceil(Int, h / (2 * yblocks)) * 2 * yblocks
    x_padded = ceil(Int, w / (2 * xblocks)) * 2 * xblocks

    img_padded = imresize(img, (y_padded, x_padded))

    hist_equalised_img = _clahe(img_padded, nbins, xblocks, yblocks, clip)
    out = similar(img)
    ImageTransformations.imresize!(out, hist_equalised_img)
end

function clahe(img::ImageMeta, nbins::Integer = 100; xblocks::Integer = 8, yblocks::Integer = 8, clip::Number = 3)
    shareproperties(clahe(data(img), nbins, xblocks = xblocks, yblocks = yblocks, clip = clip), img)
end

function _clahe{C}(img::AbstractArray{C, 2}, nbins::Integer = 100, xblocks::Integer = 8, yblocks::Integer = 8, clip::Number = 3)
    h, w = size(img)
    xb = 0:1:xblocks - 1
    yb = 0:1:yblocks - 1
    blockw = Int(w / xblocks)
    blockh = Int(h / yblocks)
    temp_cdf = Array{Float64, 1}[]
    T = graytype(eltype(img))
    edges = StatsBase.histrange([Float64(zero(T)), Float64(one(T))], nbins, :left)

    for i in xb
        for j in yb
            temp_block = img[j * blockh + 1 : (j + 1) * blockh, i * blockw + 1 : (i + 1) * blockw]
            _, histogram = imhist(temp_block, edges)
            clipped_hist = cliphist(histogram[2:end - 1], clip)
            cdf = cumsum(clipped_hist)
            n_cdf = cdf / max(convert(eltype(cdf), 1), cdf[end])
            push!(temp_cdf, n_cdf)
        end
    end

    norm_cdf = reshape(temp_cdf, yblocks, xblocks)
    res_img = zeros(C, size(img))

    #Interpolations
    xb = 1:2:xblocks * 2 - 2
    yb = 1:2:yblocks * 2 - 2

    for j in yb
        for i in xb
            p_block = img[j * Int(blockh / 2) + 1 : (j + 2) * Int(blockh / 2), i * Int(blockw / 2) + 1 : (i + 2) * Int(blockw / 2)]
            bnum_y = floor(Int, (j + 1) / 2)
            bnum_x = floor(Int, (i + 1) / 2)
            top_left = norm_cdf[bnum_y, bnum_x]
            top_right = norm_cdf[bnum_y, bnum_x + 1]
            bot_left = norm_cdf[bnum_y + 1, bnum_x]
            bot_right = norm_cdf[bnum_y + 1, bnum_x + 1]
            temp_block = zeros(C, size(p_block))
            for l in 1:blockw
                for m in 1:blockh
                    temp_block[m, l] = _clahe_pixel_rescale(p_block[m, l], top_left, top_right, bot_left, bot_right, edges, l, m, blockw, blockh)
                end
            end
            res_img[j * Int(blockh / 2) + 1 : (j + 2) * Int(blockh / 2), i * Int(blockw / 2) + 1 : (i + 2) * Int(blockw / 2)] = temp_block
        end
    end

    #Corners

    block_tl = img[1 : Int(blockh / 2), 1 : Int(blockw / 2)]
    corner_tl = map(i -> _clahe_pixel_rescale(i, norm_cdf[1, 1], edges), block_tl)
    res_img[1 : Int(blockh / 2), 1 : Int(blockw / 2)] = corner_tl

    block_tr = img[1 : Int(blockh / 2), (xblocks * 2 - 1) * Int(blockw / 2) + 1 : (xblocks * 2) * Int(blockw / 2)]
    corner_tr = map(i -> _clahe_pixel_rescale(i, norm_cdf[1, xblocks], edges), block_tr)
    res_img[1 : Int(blockh / 2), (xblocks * 2 - 1) * Int(blockw / 2) + 1 : (xblocks * 2) * Int(blockw / 2)] = corner_tr

    block_bl = img[(yblocks * 2 - 1) * Int(blockh / 2) + 1 : (yblocks * 2) * Int(blockh / 2), 1 : Int(blockw / 2)]
    corner_bl = map(i -> _clahe_pixel_rescale(i, norm_cdf[yblocks, 1], edges), block_bl)
    res_img[(yblocks * 2 - 1) * Int(blockh / 2) + 1 : (yblocks * 2) * Int(blockh / 2), 1 : Int(blockw / 2)] = corner_bl

    block_br = img[(yblocks * 2 - 1) * Int(blockh / 2) + 1 : (yblocks * 2) * Int(blockh / 2), (xblocks * 2 - 1) * Int(blockw / 2) + 1 : (xblocks * 2) * Int(blockw / 2)]
    corner_br = map(i -> _clahe_pixel_rescale(i, norm_cdf[yblocks, xblocks], edges), block_br)
    res_img[(yblocks * 2 - 1) * Int(blockh / 2) + 1 : (yblocks * 2) * Int(blockh / 2), (xblocks * 2 - 1) * Int(blockw / 2) + 1 : (xblocks * 2) * Int(blockw / 2)] = corner_br

    #Horizontal Borders

    for j in [0, yblocks * 2 - 1]
        for i in xb
            p_block = img[j * Int(blockh / 2) + 1 : (j + 1) * Int(blockh / 2), i * Int(blockw / 2) + 1 : (i + 2) * Int(blockw / 2)]
            bnum_x = floor(Int, (i + 1) / 2)
            left = norm_cdf[ceil(Int, (j + 1) / 2), bnum_x]
            right = norm_cdf[ceil(Int, (j + 1) / 2), bnum_x + 1]
            temp_block = zeros(C, size(p_block))
            for l in 1:blockw
                for m in 1:Int(blockh / 2)
                    temp_block[m, l] = _clahe_pixel_rescale(p_block[m, l], left, right, edges, l, blockw)
                end
            end
            res_img[j * Int(blockh / 2) + 1 : (j + 1) * Int(blockh / 2), i * Int(blockw / 2) + 1 : (i + 2) * Int(blockw / 2)] = temp_block
        end
    end

    #Vertical Borders

    for i in [0, xblocks * 2 - 1]
        for j in yb
            p_block = img[j * Int(blockh / 2) + 1 : (j + 2) * Int(blockh / 2), i * Int(blockw / 2) + 1 : (i + 1) * Int(blockw / 2)]
            bnum_y = floor(Int, (j + 1) / 2)
            top = norm_cdf[bnum_y, ceil(Int, (i + 1) / 2)]
            bot = norm_cdf[bnum_y + 1, ceil(Int, (i + 1) / 2)]
            temp_block = zeros(C, size(p_block))
            for l in 1:Int(blockw / 2)
                for m in 1:blockh
                    temp_block[m, l] = _clahe_pixel_rescale(p_block[m, l], top, bot, edges, m, blockh)
                end
            end
            res_img[j * Int(blockh / 2) + 1 : (j + 2) * Int(blockh / 2), i * Int(blockw / 2) + 1 : (i + 1) * Int(blockw / 2)] = temp_block
        end
    end
    res_img
end

_clahe_pixel_rescale{T<:NumberLike}(pixel::T, cdf, edges) = cdf[searchsortedlast(edges, pixel, Base.Order.Forward)]

function _clahe_pixel_rescale{T<:NumberLike}(pixel::T, first, second, edges, pos, length)
    id = searchsortedlast(edges, pixel, Base.Order.Forward)
    f = first[id]
    s = second[id]
    T(((length - pos) * f + (pos - 1) * s) / (length - 1))
end

function _clahe_pixel_rescale{T<:NumberLike}(pixel::T, top_left, top_right, bot_left, bot_right, edges, i, j, w, h)
    id = searchsortedlast(edges, pixel, Base.Order.Forward)
    tl = top_left[id]
    tr = top_right[id]
    bl = bot_left[id]
    br = bot_right[id]
    r1 = ((w - i) * tl + (i - 1) * tr) / (w - 1)
    r2 = ((w - i) * bl + (i - 1) * br) / (w - 1)
    T(((h - j) * r1 + (j - 1) * r2) / (h - 1))
end

function _clahe_pixel_rescale{C<:Color}(pixel::C, args...)
    yiq = convert(YIQ, pixel)
    y = _clahe_pixel_rescale(yiq.y, args...)
    convert(C, YIQ(y, yiq.i, yiq.q))
end

function _clahe_pixel_rescale{C<:TransparentColor}(pixel::C, args...)
    base_colorant_type(C)(_clahe_pixel_rescale(color(pixel), args...), alpha(pixel))
end

"""
```
clipped_hist = cliphist(hist, clip)
```

Clips the histogram above a certain value `clip`. The excess left in the bins
exceeding `clip` is redistributed among the remaining bins.
"""
function cliphist{T}(hist::AbstractArray{T, 1}, clip::Number)
    hist_length = length(hist)
    excess = sum(map(i -> i > clip ? i - clip : zero(i - clip), hist))
    increase = excess / hist_length
    clipped_hist = zeros(Float64, hist_length)
    removed_sum = zero(Float64)

    for i in 1:hist_length
        if hist[i] > clip - increase
            clipped_hist[i] = Float64(clip)
            if hist[i] < clip
                removed_sum += hist[i] - (clip - increase)
            end
        else
            clipped_hist[i] = hist[i] + Float64(increase)
            removed_sum += increase
        end
    end

    leftover = excess - removed_sum

    while true
        oleftover = leftover
        for i in 1:hist_length
            leftover <= 0 && break
            step = ceil(Int, max(1, hist_length / leftover))
            for h in i:step:hist_length
                leftover <= 0 && break
                diff = clip - clipped_hist[h]
                if diff > 1
                    clipped_hist[h] += 1
                    leftover -= 1
                elseif diff > 0
                    clipped_hist[h] = clip
                    leftover -= diff
                end
            end
        end
        (leftover <= 0 || leftover >= oleftover) && break
    end
    clipped_hist
end
