"""
  `imadjustintensity(img [, (minval,maxval)]) -> Image`

   Map intensities over the interval `(minval,maxval)` to the interval
   `[0,1]`. This is equivalent to `map(ScaleMinMax(eltype(img), minval,
   maxval), img)`.  (minval,maxval) defaults to `extrema(img)`.
"""
imadjustintensity(img::AbstractArray{T}, range::Tuple{Any,Any}) where {T} = map(scaleminmax(T, range...), img)
imadjustintensity(img::AbstractArray, range::AbstractArray) = imadjustintensity(img, (range...,))
imadjustintensity(img::AbstractArray) = map(takemap(scaleminmax, img), img)

_imstretch(img::AbstractArray{T}, m::Number, slope::Number) where {T} = map(i -> 1 / (1 + (m / (i + eps(T))) ^ slope), img)

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
the alpha channel is left untouched. Don't forget to add a dot when `x` is
an array: `complement.(x)`
"""
complement(x::Union{Number,Colorant}) = oneunit(x)-x
complement(x::TransparentColor) = typeof(x)(complement(color(x)), alpha(x))
@deprecate complement(x::AbstractArray) complement.(x)

imhist(img::AbstractArray{T}, nbins::Integer = 400) where {T<:Colorant} = imhist(convert(Array{Gray}, img), nbins)

function imhist(img::AbstractArray{T}, nbins::Integer = 400) where T<:NumberLike
    minval = minfinite(img)
    maxval = maxfinite(img)
    imhist(img, nbins, minval, maxval)
end


"""
```
edges, count = imhist(img, nbins)
edges, count = imhist(img, nbins, minval, maxval)
edges, count = imhist(img, edges)
```

Generates a histogram for the image over nbins spread between `(minval, maxval]`.
Color images are automatically converted to grayscale.

# Output
Returns `edges` which is a [`range`](@ref) type that specifies how the  interval
`(minval, maxval]` is divided into bins, and an array `count` which records the
concomitant bin frequencies. In particular, `count` has the following
properties:

* `count[i+1]` is the number of values `x` that satisfy `edges[i] <= x < edges[i+1]`.
* `count[1]` is the number satisfying `x < edges[1]`, and
* `count[end]` is the number satisfying `x >= edges[end]`.
* `length(count) == length(edges)+1`.

# Details

One can consider a histogram as a piecewise-constant model of a probability
density function ``f`` [1]. Suppose that ``f`` has support on some interval ``I =
[a,b]``.  Let ``m`` be an integer and ``a = a_1 < a_2 < \\ldots < a_m < a_{m+1} =
b`` a sequence of real numbers. Construct a sequence of intervals

```math
I_1 = [a_1,a_2], I_2 = (a_2, a_3], \\ldots, I_{m} = (a_m,a_{m+1}]
```

which partition ``I`` into subsets ``I_j`` ``(j = 1, \\ldots, m)`` on which
``f`` is constant. These subsets satisfy ``I_i \\cap I_j = \\emptyset, \\forall
i \\neq j``, and are commonly referred to as *bins*. Together they encompass the
entire range of data values such that ``\\sum_j |I_j | = | I |``. Each bin has
width ``w_j = |I_j| = a_{j+1} - a_j`` and height ``h_j`` which is the constant
probability density over the region of the bin. Integrating the constant
probability density over the width of the bin ``w_j`` yields a probability mass
of ``\\pi_j = h_j w_j`` for the bin.

For a sample ``x_1, x_2, \\ldots, x_N``, let


```math
n_j = \\sum_{n = 1}^{N}\\mathbf{1}_{(I_j)}(x_n),
\\quad \\text{where} \\quad
\\mathbf{1}_{(I_j)}(x) =
\\begin{cases}
 1 & \\text{if} x \\in I_j,\\\\
 0 & \\text{otherwise},
\\end{cases},
```
represents the number of samples falling into the interval ``I_j``. An estimate
for the probability mass of the ``j``th bin is given by the relative frequency
``\\hat{\\pi} = \\frac{n_j}{N}``, and the histogram estimator of the probability
density function is defined as
```math
\\begin{aligned}
\\hat{f}_n(x)  & = \\sum_{j = 1}^{m}\\frac{n_j}{Nw_j} \\mathbf{1}_{(I_j)}(x) \\\\
& = \\sum_{j = 1}^{m}\\frac{\\hat{\\pi}_j}{w_j} \\mathbf{1}_{(I_j)}(x) \\\\
& = \\sum_{j = 1}^{m}\\hat{h}_j \\mathbf{1}_{(I_j)}(x).
\\end{aligned}
```

The function ``\\hat{f}_n(x)`` is a genuine density estimator because ``\\hat{f}_n(x)  \\ge 0`` and
```math
\\begin{aligned}
\\int_{-\\infty}^{\\infty}\\hat{f}_n(x) \\operatorname{d}x & = \\sum_{j=1}^{m} \\frac{n_j}{Nw_j} w_j \\\\
& = 1.
\\end{aligned}
```

# Options
Various options for the parameters of this function are described in more detail
below.

## Choices for `nbins`
You can specify the number of discrete bins for the histogram.

## Choices for `minval`
You have the option to specify the lower bound of the interval over which the
histogram will be computed.  If `minval` is not specified then the minimum
value present in the image is taken as the lower bound.

## Choices for `maxval`
You have the option to specify the upper bound of the interval over which the
histogram will be computed.  If `maxval` is not specified then the maximum
value present in the image is taken as the upper bound.

## Choices for `edges`
If you do not designate the number of bins, nor the lower or upper bound of the
interval, then you have the option to directly stipulate how the intervals will
be divided by specifying a [`range`](@ref) type.

# Example

Compute the histogram of a grayscale image.
```julia

using TestImages, FileIO, ImageView

img =  testimage("mandril_gray");
edges, counts  = imhist(img,256);
```

Given a color image, compute the hisogram of the red channel.
```julia
img = testimage("mandrill")
r = red(img)
edges, counts  = imhist(r,256);
```

# References
[1] E. Herrholz, "Parsimonious Histograms," Ph.D. dissertation, Inst. of Math. and Comp. Sci., University of Greifswald, Greifswald, Germany, 2011.
"""
function imhist(img::AbstractArray, nbins::Integer, minval::RealLike, maxval::RealLike)
    edges = StatsBase.histrange([Float64(minval), Float64(maxval)], nbins, :left)
    imhist(img, edges)
end

function imhist(img::AbstractArray, edges::AbstractRange)
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

function _histeq_pixel_rescale(pixel::T, cdf, minval, maxval) where T<:NumberLike
    n = length(cdf)
    bin_pixel = clamp(ceil(Int, (pixel - minval) * length(cdf) / (maxval - minval)), 1, n)
    rescaled_pixel = minval + ((cdf[bin_pixel] - cdf[1]) * (maxval - minval) / (cdf[end] - cdf[1]))
    convert(T, rescaled_pixel)
end
function _histeq_pixel_rescale(pixel::C, cdf, minval, maxval) where C<:Color
    yiq = convert(YIQ, pixel)
    y = _histeq_pixel_rescale(yiq.y, cdf, minval, maxval)
    convert(C, YIQ(y, yiq.i, yiq.q))
end
function _histeq_pixel_rescale(pixel::C, cdf, minval, maxval) where C<:TransparentColor
    base_colorant_type(C)(_histeq_pixel_rescale(color(pixel), cdf, minval, maxval), alpha(pixel))
end

"""
```
hist_equalised_img = histeq(img, nbins)
hist_equalised_img = histeq(img, nbins, minval, maxval)
```

Returns a histogram equalised image with a granularity of approximately `nbins`
number of bins.

# Details

Histogram equalisation was initially conceived to  improve the contrast in a
single-channel grayscale image. The method transforms the
distribution of the intensities in an image so that they are as uniform as
possible [1]. The natural justification for uniformity
is that the image has better contrast  if the intensity levels of an image span
a wide range on the intensity scale. As it turns out, the necessary
transformation is a mapping based on the cumulative histogram.

One can consider an ``L``-bit single-channel ``I \\times J`` image with gray
values in the set ``\\{0,1,\\ldots,L-1 \\}``, as a collection of independent and
identically distributed random variables. Specifically, let the sample space
``\\Omega`` be the set of all ``IJ``-tuples ``\\omega
=(\\omega_{11},\\omega_{12},\\ldots,\\omega_{1J},\\omega_{21},\\omega_{22},\\ldots,\\omega_{2J},\\omega_{I1},\\omega_{I2},\\ldots,\\omega_{IJ})``,
where each ``\\omega_{ij} \\in \\{0,1,\\ldots, L-1 \\}``. Furthermore, impose a
probability measure on ``\\Omega`` such that the functions ``\\Omega \\ni
\\omega \\to \\omega_{ij} \\in \\{0,1,\\ldots,L-1\\}`` are independent and
identically distributed.

One can then regard an image as a matrix of random variables ``\\mathbf{G} =
[G_{i,j}(\\omega)]``, where each function ``G_{i,j}: \\Omega \\to \\mathbb{R}``
is defined by
```math
G_{i,j}(\\omega) = \\frac{\\omega_{ij}}{L-1},
```
and each ``G_{i,j}`` is distributed according to some unknown density ``f_{G}``.
While ``f_{G}`` is unknown, one can approximate it with a normalised histogram
of gray levels,

```math
\\hat{f}_{G}(v)= \\frac{n_v}{IJ},
```
where
```math
n_v = \\left | \\left\\{(i,j)\\, |\\,  G_{i,j}(\\omega)  = v \\right \\} \\right |
```
represents the number of times a gray level with intensity ``v`` occurs in
``\\mathbf{G}``. To transforming the distribution of the intensities so that
they are as uniform as possible one needs to find a mapping ``T(\\cdot)`` such
that ``T(G_{i,j}) \\thicksim U ``. The required mapping turns out to be the
cumulative distribution function (CDF) of the empirical density
``\\hat{f}_{G}``,
```math
 T(G_{i,j}) = \\int_0^{G_{i,j}}\\hat{f}_{G}(w)\\mathrm{d} w.
```

# Options

Various options for the parameters of this function are described in more detail
below.

## Choices for `img`

The `histeq` function can handle a variety of input types. The returned image
depends on the input type. If the input is an `Image` then the resulting image
is of the same type and has the same properties.

For coloured images, the input is converted to
[YIQ](https://en.wikipedia.org/wiki/YIQ) type and the Y channel is equalised.
This is the combined with the I and Q channels and the resulting image converted
to the same type as the input.

## Choices for `nbins`

You can specify the total number of bins in the histogram.

## Choices for `minval` and `maxval`

If minval and maxval are specified then intensities are equalized to the range
(minval, maxval). The default values are 0 and 1.

# Example

```julia

using TestImages, FileIO, ImageView

img =  testimage("mandril_gray");
imgeq = histeq(img,256);

imshow(img)
imshow(imgeq)
```

# References
1. R. C. Gonzalez and R. E. Woods. *Digital Image Processing (3rd Edition)*.  Upper Saddle River, NJ, USA: Prentice-Hall,  2006.

See also: [histmatch](@ref),[clahe](@ref), [imhist](@ref) and  [adjust_gamma](@ref).

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
    histeq(img, nbins, zero(T), oneunit(T))
end

function histeq(img::ImageMeta, nbins::Integer, minval::RealLike, maxval::RealLike)
    newimg = histeq(data(img), nbins, minval, maxval)
    shareproperties(img, newimg)
end

histeq(img::ImageMeta, nbins::Integer) = shareproperties(img, histeq(data(img), nbins))

adjust_gamma(img::ImageMeta, gamma::Number) = shareproperties(img, adjust_gamma(data(img), gamma))

_gamma_pixel_rescale(pixel::T, gamma::Number) where {T<:NumberLike} = pixel ^ gamma

function _gamma_pixel_rescale(pixel::C, gamma::Number) where C<:Color
    yiq = convert(YIQ, pixel)
    y = _gamma_pixel_rescale(yiq.y, gamma)
    convert(C, YIQ(y, yiq.i, yiq.q))
end

function _gamma_pixel_rescale(pixel::C, gamma::Number) where C<:TransparentColor
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
function adjust_gamma(img::AbstractArray{Gray{T}}, gamma::Number) where T<:FixedPointNumbers.Normed
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

adjust_gamma(img::AbstractArray{T}, gamma::Number) where {T<:Number} = _adjust_gamma(img, gamma, Float64)
adjust_gamma(img::AbstractArray{T}, gamma::Number) where {T<:Colorant} = _adjust_gamma(img, gamma, T)

function _adjust_gamma(img::AbstractArray, gamma::Number, C::Type)
    gamma_corrected_img = _fill(oneunit(C), axes(img))
    for I in eachindex(img)
        gamma_corrected_img[I] = _gamma_pixel_rescale(img[I], gamma)
    end
    gamma_corrected_img
end

function adjust_gamma(img::AbstractArray{T}, gamma::Number, minval::Number, maxval::Number) where T<:Number
    gamma_corrected_img = _fill(oneunit(T), axes(img))
    for I in eachindex(img)
        gamma_corrected_img[I] = _gamma_pixel_rescale(img[I], gamma, minval, maxval)
    end
    gamma_corrected_img
end

_fill(val, dim) = fill(val, dim) # fallback
_fill(val, dim::NTuple{N,Base.OneTo}) where {N} = fill(val, map(length, dim))

"""
```
hist_matched_img = histmatch(img, oimg, nbins)
```

Returns a grayscale histogram matched image with a granularity of `nbins` number of bins. `img` is the image to be
matched and `oimg` is the image having the desired histogram to be matched to.

"""
histmatch(img::ImageMeta, oimg::AbstractArray, nbins::Integer = 400) = shareproperties(img, histmatch(data(img), oimg, nbins))

_hist_match_pixel(pixel::T, bins, lookup_table) where {T<:NumberLike} = T(bins[lookup_table[searchsortedlast(bins, pixel)]])

function _hist_match_pixel(pixel::T, bins, lookup_table) where T<:Color
    yiq = convert(YIQ, pixel)
    y = _hist_match_pixel(yiq.y, bins, lookup_table)
    convert(T, YIQ(y, yiq.i, yiq.q))
end

_hist_match_pixel(pixel::T, bins, lookup_table) where {T<:TransparentColor} = base_colorant_type(T)(_hist_match_pixel(color(pixel), bins, lookup_table), alpha(pixel))

function histmatch(img::AbstractArray{T}, oimg::AbstractArray, nbins::Integer = 400) where T<:Colorant
    el_gray = graytype(eltype(img))
    oedges, ohist = imhist(oimg, nbins, zero(el_gray), oneunit(el_gray))
    _histmatch(img, oedges, ohist)
end

function _histmatch(img::AbstractArray, oedges::AbstractRange, ohist::AbstractArray{Int})
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
        lookup_table[I] = argmin(abs.(norm_ocdf .- norm_cdf[I]))
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
function clahe(img::AbstractArray{C, 2}, nbins::Integer = 100; xblocks::Integer = 8, yblocks::Integer = 8, clip::Number = 3) where C
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

function _clahe(img::AbstractArray{C, 2}, nbins::Integer = 100, xblocks::Integer = 8, yblocks::Integer = 8, clip::Number = 3) where C
    h, w = size(img)
    xb = 0:1:xblocks - 1
    yb = 0:1:yblocks - 1
    blockw = Int(w / xblocks)
    blockh = Int(h / yblocks)
    temp_cdf = Array{Float64, 1}[]
    T = graytype(eltype(img))
    edges = StatsBase.histrange([Float64(zero(T)), Float64(oneunit(T))], nbins, :left)

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

_clahe_pixel_rescale(pixel::T, cdf, edges) where {T<:NumberLike} = cdf[searchsortedlast(edges, pixel, Base.Order.Forward)]

function _clahe_pixel_rescale(pixel::T, first, second, edges, pos, length) where T<:NumberLike
    id = searchsortedlast(edges, pixel, Base.Order.Forward)
    f = first[id]
    s = second[id]
    T(((length - pos) * f + (pos - 1) * s) / (length - 1))
end

function _clahe_pixel_rescale(pixel::T, top_left, top_right, bot_left, bot_right, edges, i, j, w, h) where T<:NumberLike
    id = searchsortedlast(edges, pixel, Base.Order.Forward)
    tl = top_left[id]
    tr = top_right[id]
    bl = bot_left[id]
    br = bot_right[id]
    r1 = ((w - i) * tl + (i - 1) * tr) / (w - 1)
    r2 = ((w - i) * bl + (i - 1) * br) / (w - 1)
    T(((h - j) * r1 + (j - 1) * r2) / (h - 1))
end

function _clahe_pixel_rescale(pixel::C, args...) where C<:Color
    yiq = convert(YIQ, pixel)
    y = _clahe_pixel_rescale(yiq.y, args...)
    convert(C, YIQ(y, yiq.i, yiq.q))
end

function _clahe_pixel_rescale(pixel::C, args...) where C<:TransparentColor
    base_colorant_type(C)(_clahe_pixel_rescale(color(pixel), args...), alpha(pixel))
end

"""
```
clipped_hist = cliphist(hist, clip)
```

Clips the histogram above a certain value `clip`. The excess left in the bins
exceeding `clip` is redistributed among the remaining bins.
"""
function cliphist(hist::AbstractArray{T, 1}, clip::Number) where T
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
