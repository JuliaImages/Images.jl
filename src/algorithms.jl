using Base: axes1, tail
using OffsetArrays
import Statistics
using ImageMorphology: dilate!, erode!

# Entropy for grayscale (intensity) images
function _log(kind::Symbol)
    if kind == :shannon
        return log2
    elseif kind == :nat
        return log
    elseif kind == :hartley
        return log10
    else
        throw(ArgumentError("Invalid entropy unit. (:shannon, :nat or :hartley)"))
    end
end

"""
    entropy(logᵦ, img)
    entropy(img; [kind=:shannon])

Compute the entropy of a grayscale image defined as `-sum(p.*logᵦ(p))`.
The base β of the logarithm (a.k.a. entropy unit) is one of the following:

- `:shannon ` (log base 2, default), or use logᵦ = log2
- `:nat` (log base e), or use logᵦ = log
- `:hartley` (log base 10), or use logᵦ = log10
"""
entropy(img::AbstractArray; kind=:shannon) = entropy(_log(kind), img)
function entropy(logᵦ::Log, img) where Log<:Function
    hist = StatsBase.fit(Histogram, vec(img), nbins=256, closed=:right)
    counts = hist.weights
    p = counts / length(img)
    logp = logᵦ.(p)

    # take care of empty bins
    logp[Bool[isinf(v) for v in logp]] .= 0

    -sum(p .* logp)
end

function entropy(img::AbstractArray{Bool}; kind=:shannon)
    logᵦ = _log(kind)

    p = sum(img) / length(img)

    (0 < p < 1) ? - p*logᵦ(p) - (1-p)*logᵦ(1-p) : zero(p)
end

entropy(img::AbstractArray{C}; kind=:shannon) where {C<:AbstractGray} = entropy(channelview(img), kind=kind)

accum(::Type{T}) where {T<:Integer} = Int
accum(::Type{Float32})    = Float32
accum(::Type{T}) where {T<:Real} = Float64
accum(::Type{C}) where {C<:Colorant} = base_colorant_type(C){accum(eltype(C))}

graytype(::Type{T}) where {T<:Number} = T
graytype(::Type{C}) where {C<:AbstractGray} = C
graytype(::Type{C}) where {C<:Colorant} = Gray{eltype(C)}

# Simple image difference testing
macro test_approx_eq_sigma_eps(A, B, sigma, eps)
    quote
        if size($(esc(A))) != size($(esc(B)))
            error("Sizes ", size($(esc(A))), " and ",
                  size($(esc(B))), " do not match")
        end
        kern = KernelFactors.IIRGaussian($(esc(sigma)))
        Af = imfilter($(esc(A)), kern, NA())
        Bf = imfilter($(esc(B)), kern, NA())
        diffscale = max(_abs(maximum_finite(abs, $(esc(A)))), _abs(maximum_finite(abs, $(esc(B)))))
        d = sad(Af, Bf)
        if d > length(Af)*diffscale*($(esc(eps)))
            error("Arrays A and B differ")
        end
    end
end

# image difference testing (@tbreloff's, based on the macro)
#   A/B: images/arrays to compare
#   sigma: tuple of ints... how many pixels to blur
#   eps: error allowance
# returns: percentage difference on match, error otherwise
function test_approx_eq_sigma_eps(A::AbstractArray, B::AbstractArray,
                         sigma::AbstractVector{T} = ones(ndims(A)),
                         eps::AbstractFloat = 1e-2,
                         expand_arrays::Bool = true) where T<:Real
    if size(A) != size(B)
        if expand_arrays
            newsize = map(max, size(A), size(B))
            if size(A) != newsize
                A = copyto!(zeros(eltype(A), newsize...), A)
            end
            if size(B) != newsize
                B = copyto!(zeros(eltype(B), newsize...), B)
            end
        else
            error("Arrays differ: size(A): $(size(A)) size(B): $(size(B))")
        end
    end
    if length(sigma) != ndims(A)
        error("Invalid sigma in test_approx_eq_sigma_eps. Should be ndims(A)-length vector of the number of pixels to blur.  Got: $sigma")
    end
    kern = KernelFactors.IIRGaussian(sigma)
    Af = imfilter(A, kern, NA())
    Bf = imfilter(B, kern, NA())
    diffscale = max(_abs(maximum_finite(abs, A)), _abs(maximum_finite(abs, B)))
    d = sad(Af, Bf)
    diffpct = d / (length(Af) * diffscale)
    if diffpct > eps
        error("Arrays differ.  Difference: $diffpct  eps: $eps")
    end
    diffpct
end

# This should be removed when upstream ImageBase is updated
# In ImageBase v0.1.3: `maxabsfinite` returns a RGB instead of a Number
_abs(c::CT) where CT<:Color = mapreducec(abs, +, zero(eltype(CT)), c)
_abs(c::Number) = abs(c)


@inline function _clippedinds(Router,rstp)
    CartesianIndices(map((f,l)->f:l,
                         (first(Router)+rstp).I,(last(Router)-rstp).I))
end

function imgaussiannoise(img::AbstractArray{T}, variance::Number, mean::Number) where T
    return img + sqrt(variance)*randn(size(img)) + mean
end

imgaussiannoise(img::AbstractArray{T}, variance::Number) where {T} = imgaussiannoise(img, variance, 0)
imgaussiannoise(img::AbstractArray{T}) where {T} = imgaussiannoise(img, 0.01, 0)


# morphological operations for ImageMeta
dilate(img::ImageMeta, region=coords_spatial(img)) = shareproperties(img, dilate!(copy(arraydata(img)), region))
erode(img::ImageMeta, region=coords_spatial(img)) = shareproperties(img, erode!(copy(arraydata(img)), region))

"""
```
pyramid = gaussian_pyramid(img, n_scales, downsample, sigma)
```

Returns a  gaussian pyramid of scales `n_scales`, each downsampled
by a factor `downsample` > 1 and `sigma` for the gaussian kernel.

"""
function gaussian_pyramid(img::AbstractArray{T,N}, n_scales::Int, downsample::Real, sigma::Real) where {T,N}
    kerng = KernelFactors.IIRGaussian(sigma)
    kern = ntuple(d->kerng, Val(N))
    gaussian_pyramid(img, n_scales, downsample, kern)
end

function gaussian_pyramid(img::AbstractArray{T,N}, n_scales::Int, downsample::Real, kern::NTuple{N,Any}) where {T,N}
    downsample > 1 || @warn("downsample factor should be greater than one")
    # To guarantee inferability, we make sure that we do at least one
    # round of smoothing and resizing
    img_smoothed_main = imfilter(img, kern, NA())
    img_scaled = pyramid_scale(img_smoothed_main, downsample)
    prev = convert(typeof(img_scaled), img)
    pyramid = typeof(img_scaled)[prev]
    if n_scales ≥ 1
        # Take advantage of the work we've already done
        push!(pyramid, img_scaled)
        prev = img_scaled
    end
    for i in 2:n_scales
        img_smoothed = imfilter(prev, kern, NA())
        img_scaled = pyramid_scale(img_smoothed, downsample)
        push!(pyramid, img_scaled)
        prev = img_scaled
    end
    pyramid
end

function pyramid_scale(img, downsample)
    sz_next = map(s->ceil(Int, s/downsample), size(img))
    imresize(img, sz_next)
end

function pyramid_scale(img::OffsetArray, downsample)
    sz_next = map(s->ceil(Int, s/downsample), length.(axes(img)))
#    off = (.-ceil.(Int,(.-iterate.(axes(img).-(1,1))[1])./downsample))
    off = (.-ceil.(Int,(.-iterate.(map(x->UnitRange(x).-1,axes(img)))[1])./downsample))
    OffsetArray(imresize(img, sz_next), off)
end

"""
```
thres = otsu_threshold(img)
thres = otsu_threshold(img, bins)
```

Computes threshold for grayscale image using Otsu's method.

Parameters:
-    img         = Grayscale input image
-    bins        = Number of bins used to compute the histogram. Needed for floating-point images.

"""
function otsu_threshold(img::AbstractArray{T, N}, bins::Int = 256) where {T<:Union{Gray,Real}, N}

    min, max = extrema(img)
    edges, counts = imhist(img, range(gray(min), stop=gray(max), length=bins))
    histogram = counts./sum(counts)

    ω0 = 0
    μ0 = 0
    μt = 0
    μT = sum((1:(bins+1)).*histogram)
    max_σb=0.0
    thres=1

    for t in 1:bins
        ω0 += histogram[t]
        ω1 = 1 - ω0
        μt += t*histogram[t]

        σb = (μT*ω0-μt)^2/(ω0*ω1)

        if(σb > max_σb)
            max_σb = σb
            thres = t
        end
    end

    return T((edges[thres-1]+edges[thres])/2)
end

"""
```
thres = yen_threshold(img)
thres = yen_threshold(img, bins)
```

Computes threshold for grayscale image using Yen's maximum correlation criterion for bilevel thresholding

Parameters:
-    img         = Grayscale input image
-    bins        = Number of bins used to compute the histogram. Needed for floating-point images.


#Citation
Yen J.C., Chang F.J., and Chang S. (1995) “A New Criterion for Automatic Multilevel Thresholding” IEEE Trans. on Image Processing, 4(3): 370-378. DOI:10.1109/83.366472
"""
function yen_threshold(img::AbstractArray{T, N}, bins::Int = 256) where {T<:Union{Gray, Real}, N}

    min, max = extrema(img)
    if(min == max)
        return T(min)
    end

    edges, counts = imhist(img, range(gray(min), stop=gray(max), length=bins))

    prob_mass_function = counts./sum(counts)
    clamp!(prob_mass_function,eps(),Inf)
    prob_mass_function_sq = prob_mass_function.^2
    cum_pmf = cumsum(prob_mass_function)
    cum_pmf_sq_1 = cumsum(prob_mass_function_sq)
    cum_pmf_sq_2 = reverse!(cumsum(reverse!(prob_mass_function_sq)))

    #Equation (4) cited in the paper.
    criterion = log.(((cum_pmf[1:end-1].*(1.0 .- cum_pmf[1:end-1])).^2) ./ (cum_pmf_sq_1[1:end-1].*cum_pmf_sq_2[2:end]))

    thres = edges[findmax(criterion)[2]]
    return T(thres)

end
