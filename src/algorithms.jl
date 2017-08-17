using Base: indices1, tail

Compat.@dep_vectorize_2arg Gray atan2
Compat.@dep_vectorize_2arg Gray hypot

"""
`M = meanfinite(img, region)` calculates the mean value along the dimensions listed in `region`, ignoring any non-finite values.
"""
meanfinite{T<:Real}(A::AbstractArray{T}, region) = _meanfinite(A, T, region)
meanfinite{CT<:Colorant}(A::AbstractArray{CT}, region) = _meanfinite(A, eltype(CT), region)
function _meanfinite{T<:AbstractFloat}(A::AbstractArray, ::Type{T}, region)
    inds = Base.reduced_indices(A, region)
    K = similar(Array{Int}, inds)
    S = similar(Array{eltype(A)}, inds)
    fill!(K, 0)
    fill!(S, zero(eltype(A)))
    sumfinite!(S, K, A)
    S./K
end
_meanfinite(A::AbstractArray, ::Type, region) = mean(A, region)  # non floating-point

using Base: check_reducedims, reducedim1, safe_tail
using Base.Broadcast: newindex

"""
    sumfinite!(S, K, A)

Compute the sum `S` and number of contributing pixels `K` for
reductions of the array `A` over dimensions. `S` and `K` must have
identical indices, and either match `A` or have singleton-dimensions for
the dimensions that are being summed over. Only pixels with finite
value are included in the tallies of `S` and `K`.

Note that the pixel mean is just S./K.
"""
function sumfinite!{T,N}(S, K, A::AbstractArray{T,N})
    check_reducedims(S, A)
    isempty(A) && return S, K
    indices(S) == indices(K) || throw(DimensionMismatch("S and K must have identical indices"))

    indsAt, indsSt = safe_tail(indices(A)), safe_tail(indices(S))
    keep, Idefault = _newindexer(indsAt, indsSt)
    if reducedim1(S, A)
        # keep the accumulators as a local variable when reducing along the first dimension
        i1 = first(indices1(S))
        @inbounds for IA in CartesianRange(indsAt)
            IS = newindex(IA, keep, Idefault)
            s, k = S[i1,IS], K[i1,IS]
            for i in indices(A, 1)
                tmp = A[i, IA]
                if isfinite(tmp)
                    s += tmp
                    k += 1
                end
            end
            S[i1,IS], K[i1,IS] = s, k
        end
    else
        @inbounds for IA in CartesianRange(indsAt)
            IS = newindex(IA, keep, Idefault)
            for i in indices(A, 1)
                tmp = A[i, IA]
                if isfinite(tmp)
                    S[i, IS] += tmp
                    K[i, IS] += 1
                end
            end
        end
    end
    S, K
end
if VERSION < v"0.6.0-dev.693"
    _newindexer(shape, inds) = Base.Broadcast.newindexer(shape, inds)
else
    _newindexer(shape, inds) = Base.Broadcast.shapeindexer(shape, inds)
end

function Base.var{C<:AbstractGray}(A::AbstractArray{C}; kwargs...)
    imgc = channelview(A)
    base_colorant_type(C)(var(imgc; kwargs...))
end

function Base.var{C<:Colorant,N}(A::AbstractArray{C,N}; kwargs...)
    imgc = channelview(A)
    colons = ntuple(d->Colon(), Val{N})
    inds1 = indices(imgc, 1)
    val1 = var(view(imgc, first(inds1), colons...); kwargs...)
    vals = similar(imgc, typeof(val1), inds1)
    vals[1] = val1
    for i in first(inds1)+1:last(inds1)
        vals[i] = var(view(imgc, i, colons...); kwargs...)
    end
    base_colorant_type(C)(vals...)
end

Base.std{C<:Colorant}(A::AbstractArray{C}; kwargs...) = mapc(sqrt, var(A; kwargs...))

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
function entropy{Log<:Function}(logᵦ::Log, img)
    hist = StatsBase.fit(Histogram, vec(img), nbins=256, closed=:right)
    counts = hist.weights
    p = counts / length(img)
    logp = logᵦ.(p)

    # take care of empty bins
    logp[Bool[isinf(v) for v in logp]] = 0

    -sum(p .* logp)
end

function entropy(img::AbstractArray{Bool}; kind=:shannon)
    logᵦ = _log(kind)

    p = sum(img) / length(img)

    (0 < p < 1) ? - p*logᵦ(p) - (1-p)*logᵦ(1-p) : zero(p)
end

entropy{C<:AbstractGray}(img::AbstractArray{C}; kind=:shannon) = entropy(channelview(img), kind=kind)

# functions red, green, and blue
for (funcname, fieldname) in ((:red, :r), (:green, :g), (:blue, :b))
    fieldchar = string(fieldname)[1]
    @eval begin
        function $funcname{CV<:Color}(img::AbstractArray{CV})
            T = eltype(CV)
            out = Array(T, size(img))
            for i = 1:length(img)
                out[i] = convert(RGB{T}, img[i]).$fieldname
            end
            out
        end

        function $funcname(img::AbstractArray)
            pos = search(lowercase(colorspace(img)), $fieldchar)
            pos == 0 && error("channel $fieldchar not found in colorspace $(colorspace(img))")
            sliceim(img, "color", pos)
        end
    end
end

"`r = red(img)` extracts the red channel from an RGB image `img`" red
"`g = green(img)` extracts the green channel from an RGB image `img`" green
"`b = blue(img)` extracts the blue channel from an RGB image `img`" blue

"""
`m = minfinite(A)` calculates the minimum value in `A`, ignoring any values that are not finite (Inf or NaN).
"""
function minfinite{T}(A::AbstractArray{T})
    ret = sentinel_min(T)
    for a in A
        ret = minfinite_scalar(a, ret)
    end
    ret
end
function minfinite(f, A::AbstractArray)
    ret = sentinel_min(typeof(f(first(A))))
    for a in A
        ret = minfinite_scalar(f(a), ret)
    end
    ret
end

"""
`m = maxfinite(A)` calculates the maximum value in `A`, ignoring any values that are not finite (Inf or NaN).
"""
function maxfinite{T}(A::AbstractArray{T})
    ret = sentinel_max(T)
    for a in A
        ret = maxfinite_scalar(a, ret)
    end
    ret
end
function maxfinite(f, A::AbstractArray)
    ret = sentinel_max(typeof(f(first(A))))
    for a in A
        ret = maxfinite_scalar(f(a), ret)
    end
    ret
end

"""
`m = maxabsfinite(A)` calculates the maximum absolute value in `A`, ignoring any values that are not finite (Inf or NaN).
"""
function maxabsfinite{T}(A::AbstractArray{T})
    ret = sentinel_min(typeof(abs(A[1])))
    for a in A
        ret = maxfinite_scalar(abs(a), ret)
    end
    ret
end

minfinite_scalar{T}(a::T, b::T) = isfinite(a) ? (b < a ? b : a) : b
maxfinite_scalar{T}(a::T, b::T) = isfinite(a) ? (b > a ? b : a) : b
minfinite_scalar{T<:Union{Integer,FixedPoint}}(a::T, b::T) = b < a ? b : a
maxfinite_scalar{T<:Union{Integer,FixedPoint}}(a::T, b::T) = b > a ? b : a
minfinite_scalar(a, b) = minfinite_scalar(promote(a, b)...)
maxfinite_scalar(a, b) = maxfinite_scalar(promote(a, b)...)

function minfinite_scalar{C<:AbstractRGB}(c1::C, c2::C)
    C(minfinite_scalar(c1.r, c2.r),
      minfinite_scalar(c1.g, c2.g),
      minfinite_scalar(c1.b, c2.b))
end
function maxfinite_scalar{C<:AbstractRGB}(c1::C, c2::C)
    C(maxfinite_scalar(c1.r, c2.r),
      maxfinite_scalar(c1.g, c2.g),
      maxfinite_scalar(c1.b, c2.b))
end

sentinel_min{T<:Union{Integer,FixedPoint}}(::Type{T}) = typemax(T)
sentinel_max{T<:Union{Integer,FixedPoint}}(::Type{T}) = typemin(T)
sentinel_min{T<:AbstractFloat}(::Type{T}) = convert(T, NaN)
sentinel_max{T<:AbstractFloat}(::Type{T}) = convert(T, NaN)
sentinel_min{C<:AbstractRGB}(::Type{C}) = _sentinel_min(C, eltype(C))
_sentinel_min{C<:AbstractRGB,T}(::Type{C},::Type{T}) = (s = sentinel_min(T); C(s,s,s))
sentinel_max{C<:AbstractRGB}(::Type{C}) = _sentinel_max(C, eltype(C))
_sentinel_max{C<:AbstractRGB,T}(::Type{C},::Type{T}) = (s = sentinel_max(T); C(s,s,s))
sentinel_min{C<:AbstractGray}(::Type{C}) = _sentinel_min(C, eltype(C))
_sentinel_min{C<:AbstractGray,T}(::Type{C},::Type{T}) = C(sentinel_min(T))
sentinel_max{C<:AbstractGray}(::Type{C}) = _sentinel_max(C, eltype(C))
_sentinel_max{C<:AbstractGray,T}(::Type{C},::Type{T}) = C(sentinel_max(T))


# FIXME: replace with IntegralImage
# average filter
"""
`kern = imaverage(filtersize)` constructs a boxcar-filter of the specified size.
"""
function imaverage(filter_size=[3,3])
    if length(filter_size) != 2
        error("wrong filter size")
    end
    m, n = filter_size
    if mod(m, 2) != 1 || mod(n, 2) != 1
        error("filter dimensions must be odd")
    end
    f = ones(Float64, m, n)/(m*n)
end

# FIXME: do something about this
# more general version
function imlaplacian(alpha::Number)
    lc = alpha/(1 + alpha)
    lb = (1 - alpha)/(1 + alpha)
    lm = -4/(1 + alpha)
    return [lc lb lc; lb lm lb; lc lb lc]
end

function sumdiff(f, A::AbstractArray, B::AbstractArray)
    indices(A) == indices(B) || throw(DimensionMismatch("A and B must have the same indices"))
    T = promote_type(difftype(eltype(A)), difftype(eltype(B)))
    s = zero(accum(eltype(T)))
    for (a, b) in zip(A, B)
        x = convert(T, a) - convert(T, b)
        s += f(x)
    end
    s
end

"`s = ssd(A, B)` computes the sum-of-squared differences over arrays/images A and B"
ssd(A::AbstractArray, B::AbstractArray) = sumdiff(abs2, A, B)

"`s = sad(A, B)` computes the sum-of-absolute differences over arrays/images A and B"
sad(A::AbstractArray, B::AbstractArray) = sumdiff(abs, A, B)

difftype{T<:Integer}(::Type{T}) = Int
difftype{T<:Real}(::Type{T}) = Float32
difftype(::Type{Float64}) = Float64
difftype{CV<:Colorant}(::Type{CV}) = difftype(CV, eltype(CV))
difftype{CV<:RGBA,T<:Real}(::Type{CV}, ::Type{T}) = RGBA{Float32}
difftype{CV<:RGBA}(::Type{CV}, ::Type{Float64}) = RGBA{Float64}
difftype{CV<:BGRA,T<:Real}(::Type{CV}, ::Type{T}) = BGRA{Float32}
difftype{CV<:BGRA}(::Type{CV}, ::Type{Float64}) = BGRA{Float64}
difftype{CV<:AbstractGray,T<:Real}(::Type{CV}, ::Type{T}) = Gray{Float32}
difftype{CV<:AbstractGray}(::Type{CV}, ::Type{Float64}) = Gray{Float64}
difftype{CV<:AbstractRGB,T<:Real}(::Type{CV}, ::Type{T}) = RGB{Float32}
difftype{CV<:AbstractRGB}(::Type{CV}, ::Type{Float64}) = RGB{Float64}

accum{T<:Integer}(::Type{T}) = Int
accum(::Type{Float32})    = Float32
accum{T<:Real}(::Type{T}) = Float64
accum{C<:Colorant}(::Type{C}) = base_colorant_type(C){accum(eltype(C))}

graytype{T<:Number}(::Type{T}) = T
graytype{C<:AbstractGray}(::Type{C}) = C
graytype{C<:Colorant}(::Type{C}) = Gray{eltype(C)}

# normalized by Array size
"`s = ssdn(A, B)` computes the sum-of-squared differences over arrays/images A and B, normalized by array size"
ssdn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = ssd(A, B)/length(A)

# normalized by Array size
"`s = sadn(A, B)` computes the sum-of-absolute differences over arrays/images A and B, normalized by array size"
sadn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sad(A, B)/length(A)

# normalized cross correlation
"""
`C = ncc(A, B)` computes the normalized cross-correlation of `A` and `B`.
"""
function ncc{T}(A::AbstractArray{T}, B::AbstractArray{T})
    Am = (A.-mean(A))[:]
    Bm = (B.-mean(B))[:]
    return dot(Am,Bm)/(norm(Am)*norm(Bm))
end

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
        diffscale = max(maxabsfinite($(esc(A))), maxabsfinite($(esc(B))))
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
function test_approx_eq_sigma_eps{T<:Real}(A::AbstractArray, B::AbstractArray,
                                  sigma::AbstractVector{T} = ones(ndims(A)),
                                  eps::AbstractFloat = 1e-2,
                                  expand_arrays::Bool = true)
    if size(A) != size(B)
        if expand_arrays
            newsize = map(max, size(A), size(B))
            if size(A) != newsize
                A = copy!(zeros(eltype(A), newsize...), A)
            end
            if size(B) != newsize
                B = copy!(zeros(eltype(B), newsize...), B)
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
    diffscale = max(maxabsfinite(A), maxabsfinite(B))
    d = sad(Af, Bf)
    diffpct = d / (length(Af) * diffscale)
    if diffpct > eps
        error("Arrays differ.  Difference: $diffpct  eps: $eps")
    end
    diffpct
end

"""
BlobLoG stores information about the location of peaks as discovered by `blob_LoG`.
It has fields:

- location: the location of a peak in the filtered image (a CartesianIndex)
- σ: the value of σ which lead to the largest `-LoG`-filtered amplitude at this location
- amplitude: the value of the `-LoG(σ)`-filtered image at the peak

Note that the radius is equal to σ√2.

See also: [`blob_LoG`](@ref).
"""
immutable BlobLoG{T,S,N}
    location::CartesianIndex{N}
    σ::S
    amplitude::T
end

"""
    blob_LoG(img, σscales, [edges], [σshape]) -> Vector{BlobLoG}

Find "blobs" in an N-D image using the negative Lapacian of Gaussians
with the specifed vector or tuple of σ values. The algorithm searches for places
where the filtered image (for a particular σ) is at a peak compared to all
spatially- and σ-adjacent voxels, where σ is `σscales[i] * σshape` for some i.
By default, `σshape` is an ntuple of 1s.

The optional `edges` argument controls whether peaks on the edges are
included. `edges` can be `true` or `false`, or a N+1-tuple in which
the first entry controls whether edge-σ values are eligible to serve
as peaks, and the remaining N entries control each of the N dimensions
of `img`.

# Citation:

Lindeberg T (1998), "Feature Detection with Automatic Scale Selection",
International Journal of Computer Vision, 30(2), 79–116.

See also: [`BlobLoG`](@ref).
"""
function blob_LoG{T,N}(img::AbstractArray{T,N}, σscales::Union{AbstractVector,Tuple},
                       edges::Tuple{Vararg{Bool}}=(true, ntuple(d->false, Val{N})...), σshape=ntuple(d->1, Val{N}))
    sigmas = sort(σscales)
    img_LoG = Array{Float64}(length(sigmas), size(img)...)
    colons = ntuple(d->Colon(), Val{N})
    @inbounds for isigma in eachindex(sigmas)
        img_LoG[isigma,colons...] = (-sigmas[isigma]) * imfilter(img, Kernel.LoG(ntuple(i->sigmas[isigma]*σshape[i],Val{N})))
    end
    maxima = findlocalmaxima(img_LoG, 1:ndims(img_LoG), edges)
    [BlobLoG(CartesianIndex(tail(x.I)), sigmas[x[1]], img_LoG[x]) for x in maxima]
end
blob_LoG{T,N}(img::AbstractArray{T,N}, σscales, edges::Bool, σshape=ntuple(d->1, Val{N})) =
    blob_LoG(img, σscales, (edges, ntuple(d->edges,Val{N})...), σshape)

blob_LoG{T,N}(img::AbstractArray{T,N}, σscales, σshape=ntuple(d->1, Val{N})) =
    blob_LoG(img, σscales, (true, ntuple(d->false,Val{N})...), σshape)

findlocalextrema{T,N}(img::AbstractArray{T,N}, region, edges::Bool, order) = findlocalextrema(img, region, ntuple(d->edges,Val{N}), order)

function findlocalextrema{T<:Union{Gray,Number},N}(img::AbstractArray{T,N}, region::Union{Tuple{Int,Vararg{Int}},Vector{Int},UnitRange{Int},Int}, edges::NTuple{N,Bool}, order::Base.Order.Ordering)
    issubset(region,1:ndims(img)) || throw(ArgumentError("invalid region"))
    extrema = Array{CartesianIndex{N}}(0)
    edgeoffset = CartesianIndex(map(!, edges))
    R0 = CartesianRange(indices(img))
    R = CartesianRange(R0.start+edgeoffset, R0.stop-edgeoffset)
    Rinterior = CartesianRange(R0.start+1, R0.stop-1)
    iregion = CartesianIndex(ntuple(d->d∈region, Val{N}))
    Rregion = CartesianRange(-iregion, iregion)
    z = zero(iregion)
    for i in R
        isextrema = true
        img_i = img[i]
        if i ∈ Rinterior
            # If i is in the interior, we don't have to worry about i+j being out-of-bounds
            for j in Rregion
                j == z && continue
                if !Base.Order.lt(order, img[i+j], img_i)
                    isextrema = false
                    break
                end
            end
        else
            for j in Rregion
                (j == z || i+j ∉ R0) && continue
                if !Base.Order.lt(order, img[i+j], img_i)
                    isextrema = false
                    break
                end
            end
        end
        isextrema && push!(extrema, i)
    end
    extrema
end

"""
`findlocalmaxima(img, [region, edges]) -> Vector{Tuple}`

Returns the coordinates of elements whose value is larger than all of
their immediate neighbors.  `region` is a list of dimensions to
consider.  `edges` is a boolean specifying whether to include the
first and last elements of each dimension, or a tuple-of-Bool
specifying edge behavior for each dimension separately.
"""
findlocalmaxima(img::AbstractArray, region=coords_spatial(img), edges=true) =
        findlocalextrema(img, region, edges, Base.Order.Forward)

"""
Like `findlocalmaxima`, but returns the coordinates of the smallest elements.
"""
findlocalminima(img::AbstractArray, region=coords_spatial(img), edges=true) =
        findlocalextrema(img, region, edges, Base.Order.Reverse)

# restrict for AxisArray and ImageMeta
restrict(img::AxisArray, ::Tuple{}) = img
restrict(img::ImageMeta, ::Tuple{}) = img

function restrict(img::ImageMeta, region::Dims)
    shareproperties(img, restrict(data(img), region))
end

function restrict{T,N}(img::AxisArray{T,N}, region::Dims)
    inregion = falses(ndims(img))
    inregion[[region...]] = true
    inregiont = (inregion...,)::NTuple{N,Bool}
    AxisArray(restrict(img.data, region), map(modax, axes(img), inregiont))
end

# FIXME: this doesn't get inferred, but it should be (see issue #628)
function restrict{Ax}(img::Union{AxisArray,ImageMetaAxis}, ::Type{Ax})
    A = restrict(img.data, axisdim(img, Ax))
    AxisArray(A, replace_axis(modax(img[Ax]), axes(img)))
end

replace_axis(newax, axs) = _replace_axis(newax, axnametype(newax), axs...)
@inline _replace_axis{Ax}(newax, ::Type{Ax}, ax::Ax, axs...) = (newax, _replace_axis(newax, Ax, axs...)...)
@inline _replace_axis{Ax}(newax, ::Type{Ax}, ax, axs...) = (ax, _replace_axis(newax, Ax, axs...)...)
_replace_axis{Ax}(newax, ::Type{Ax}) = ()

axnametype{name}(ax::Axis{name}) = Axis{name}

function modax(ax)
    v = ax.val
    veven = linspace(v[1]-step(v)/2, v[end]+step(v)/2, length(v)÷2 + 1)
    return ax(isodd(length(v)) ? oftype(veven, v[1:2:end]) : veven)
end

modax(ax, inregion::Bool) = inregion ? modax(ax) : ax


function imlineardiffusion{T}(img::Array{T,2}, dt::AbstractFloat, iterations::Integer)
    u = img
    f = imlaplacian()
    for i = dt:dt:dt*iterations
        u = u + dt*imfilter(u, f, "replicate")
    end
    u
end

function imgaussiannoise{T}(img::AbstractArray{T}, variance::Number, mean::Number)
    return img + sqrt(variance)*randn(size(img)) + mean
end

imgaussiannoise{T}(img::AbstractArray{T}, variance::Number) = imgaussiannoise(img, variance, 0)
imgaussiannoise{T}(img::AbstractArray{T}) = imgaussiannoise(img, 0.01, 0)

# image gradients

# forward and backward differences
# can be very helpful for discretized continuous models
forwarddiffy{T}(u::Array{T,2}) = [u[2:end,:]; u[end,:]] - u
forwarddiffx{T}(u::Array{T,2}) = [u[:,2:end] u[:,end]] - u
backdiffy{T}(u::Array{T,2}) = u - [u[1,:]; u[1:end-1,:]]
backdiffx{T}(u::Array{T,2}) = u - [u[:,1] u[:,1:end-1]]

"""
```
imgr = imROF(img, lambda, iterations)
```

Perform Rudin-Osher-Fatemi (ROF) filtering, more commonly known as Total
Variation (TV) denoising or TV regularization. `lambda` is the regularization
coefficient for the derivative, and `iterations` is the number of relaxation
iterations taken. 2d only.
"""
function imROF{T}(img::Array{T,2}, lambda::Number, iterations::Integer)
    # Total Variation regularized image denoising using the primal dual algorithm
    # Also called Rudin Osher Fatemi (ROF) model
    # lambda: regularization parameter
    s1, s2 = size(img)
    p = zeros(T, s1, s2, 2)
    u = zeros(T, s1, s2)
    grad_u = zeros(T, s1, s2, 2)
    div_p = zeros(T, s1, s2)
    dt = lambda/4
    for i = 1:iterations
        div_p = backdiffx(p[:,:,1]) + backdiffy(p[:,:,2])
        u = img + div_p/lambda
        grad_u = cat(3, forwarddiffx(u), forwarddiffy(u))
        grad_u_mag = sqrt(grad_u[:,:,1].^2 + grad_u[:,:,2].^2)
        tmp = 1 + grad_u_mag*dt
        p = (dt*grad_u + p)./cat(3, tmp, tmp)
    end
    return u
end

# ROF Model for color images
function imROF(img::AbstractArray, lambda::Number, iterations::Integer)
    cd = colordim(img)
    local out
    if cd != 0
        out = similar(img)
        for i = size(img, cd)
            imsl = img["color", i]
            outsl = slice(out, "color", i)
            copy!(outsl, imROF(imsl, lambda, iterations))
        end
    else
        out = shareproperties(img, imROF(img, lambda, iterations))
    end
    out
end

# morphological operations for ImageMeta
dilate(img::ImageMeta, region=coords_spatial(img)) = shareproperties(img, dilate!(copy(data(img)), region))
erode(img::ImageMeta, region=coords_spatial(img)) = shareproperties(img, erode!(copy(data(img)), region))

# what are these `extr` functions? TODO: documentation

extr(order::ForwardOrdering, x::Real, y::Real) = max(x,y)
extr(order::ForwardOrdering, x::Real, y::Real, z::Real) = max(x,y,z)
extr(order::ReverseOrdering, x::Real, y::Real) = min(x,y)
extr(order::ReverseOrdering, x::Real, y::Real, z::Real) = min(x,y,z)

extr(order::Ordering, x::RGB, y::RGB) = RGB(extr(order, x.r, y.r), extr(order, x.g, y.g), extr(order, x.b, y.b))
extr(order::Ordering, x::RGB, y::RGB, z::RGB) = RGB(extr(order, x.r, y.r, z.r), extr(order, x.g, y.g, z.g), extr(order, x.b, y.b, z.b))

extr(order::Ordering, x::Color, y::Color) = extr(order, convert(RGB, x), convert(RGB, y))
extr(order::Ordering, x::Color, y::Color, z::Color) = extr(order, convert(RGB, x), convert(RGB, y), convert(RGB, z))


# phantom images

"""
```
phantom = shepp_logan(N,[M]; highContrast=true)
```

output the NxM Shepp-Logan phantom, which is a standard test image usually used
for comparing image reconstruction algorithms in the field of computed
tomography (CT) and magnetic resonance imaging (MRI). If the argument M is
omitted, the phantom is of size NxN. When setting the keyword argument
`highConstrast` to false, the CT version of the phantom is created. Otherwise,
the high contrast MRI version is calculated.
"""
function shepp_logan(M,N; highContrast=true)
  # Initially proposed in Shepp, Larry; B. F. Logan (1974).
  # "The Fourier Reconstruction of a Head Section". IEEE Transactions on Nuclear Science. NS-21.

  P = zeros(M,N)

  x = linspace(-1,1,M)'
  y = linspace(1,-1,N)

  centerX = [0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06]
  centerY = [0, -0.0184, 0, 0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605]
  majorAxis = [0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 0.046, 0.046, 0.023, 0.023]
  minorAxis = [0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 0.046, 0.023, 0.023, 0.046]
  theta = [0, 0, -18.0, 18.0, 0, 0, 0, 0, 0, 0]

  # original (CT) version of the phantom
  grayLevel = [2, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

  if(highContrast)
    # high contrast (MRI) version of the phantom
    grayLevel = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  end

  for l=1:length(theta)
    P += grayLevel[l] * (
           ( (cos(theta[l] / 360*2*pi) * (x .- centerX[l]) .+
              sin(theta[l] / 360*2*pi) * (y .- centerY[l])) / majorAxis[l] ).^2 .+
           ( (sin(theta[l] / 360*2*pi) * (x .- centerX[l]) .-
              cos(theta[l] / 360*2*pi) * (y .- centerY[l])) / minorAxis[l] ).^2 .< 1 )
  end

  return P
end

shepp_logan(N;highContrast=true) = shepp_logan(N,N;highContrast=highContrast)

"""
```
integral_img = integral_image(img)
```

Returns the integral image of an image. The integral image is calculated by assigning
to each pixel the sum of all pixels above it and to its left, i.e. the rectangle from
(1, 1) to the pixel. An integral image is a data structure which helps in efficient
calculation of sum of pixels in a rectangular subset of an image. See `boxdiff` for more
information.
"""
function integral_image(img::AbstractArray)
    integral_img = Array{accum(eltype(img))}(size(img))
    sd = coords_spatial(img)
    cumsum!(integral_img, img, sd[1])
    for i = 2:length(sd)
        cumsum!(integral_img, integral_img, sd[i])
    end
    integral_img
end

"""
```
sum = boxdiff(integral_image, ytop:ybot, xtop:xbot)
sum = boxdiff(integral_image, CartesianIndex(tl_y, tl_x), CartesianIndex(br_y, br_x))
sum = boxdiff(integral_image, tl_y, tl_x, br_y, br_x)
```

An integral image is a data structure which helps in efficient calculation of sum of pixels in
a rectangular subset of an image. It stores at each pixel the sum of all pixels above it and to
its left. The sum of a window in an image can be directly calculated using four array
references of the integral image, irrespective of the size of the window, given the `yrange` and
`xrange` of the window. Given an integral image -

        A - - - - - - B -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        - * * * * * * * -
        C * * * * * * D -
        - - - - - - - - -

The sum of pixels in the area denoted by * is given by S = D + A - B - C.
"""
boxdiff{T}(int_img::AbstractArray{T, 2}, y::UnitRange, x::UnitRange) = boxdiff(int_img, y.start, x.start, y.stop, x.stop)
boxdiff{T}(int_img::AbstractArray{T, 2}, tl::CartesianIndex, br::CartesianIndex) = boxdiff(int_img, tl[1], tl[2], br[1], br[2])

function boxdiff{T}(int_img::AbstractArray{T, 2}, tl_y::Integer, tl_x::Integer, br_y::Integer, br_x::Integer)
    sum = int_img[br_y, br_x]
    sum -= tl_x > 1 ? int_img[br_y, tl_x - 1] : zero(T)
    sum -= tl_y > 1 ? int_img[tl_y - 1, br_x] : zero(T)
    sum += tl_y > 1 && tl_x > 1 ? int_img[tl_y - 1, tl_x - 1] : zero(T)
    sum
end

"""
```
P = bilinear_interpolation(img, r, c)
```

Bilinear Interpolation is used to interpolate functions of two variables
on a rectilinear 2D grid.

The interpolation is done in one direction first and then the values obtained
are used to do the interpolation in the second direction.

"""
function bilinear_interpolation{T}(img::AbstractArray{T, 2}, y::Number, x::Number)
    y_min = floor(Int, y)
    x_min = floor(Int, x)
    y_max = ceil(Int, y)
    x_max = ceil(Int, x)

    topleft = chkbounds(Bool, img, y_min, x_min) ? img[y_min, x_min] : zero(T)
    bottomleft = chkbounds(Bool, img, y_max, x_min) ? img[y_max, x_min] : zero(T)
    topright = chkbounds(Bool, img, y_min, x_max) ? img[y_min, x_max] : zero(T)
    bottomright = chkbounds(Bool, img, y_max, x_max) ? img[y_max, x_max] : zero(T)

    if x_max == x_min
        if y_max == y_min
            return T(topleft)
        end
        return T(((y_max - y) * topleft + (y - y_min) * bottomleft) / (y_max - y_min))
    elseif y_max == y_min
        return T(((x_max - x) * topleft + (x - x_min) * topright) / (x_max - x_min))
    end

    r1 = ((x_max - x) * topleft + (x - x_min) * topright) / (x_max - x_min)
    r2 = ((x_max - x) * bottomleft + (x - x_min) * bottomright) / (x_max - x_min)

    T(((y_max - y) * r1 + (y - y_min) * r2) / (y_max - y_min))

end

if VERSION < v"0.5.0-dev+4754"
    chkbounds(::Type{Bool}, img, x, y)  = checkbounds(Bool, size(img, 1), y) && checkbounds(Bool, size(img, 2), x)
else
    chkbounds(::Type{Bool}, img, x, y) = checkbounds(Bool, img, x, y)
end

"""
```
pyramid = gaussian_pyramid(img, n_scales, downsample, sigma)
```

Returns a  gaussian pyramid of scales `n_scales`, each downsampled
by a factor `downsample` and `sigma` for the gaussian kernel.

"""
function gaussian_pyramid{T,N}(img::AbstractArray{T,N}, n_scales::Int, downsample::Real, sigma::Real)
    kerng = KernelFactors.IIRGaussian(sigma)
    kern = ntuple(d->kerng, Val{N})
    gaussian_pyramid(img, n_scales, downsample, kern)
end

function gaussian_pyramid{T,N}(img::AbstractArray{T,N}, n_scales::Int, downsample::Real, kern::NTuple{N,Any})
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
function otsu_threshold{T<:Union{Gray,Real}, N}(img::AbstractArray{T, N}, bins::Int = 256)

    min, max = extrema(img)
    edges, counts = imhist(img, linspace(gray(min), gray(max), bins))
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
