using Base: axes1, tail
using OffsetArrays
import Statistics
using Statistics: mean
import AxisArrays

# Compat.@dep_vectorize_2arg Gray atan2
# Compat.@dep_vectorize_2arg Gray hypot

"""
`M = meanfinite(img, region)` calculates the mean value along the dimensions listed in `region`, ignoring any non-finite values.
"""
meanfinite(A::AbstractArray{T}, region) where {T<:Real} = _meanfinite(A, T, region)
meanfinite(A::AbstractArray{CT}, region) where {CT<:Colorant} = _meanfinite(A, eltype(CT), region)
function _meanfinite(A::AbstractArray, ::Type{T}, region) where T<:AbstractFloat
    inds = Base.reduced_indices(A, region)
    K = similar(Array{Int}, inds)
    S = similar(Array{eltype(A)}, inds)
    fill!(K, 0)
    fill!(S, zero(eltype(A)))
    sumfinite!(S, K, A)
    S./K
end
_meanfinite(A::AbstractArray, ::Type, region) = mean(A, dims=region)  # non floating-point

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
function sumfinite!(S, K, A::AbstractArray{T,N}) where {T,N}
    check_reducedims(S, A)
    isempty(A) && return S, K
    axes(S) == axes(K) || throw(DimensionMismatch("S and K must have identical axes"))

    indsAt, indsSt = safe_tail(axes(A)), safe_tail(axes(S))
    keep, Idefault = _newindexer(indsSt)
    if reducedim1(S, A)
        # keep the accumulators as a local variable when reducing along the first dimension
        i1 = first(axes1(S))
        @inbounds for IA in CartesianIndices(indsAt)
            IS = newindex(IA, keep, Idefault)
            s, k = S[i1,IS], K[i1,IS]
            for i in axes(A, 1)
                tmp = A[i, IA]
                if isfinite(tmp)
                    s += tmp
                    k += 1
                end
            end
            S[i1,IS], K[i1,IS] = s, k
        end
    else
        @inbounds for IA in CartesianIndices(indsAt)
            IS = newindex(IA, keep, Idefault)
            for i in axes(A, 1)
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
_newindexer(ax) = Base.Broadcast.shapeindexer(ax)

function Statistics.var(A::AbstractArray{C}; kwargs...) where C<:AbstractGray
    imgc = channelview(A)
    base_colorant_type(C)(var(imgc; kwargs...))
end

function Statistics.var(A::AbstractArray{C,N}; kwargs...) where {C<:Colorant,N}
    imgc = channelview(A)
    colons = ntuple(d->Colon(), Val(N))
    inds1 = axes(imgc, 1)
    val1 = Statistics.var(view(imgc, first(inds1), colons...); kwargs...)
    vals = similar(imgc, typeof(val1), inds1)
    vals[1] = val1
    for i in first(inds1)+1:last(inds1)
        vals[i] = Statistics.var(view(imgc, i, colons...); kwargs...)
    end
    base_colorant_type(C)(vals...)
end

Statistics.std(A::AbstractArray{C}; kwargs...) where {C<:Colorant} = mapc(sqrt, Statistics.var(A; kwargs...))

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

# functions red, green, and blue
for (funcname, fieldname) in ((:red, :r), (:green, :g), (:blue, :b))
    fieldchar = string(fieldname)[1]
    @eval begin
        function $funcname(img::AbstractArray{CV}) where CV<:Color
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
function minfinite(A::AbstractArray{T}) where T
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
function maxfinite(A::AbstractArray{T}) where T
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
function maxabsfinite(A::AbstractArray{T}) where T
    ret = sentinel_min(typeof(abs(A[1])))
    for a in A
        ret = maxfinite_scalar(abs(a), ret)
    end
    ret
end

minfinite_scalar(a::T, b::T) where {T} = isfinite(a) ? (b < a ? b : a) : b
maxfinite_scalar(a::T, b::T) where {T} = isfinite(a) ? (b > a ? b : a) : b
minfinite_scalar(a::T, b::T) where {T<:Union{Integer,FixedPoint}} = b < a ? b : a
maxfinite_scalar(a::T, b::T) where {T<:Union{Integer,FixedPoint}} = b > a ? b : a
minfinite_scalar(a, b) = minfinite_scalar(promote(a, b)...)
maxfinite_scalar(a, b) = maxfinite_scalar(promote(a, b)...)

function minfinite_scalar(c1::C, c2::C) where C<:AbstractRGB
    C(minfinite_scalar(c1.r, c2.r),
      minfinite_scalar(c1.g, c2.g),
      minfinite_scalar(c1.b, c2.b))
end
function maxfinite_scalar(c1::C, c2::C) where C<:AbstractRGB
    C(maxfinite_scalar(c1.r, c2.r),
      maxfinite_scalar(c1.g, c2.g),
      maxfinite_scalar(c1.b, c2.b))
end

sentinel_min(::Type{T}) where {T<:Union{Integer,FixedPoint}} = typemax(T)
sentinel_max(::Type{T}) where {T<:Union{Integer,FixedPoint}} = typemin(T)
sentinel_min(::Type{T}) where {T<:AbstractFloat} = convert(T, NaN)
sentinel_max(::Type{T}) where {T<:AbstractFloat} = convert(T, NaN)
sentinel_min(::Type{C}) where {C<:AbstractRGB} = _sentinel_min(C, eltype(C))
_sentinel_min(::Type{C},::Type{T}) where {C<:AbstractRGB,T} = (s = sentinel_min(T); C(s,s,s))
sentinel_max(::Type{C}) where {C<:AbstractRGB} = _sentinel_max(C, eltype(C))
_sentinel_max(::Type{C},::Type{T}) where {C<:AbstractRGB,T} = (s = sentinel_max(T); C(s,s,s))
sentinel_min(::Type{C}) where {C<:AbstractGray} = _sentinel_min(C, eltype(C))
_sentinel_min(::Type{C},::Type{T}) where {C<:AbstractGray,T} = C(sentinel_min(T))
sentinel_max(::Type{C}) where {C<:AbstractGray} = _sentinel_max(C, eltype(C))
_sentinel_max(::Type{C},::Type{T}) where {C<:AbstractGray,T} = C(sentinel_max(T))


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
    axes(A) == axes(B) || throw(DimensionMismatch("A and B must have the same axes"))
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

difftype(::Type{T}) where {T<:Integer} = Int
difftype(::Type{T}) where {T<:Real} = Float32
difftype(::Type{Float64}) = Float64
difftype(::Type{CV}) where {CV<:Colorant} = difftype(CV, eltype(CV))
difftype(::Type{CV}, ::Type{T}) where {CV<:RGBA,T<:Real} = RGBA{Float32}
difftype(::Type{CV}, ::Type{Float64}) where {CV<:RGBA} = RGBA{Float64}
difftype(::Type{CV}, ::Type{T}) where {CV<:BGRA,T<:Real} = BGRA{Float32}
difftype(::Type{CV}, ::Type{Float64}) where {CV<:BGRA} = BGRA{Float64}
difftype(::Type{CV}, ::Type{T}) where {CV<:AbstractGray,T<:Real} = Gray{Float32}
difftype(::Type{CV}, ::Type{Float64}) where {CV<:AbstractGray} = Gray{Float64}
difftype(::Type{CV}, ::Type{T}) where {CV<:AbstractRGB,T<:Real} = RGB{Float32}
difftype(::Type{CV}, ::Type{Float64}) where {CV<:AbstractRGB} = RGB{Float64}

accum(::Type{T}) where {T<:Integer} = Int
accum(::Type{Float32})    = Float32
accum(::Type{T}) where {T<:Real} = Float64
accum(::Type{C}) where {C<:Colorant} = base_colorant_type(C){accum(eltype(C))}

graytype(::Type{T}) where {T<:Number} = T
graytype(::Type{C}) where {C<:AbstractGray} = C
graytype(::Type{C}) where {C<:Colorant} = Gray{eltype(C)}

# normalized by Array size
"`s = ssdn(A, B)` computes the sum-of-squared differences over arrays/images A and B, normalized by array size"
ssdn(A::AbstractArray{T}, B::AbstractArray{T}) where {T} = ssd(A, B)/length(A)

# normalized by Array size
"`s = sadn(A, B)` computes the sum-of-absolute differences over arrays/images A and B, normalized by array size"
sadn(A::AbstractArray{T}, B::AbstractArray{T}) where {T} = sad(A, B)/length(A)

# normalized cross correlation
"""
`C = ncc(A, B)` computes the normalized cross-correlation of `A` and `B`.
"""
function ncc(A::AbstractArray{T}, B::AbstractArray{T}) where T
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
struct BlobLoG{T,S,N}
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
function blob_LoG(img::AbstractArray{T,N}, σscales::Union{AbstractVector,Tuple},
                  edges::Tuple{Vararg{Bool}}=(true, ntuple(d->false, Val(N))...), σshape=ntuple(d->1, Val(N))) where {T,N}
    sigmas = sort(σscales)
    img_LoG = Array{Float64}(undef, length(sigmas), size(img)...)
    colons = ntuple(d->Colon(), Val(N))
    @inbounds for isigma in eachindex(sigmas)
        img_LoG[isigma,colons...] = (-sigmas[isigma]) * imfilter(img, Kernel.LoG(ntuple(i->sigmas[isigma]*σshape[i],Val(N))))
    end
    maxima = findlocalmaxima(img_LoG, 1:ndims(img_LoG), edges)
    [BlobLoG(CartesianIndex(tail(x.I)), sigmas[x[1]], img_LoG[x]) for x in maxima]
end
blob_LoG(img::AbstractArray{T,N}, σscales, edges::Bool, σshape=ntuple(d->1, Val(N))) where {T,N} =
    blob_LoG(img, σscales, (edges, ntuple(d->edges,Val(N))...), σshape)

blob_LoG(img::AbstractArray{T,N}, σscales, σshape=ntuple(d->1, Val(N))) where {T,N} =
    blob_LoG(img, σscales, (true, ntuple(d->false,Val(N))...), σshape)

@inline function _clippedinds(Router,rstp)
    CartesianIndices(map((f,l)->f:l,
                         (first(Router)+rstp).I,(last(Router)-rstp).I))
end

findlocalextrema(img::AbstractArray{T,N}, region, edges::Bool, order) where {T,N} = findlocalextrema(img, region, ntuple(d->edges,Val(N)), order)

function findlocalextrema(img::AbstractArray{T,N}, region::Union{Tuple{Int,Vararg{Int}},Vector{Int},UnitRange{Int},Int}, edges::NTuple{N,Bool}, order::Base.Order.Ordering) where {T<:Union{Gray,Number},N}
    issubset(region,1:ndims(img)) || throw(ArgumentError("invalid region"))
    extrema = Array{CartesianIndex{N}}(undef, 0)
    edgeoffset = CartesianIndex(map(!, edges))
    R0 = CartesianIndices(axes(img))
    R = _clippedinds(R0,edgeoffset)
    rstp = one(first(R0))
    Rinterior = _clippedinds(R0,rstp)
    iregion = CartesianIndex(ntuple(d->d∈region, Val(N)))
    Rregion = CartesianIndices(map((f,l)->f:l,(-iregion).I, iregion.I))
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
`findlocalmaxima(img, [region, edges]) -> Vector{CartesianIndex}`

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

function restrict(img::AxisArray{T,N}, region::Dims) where {T,N}
    inregion = falses(ndims(img))
    inregion[[region...]] .= true
    inregiont = (inregion...,)::NTuple{N,Bool}
    AxisArray(restrict(img.data, region), map(modax, AxisArrays.axes(img), inregiont))
end

# FIXME: this doesn't get inferred, but it should be (see issue #628)
function restrict(img::Union{AxisArray,ImageMetaAxis}, ::Type{Ax}) where Ax
    A = restrict(img.data, axisdim(img, Ax))
    AxisArray(A, replace_axis(modax(img[Ax]), AxisArrays.axes(img)))
end

replace_axis(newax, axs) = _replace_axis(newax, axnametype(newax), axs...)
@inline _replace_axis(newax, ::Type{Ax}, ax::Ax, axs...) where {Ax} = (newax, _replace_axis(newax, Ax, axs...)...)
@inline _replace_axis(newax, ::Type{Ax}, ax, axs...) where {Ax} = (ax, _replace_axis(newax, Ax, axs...)...)
_replace_axis(newax, ::Type{Ax}) where {Ax} = ()

axnametype(ax::Axis{name}) where {name} = Axis{name}

function modax(ax)
    v = ax.val
    veven = range(v[1]-step(v)/2, stop=v[end]+step(v)/2, length=length(v)÷2 + 1)
    return ax(isodd(length(v)) ? oftype(veven, v[1:2:end]) : veven)
end

modax(ax, inregion::Bool) = inregion ? modax(ax) : ax


function imlineardiffusion(img::Array{T,2}, dt::AbstractFloat, iterations::Integer) where T
    u = img
    f = imlaplacian()
    for i = dt:dt:dt*iterations
        u = u + dt*imfilter(u, f, "replicate")
    end
    u
end

function imgaussiannoise(img::AbstractArray{T}, variance::Number, mean::Number) where T
    return img + sqrt(variance)*randn(size(img)) + mean
end

imgaussiannoise(img::AbstractArray{T}, variance::Number) where {T} = imgaussiannoise(img, variance, 0)
imgaussiannoise(img::AbstractArray{T}) where {T} = imgaussiannoise(img, 0.01, 0)

# image gradients

# forward and backward differences
# can be very helpful for discretized continuous models
forwarddiffy(u::AbstractMatrix) = [u[2:end,:]; u[end:end,:]] - u
forwarddiffx(u::AbstractMatrix) = [u[:,2:end] u[:,end:end]] - u
backdiffy(u::AbstractMatrix) = u - [u[1:1,:]; u[1:end-1,:]]
backdiffx(u::AbstractMatrix) = u - [u[:,1:1] u[:,1:end-1]]
function div(p::AbstractArray{T,3}) where T
    # Definition from the Chambolle citation below, between Eqs. 5 and 6
    # This is the adjoint of -forwarddiff
    inds = axes(p)[1:2]
    out = similar(p, inds)
    Router = CartesianIndices(inds)
    rstp = one(first(Router))
    Rinner = _clippedinds(Router,rstp)
    # Since most of the points are in the interior, compute them more quickly by avoiding branches
    for I in Rinner
        out[I] = p[I,1] - p[I[1]-1, I[2], 1] +
                 p[I,2] - p[I[1], I[2]-1, 2]
    end
    # Handle the edge points
    for I in EdgeIterator(Router, Rinner)
        out[I] = 0
        if I[1] == first(inds[1])
            out[I] += p[I, 1]
        elseif I[1] == last(inds[1])
            out[I] -= p[I[1]-1, I[2], 1]
        else
            out[I] += p[I,1] - p[I[1]-1, I[2], 1]
        end
        if I[2] == first(inds[2])
            out[I] += p[I, 2]
        elseif I[2] == last(inds[2])
            out[I] -= p[I[1], I[2]-1, 2]
        else
            out[I] += p[I,2] - p[I[1], I[2]-1, 2]
        end
    end
    out
end

"""
```
imgr = imROF(img, λ, iterations)
```

Perform Rudin-Osher-Fatemi (ROF) filtering, more commonly known as Total
Variation (TV) denoising or TV regularization. `λ` is the regularization
coefficient for the derivative, and `iterations` is the number of relaxation
iterations taken. 2d only.

See https://en.wikipedia.org/wiki/Total_variation_denoising and
Chambolle, A. (2004). "An algorithm for total variation minimization and applications".
    Journal of Mathematical Imaging and Vision. 20: 89–97
"""
function imROF(img::AbstractMatrix{T}, λ::Number, iterations::Integer) where T<:NumberLike
    # Total Variation regularized image denoising using the primal dual algorithm
    # Also called Rudin Osher Fatemi (ROF) model
    # λ: regularization parameter
    s1, s2 = size(img)
    p = zeros(T, s1, s2, 2)
    # This iterates Eq. (9) of the Chambolle citation
    local u
    τ = 1/4   # see 2nd remark after proof of Theorem 3.1.
    for i = 1:iterations
        div_p = div(p)
        u = img - λ*div_p # multiply term inside ∇ by -λ. Thm. 3.1 relates this to u via Eq. 7.
        grad_u = cat(forwarddiffy(u), forwarddiffx(u), dims=3)
        grad_u_mag = sqrt.(sum(abs2, grad_u, dims=3))
        p .= (p .- (τ/λ).*grad_u)./(1 .+ (τ/λ).*grad_u_mag)
    end
    return u
end

# ROF Model for color images
function imROF(img::AbstractMatrix{<:Color}, λ::Number, iterations::Integer)
    out = similar(img)
    imgc = channelview(img)
    outc = channelview(out)
    for chan = 1:size(imgc, 1)
        outc[chan, :, :] = imROF(imgc[chan, :, :], λ, iterations)
    end
    out
end

# morphological operations for ImageMeta
dilate(img::ImageMeta, region=coords_spatial(img)) = shareproperties(img, dilate!(copy(data(img)), region))
erode(img::ImageMeta, region=coords_spatial(img)) = shareproperties(img, erode!(copy(data(img)), region))

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

  x = range(-1, stop=1, length=M)'
  y = range(1, stop=-1, length=N)

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
    integral_img = Array{accum(eltype(img))}(undef, size(img))
    sd = coords_spatial(img)
    cumsum!(integral_img, img, dims=sd[1])
    for i = 2:length(sd)
        cumsum!(integral_img, integral_img, dims=sd[i])
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
boxdiff(int_img::AbstractArray{T, 2}, y::UnitRange, x::UnitRange) where {T} = boxdiff(int_img, y.start, x.start, y.stop, x.stop)
boxdiff(int_img::AbstractArray{T, 2}, tl::CartesianIndex, br::CartesianIndex) where {T} = boxdiff(int_img, tl[1], tl[2], br[1], br[2])

function boxdiff(int_img::AbstractArray{T, 2}, tl_y::Integer, tl_x::Integer, br_y::Integer, br_x::Integer) where T
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
function bilinear_interpolation(img::AbstractArray{T, 2}, y::Number, x::Number) where T
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

chkbounds(::Type{Bool}, img, x, y) = checkbounds(Bool, img, x, y)

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
    off = (.-ceil.(Int,(.-iterate.(axes(img).-(1,1))[1])./downsample))
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

"""
```
cleared_img = clearborder(img)
cleared_img = clearborder(img, width)
cleared_img = clearborder(img, width, background)
```

Returns a copy of the original image after clearing objects connected to the border of the image.

Parameters:

 -  img          = Binary/Grayscale input image
 -  width        = Width of the border examined (Default value is 1)
 -  background   = Value to be given to pixels/elements that are cleared (Default value is 0)

"""
function clearborder(img::AbstractArray, width::Int=1, background::Int=0)

    for i in size(img)
        if(width > i)
            throw(ArgumentError("Border width must not be greater than size of the image."))
        end
    end

    connectivity = ntuple(i -> 3, ndims(img))
    labels = label_components(img,trues(connectivity))
    number = maximum(labels) + 1

    dimensions = size(img)
    outerrange = CartesianIndices(map(i -> 1:i, dimensions))
    innerrange = CartesianIndices(map(i -> 1+width:i-width, dimensions))

    border_labels = Set{Int64}()
    for i in EdgeIterator(outerrange,innerrange)
        push!(border_labels, labels[i])
    end

    new_img = similar(img)
    for itr in eachindex(labels)
        if labels[itr] in border_labels
            new_img[itr] = background
        else
        	new_img[itr] = img[itr]
        end
    end

    return new_img

end
