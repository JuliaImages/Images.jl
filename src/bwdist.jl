module FeatureTransform

export feature_transform, distance_transform

"""
    feature_transform(I::AbstractArray{Bool, N}, [w=nothing]) -> F

Compute the feature transform of a binary image `I`, finding the
closest "feature" (positions where `I` is `true`) for each location in
`I`.  Specifically, `F[i]` is a `CartesianIndex` encoding the position
closest to `i` for which `I[F[i]]` is `true`.  In cases where two or
more features in `I` have the same distance from `i`, an arbitrary
feature is chosen. If `I` has no `true` values, then all locations are
mapped to an index where each coordinate is `typemin(Int)`.

Optionally specify the weight `w` assigned to each coordinate.  For
example, if `I` corresponds to an image where voxels are anisotropic,
`w` could be the voxel spacing along each coordinate axis. The default
value of `nothing` is equivalent to `w=(1,1,...)`.

See also: [`distance_transform`](@ref).

# Citation

'A Linear Time Algorithm for Computing Exact Euclidean Distance
Transforms of Binary Images in Arbitrary Dimensions' [Maurer et al.,
2003] (DOI: 10.1109/TPAMI.2003.1177156)
"""
function feature_transform(I::AbstractArray{Bool,N}, w::Union{Nothing,NTuple{N}}=nothing) where N
    # To allocate temporary storage for voronoift!, compute one
    # element (so we have the proper type)
    fi = first(CartesianIndices(axes(I)))
    drft = DistRFT(fi, w, (), Base.tail(fi.I))
    tmp = Vector{typeof(drft)}()

    # Allocate the output
    F = similar(I, CartesianIndex{N})

    # Compute the feature transform (recursive algorithm)
    computeft!(F, I, w, CartesianIndex(()), tmp)

    F
end

"""
    distance_transform(F::AbstractArray{CartesianIndex}, [w=nothing]) -> D

Compute the distance transform of `F`, where each element `F[i]`
represents a "target" or "feature" location assigned to `i`.
Specifically, `D[i]` is the distance between `i` and `F[i]`.
Optionally specify the weight `w` assigned to each coordinate; the
default value of `nothing` is equivalent to `w=(1,1,...)`.

See also: [`feature_transform`](@ref).
"""
function distance_transform(F::AbstractArray{CartesianIndex{N},N}, w::Union{Nothing,NTuple{N}}=nothing) where N
    # To allocate the proper output type, compute the distance for one element
    R = CartesianIndices(axes(F))
    dst = wnorm2(zero(eltype(R)), w)
    D = similar(F, typeof(sqrt(dst)))

    _null = nullindex(F)
    @inbounds for i in R
        fi = F[i]
        D[i] = fi == _null ? Inf : sqrt(wnorm2(fi - i, w))
    end

    D
end

function computeft!(F, I, w, jpost::CartesianIndex{K}, tmp) where K
    _null = nullindex(F)
    if K == ndims(I)-1
        # Fig. 2, lines 2-8
        @inbounds @simd for i1 in axes(I, 1)
            F[i1, jpost] = ifelse(I[i1, jpost], CartesianIndex(i1, jpost), _null)
        end
    else
        # Fig. 2, lines 10-12
        for i1 in axes(I, ndims(I) - K)
            computeft!(F, I, w, CartesianIndex(i1, jpost), tmp)
        end
    end
    # Fig. 2, lines 14-20
    indspre = ftfront(axes(F), jpost)  # discards the trailing indices of F
    for jpre in CartesianIndices(indspre)
        voronoift!(F, I, w, jpre, jpost, tmp)
    end
    F
end

function voronoift!(F, I, w, jpre, jpost, tmp)
    d = length(jpre)+1
    _null = nullindex(F)
    empty!(tmp)
    for i in axes(I, d)
        # Fig 3, lines 3-13
        xi = CartesianIndex(jpre, i, jpost)
        @inbounds fi = F[xi]
        if fi != _null
            fidist = DistRFT(fi, w, jpre, jpost)
            if length(tmp) < 2
                push!(tmp, fidist)
            else
                @inbounds while length(tmp) >= 2 && removeft(tmp[end-1], tmp[end], fidist)
                    pop!(tmp)
                end
                push!(tmp, fidist)
            end
        end
    end
    nS = length(tmp)
    nS == 0 && return F
    # Fig 3, lines 18-24
    l = 1
    @inbounds fthis = tmp[l].fi
    for i in axes(I, d)
        xi = CartesianIndex(jpre, i, jpost)
        d2this = wnorm2(xi-fthis, w)
        while l < nS
            @inbounds fnext = tmp[l+1].fi
            d2next = wnorm2(xi-fnext, w)
            if d2this > d2next
                d2this, fthis = d2next, fnext
                l += 1
            else
                break
            end
        end
        @inbounds F[xi] = fthis
    end
    F
end

## Utilities

# Stores a feature location and its distance from the hyperplane Rd
struct DistRFT{N,T}
    fi::CartesianIndex{N}
    dist2::T
    d::Int  # the coordinate in dimension d
end

"""
    DistRFT(fi::CartesianIndex, w, jpre, jpost)

Bundles a feature `fi` together with its distance from the line Rd,
where Rd is specified by `(jpre..., :, jpost...)`. `w` is the
weighting applied to each coordinate, and must be `nothing` or be a
tuple with the same number of coordiantes as `fi`.

"""
function DistRFT(fi::CartesianIndex, w, jpre::CartesianIndex, jpost::CartesianIndex)
    d2pre, ipost, wpost = dist2pre(fi.I, w, jpre.I)
    d2post = wnorm2(CartesianIndex(ipost) - jpost, wpost)
    @inbounds fid = fi[length(jpre)+1]
    DistRFT(fi, d2pre + d2post, fid)
end
DistRFT(fi::CartesianIndex, w, jpre::Tuple, jpost::Tuple) =
    DistRFT(fi, w, CartesianIndex(jpre), CartesianIndex(jpost))

@inline function removeft(u, v, w)
    a, b, c = v.d-u.d, w.d-v.d, w.d-u.d
    c*v.dist2 - b*u.dist2 - a*w.dist2 > a*b*c
end

"""
    ftfront(inds, j::CartesianIndex{K})

Discard the last `K+1` elements of the tuple `inds`.
"""
ftfront(inds, j::CartesianIndex) = _ftfront((), inds, j)
_ftfront(out, inds::NTuple{N}, j::CartesianIndex{N}) where {N} = Base.front(out)
@inline _ftfront(out, inds, j) = _ftfront((out..., inds[1]), Base.tail(inds), j)

nullindex(A::AbstractArray{T,N}) where {T,N} = typemin(Int)*one(CartesianIndex{N})

"""
    wnorm2(x::CartesianIndex, w)

Compute `∑ (w[i]*x[i])^2`.  Specifying `nothing` for `w` is equivalent to `w = (1,1,...)`.
"""
wnorm2(x::CartesianIndex, w) = _wnorm2(0, x.I, w)
_wnorm2(s, ::Tuple{}, ::Nothing)    = s
_wnorm2(s, ::Tuple{}, ::Tuple{}) = s
@inline _wnorm2(s, x, w::Nothing) = _wnorm2(s + sqr(x[1]), Base.tail(x), w)
@inline _wnorm2(s, x, w) = _wnorm2(s + sqr(w[1]*x[1]), Base.tail(x), Base.tail(w))

"""
    dist2pre(x, w, jpre) -> s, xpost, wpost

`s` is equivalent to `wnorm2(x[1:length(jpre)]-jpre, w)`. `xpost` and
`wpost` contain the trailing indices of `x` and `w` (skipping the
element `length(jpre)+1`).
"""
dist2pre(x::Tuple, w, jpre) = _dist2pre(0, x, w, jpre)
_dist2pre(s, x, w::Nothing, ::Tuple{}) = s, Base.tail(x), w
_dist2pre(s, x, w,       ::Tuple{}) = s, Base.tail(x), Base.tail(w)
@inline _dist2pre(s, x, w::Nothing, jpre) = _dist2pre(s + sqr(x[1]-jpre[1]), Base.tail(x), w, Base.tail(jpre))
@inline _dist2pre(s, x, w, jpre) = _dist2pre(s + sqr(w[1]*(x[1]-jpre[1])), Base.tail(x), Base.tail(w), Base.tail(jpre))

@inline sqr(x) = x*x

end # module
