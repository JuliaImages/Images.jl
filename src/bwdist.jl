using Base.Cartesian

"""
    bwdist(I::AbstractArray{Bool, N}) -> F, D

Compute the euclidean distance and feature transform of a binary image,
where `F` is trhe feature transform (the closest feature pixel map)
and `D` is the euclidean distance transform of `I`.

Implemented according to
'A Linear Time Algorithm for Computing
Exact Euclidean Distance Transforms of
Binary Images in Arbitrary Dimensions' [Maurer et al., 2003]
(DOI: 10.1109/TPAMI.2003.1177156)
"""
function bwdist{N}(I::AbstractArray{Bool, N})
    sizeI = size(I)
    # generate temporary arrays
    tempArray = Array{Int}(length(I) + 1)

    # F and D
    F = zeros(Int, sizeI)
    D = zeros(Float64, sizeI)
    stride = collect(strides(F))

    _computeft!(F, I, stride)

    if N > 1
        for i = 1:N
            sizeF = size(F)
            _voronoift!(F, I, sizeF, stride, tempArray)
            (F, stride) = permutedimsubs(F, sizeF)
        end
    else
        sizeF = size(F)
        _voronoift!(F, I, sizeF, stride, tempArray)
    end

    @inbounds for i in eachindex(F)
        D[i] = euclidean(ind2sub(sizeI, F[i]), subindex(sizeI, i))
    end

    return (F, D)
end

@inline subindex(dims::Tuple, i::Int) = ind2sub(dims, i)
@inline subindex(dims::Tuple, i::CartesianIndex) = i.I

"""
    \_computeft!(F::AbstractArray{Int, N}, I::AbstractArray{Bool, N}, stride::AbstractArray{Int})

Compute F_0, indicating the closest feature voxel in the 0th dimension where
for each voxel x in I `F_0(x) = x if x = 1 and 0 otherwise`,
in the parlance of Maurer et al. 2003.
"""
function _computeft!(F::AbstractArray{Int}, I::AbstractArray{Bool}, stride::AbstractArray{Int})
    @inbounds @simd for i in eachindex(F)
        F[i] = ifelse(I[i], stridedGetindex(i, stride), 0)
    end
end

@inline stridedGetindex(i::Int, stride::AbstractArray{Int}) = i
@inline stridedGetindex(i::CartesianIndex, stride::AbstractArray{Int}) = stridedSub2Ind(stride, i.I)

"""
    \_voronoift!(F::AbstractArray{Int, N}, I::AbstractArray{Bool, N}, sizeF::Tuple, stride::AbstractArray{Int}, g::AbstractArray{Int})

Compute the partial Voronoi diagram along the first dimension of `F`,
using `g` as a temporary array, following Maurer et al. 2003.
"""
@generated function _voronoift!{N}(F::AbstractArray{Int, N}, I::AbstractArray{Bool, N}, sizeF::Tuple, stride::AbstractArray{Int}, g::AbstractArray{Int})
  quote
    @inbounds @nloops $N d j -> (j == 1 ? 0 : 1:sizeF[j]) begin
      l = 0

      @inbounds for i = 1:sizeF[1]
        f = (@nref $N F j -> (j == 1 ? i : d_j))
        if f != 0
          if l < 2
            l += 1
            g[l] = f
          else
            while l >= 2 && removeft(g[l-1], g[l], f, d_2, sizeF)
              l -= 1
            end
            l += 1
            g[l] = f
          end
          g[l] = f
        end
      end

      n_s = l
      if n_s != 0
        l = 1
        @inbounds @fastmath for d_1 = 1:sizeF[1]
          # This makes the index x from subscripts d_{1, ...}
          x = 1
          (@nexprs $N j -> x = x + (d_j - 1) * stride[j])
          while l < n_s && (sqeuclidean(x, g[l], sizeF) > sqeuclidean(x, g[l+1], sizeF))
            l += 1
          end
          (@nref $N F d) = g[l]
        end
      end
    end
  end
end

"""
    removeft(u::Int, v::Int, w::Int, r::Int, dims::Tuple)

Calculate whether we should remove a feature pixel from the Voronoi diagram.
"""
function removeft(u::Int, v::Int, w::Int, r::Int, dims::Tuple)
    @inbounds begin
        uIdx = ind2sub(dims, u)
        vIdx = ind2sub(dims, v)
        wIdx = ind2sub(dims, w)

        a = vIdx[1] - uIdx[1]
        b = wIdx[1] - vIdx[1]
        c = a + b

        return c*distance2(vIdx[2], r) - b*distance2(uIdx[2], r) - a*distance2(wIdx[2], r) - a*b*c > 0
    end
end

"""
    circperm(t::Tuple)

Circularly shift the data in the tuple.
"""
circperm(t::Tuple) = _circperm(t)
@inline _circperm(t::Tuple) = _circperm(t...)
@inline _circperm(t1, trest...) = (trest..., t1)

"""
    permutedimsubs(F::AbstractArray{Int, N}, sizeF::Tuple) -> B, stride

Permute the dimensions of an array and those of the linear indices stored in that array,
and returning `B` as the permuted array and `stride` indicating `strides(B)`.
"""
function permutedimsubs(F::AbstractArray{Int}, sizeF::Tuple)
    B = transpose(F)

    stride = collect(strides(B))
    @inbounds for i in eachindex(B)
        B[i] = B[i] == 0 ? 0 : stridedSub2Ind(stride, circperm(ind2sub(sizeF, B[i])))
    end

    return (B, stride)
end

"""
    stridedSub2Ind(stride::AbstractArray{Int}, i::Tuple)

Compute the index for given subindices using an array's strides.

Replacing `sub2ind` in order to reduce memory consumption.
"""
function stridedSub2Ind(stride::AbstractArray{Int}, i::Tuple)
    s = 1
    @inbounds @fastmath @simd for j in eachindex(i)
        ij = i[j] - 1
        stridej = stride[j]
        s = s + ij * stridej
    end
    return s
end

"""
    distance2(u::Int, r::Int, dims::Tuple)

Calculate the Squared Euclidean distance of u, given by a linear index, to a row index.
"""
@inline distance2(u::Int, r::Int) = abs2(u - r)

"""
    sqeuclidean(x::Int, g::Int, dims::Tuple)

Calculate the Squared Euclidean Distance between linear indices.
"""
function sqeuclidean(x::Int, g::Int, dims::Tuple)
    x_subs = ind2sub(dims, x)
    g_subs = ind2sub(dims, g)
    return sqeuclidean(x_subs, g_subs)
end

"""
    sqeuclidean(x::Tuple, g::Tuple)

Calculate the Squared Euclidean Distance between cartesian indices.
"""
function sqeuclidean(x_subs::Tuple, g_subs::Tuple)
    # taken from Distances.jl
    s = 0
    @inbounds @simd for I in eachindex(x_subs, g_subs)
        xi = x_subs[I]
        gi = g_subs[I]
        s = s + abs2(xi - gi)
    end
    return s
end

"""
    euclidean(x::Int, g::Int, dims::Tuple)

Calculate the Euclidean Distance between linear indices.
"""
euclidean(x::Int, g::Int, dims::Tuple) = sqrt(sqeuclidean(x, g, dims))

"""
    euclidean(x::Int, g::CartesianIndex, dims::Tuple)

Calculate the Euclidean Distance between cartesian indices.
"""
euclidean(x::Tuple, g::Tuple) = sqrt(sqeuclidean(x, g))
