using Base.Cartesian


"""
    permutesubs!(subs, perm::Vector{Int}, result:AbstractArray{Int})

Permute a tuple of subscripts given a permutation vector and
writing the permutation into `result`.
"""
function permutesubs!(subs, perm::Vector{Int}, result::Array{Int})
  n = length(subs)
  @inbounds @simd for i = 1:n
    result[i] = subs[perm[i]]
  end
  return result
end

"""
    permutedimsubs!(F::AbstractArray{Int, N}, perm::AbstractVector{Int}, stride::AbstractArray{Int}, sizeI::Tuple, B::AbstractArray{Int, N}, tempArray::AbstractArray{Int})

Permute the dimensions of an array and those of the linear indices stored in that array,
using `tempArray` as a temporary array for permuting the subscripts,
`B` as a temporary array for permuted dims, and `stride` as the strides of array `F`.
"""
function permutedimsubs!{N}(F::AbstractArray{Int, N}, perm::Vector{Int}, stride::AbstractArray{Int}, sizeI::Tuple, B::AbstractArray{Int, N}, tempArray::AbstractArray{Int})
  permutedims!(B, F, perm)

  @inbounds @simd for i = 1:length(B)
    F[i] = B[i] == 0 ? 0 : stridedSub2Ind(stride, permutesubs!(ind2sub(sizeI, B[i]), perm, tempArray))
  end

  return F
end

"""
    stridedSub2Ind(stride::AbstractArray{Int}, i::AbstractArray{Int})

Compute the index for given subindices using an array's strides.

Replacing `sub2ind` in order to reduce memory consumption.
"""
function stridedSub2Ind(stride::AbstractArray{Int}, i::AbstractArray{Int})
  s = 1
  @inbounds @fastmath for j = 1:length(stride)
    s += (i[j]-1)*stride[j]
  end
  return s
end


# Relevant distance functions copied from Distance.jl
function get_common_len(a::AbstractVector, b::AbstractVector)
  n = length(a)
  length(b) == n || throw(DimensionMismatch("The lengths of a and b must match."))
  return n
end

function sumsqdiff(a::AbstractVector, b::AbstractVector)
  n = get_common_len(a, b)::Int
  s = 0.

  @inbounds for i = 1:n
    s += abs2(a[i] - b[i])
  end

  return s
end

sqeuclidean(a::AbstractVector, b::AbstractVector) = sumsqdiff(a, b)
euclidean(a::AbstractVector, b::AbstractVector) = sqrt(sumsqdiff(a, b))


"""
    bwdist(I::AbstractArray{Bool, N})

Compute the euclidean distance and feature transform of a binary image.

Implemented according to
'A Linear Time Algorithm for Computing
Exact Euclidean Distance Transforms of
Binary Images in Arbitrary Dimensions' [Maurer et al., 2003]
(DOI: 10.1109/TPAMI.2003.1177156)
"""
@generated function bwdist{N}(I::AbstractArray{Bool, N})
  quote
    # size
    sizeI = size(I)
    # generate temporary arrays
    tempArray = Array{Int}($N)
    tempArray2 = Array{Int}(length(I) + 1)
    # generate strides array
    stride = collect(strides(I))

    # F and D, we use D as a temporary array for permutedimsubs
    F = zeros(Int, sizeI)
    D = zeros(Int, sizeI)

    _computeft!(F, I, stride)
    d = 1:$N
    @inbounds for i = d
      _voronoift!(F, I, stride, sizeI, tempArray2)
      F = permutedimsubs!(F, circshift(d, 1), stride, sizeI, D, tempArray)
    end

    setindex!(D, 0)
    @inbounds @nloops $N i F begin
      x = 1
      (@nexprs $N j -> (x += (i_j - 1)*stride[j]))
      (@nref $N D i) = sqeuclidean((@nref $N F i), x, sizeI)
    end

    return (F, D)
  end
end

"""
    \_computeft!(F::AbstractArray{Int, N}, I::AbstractArray{Bool, N}, stride::AbstractArray{Int})

Compute F_0 in the parlance of Maurer et al. 2003.
"""
@generated function _computeft!{N}(F::AbstractArray{Int, N}, I::AbstractArray{Bool, N}, stride::AbstractArray{Int})
  quote
    @inbounds @nloops $N i I begin
      ind = 1
      (@nref $N F i) = (@nref $N I i) ? (@nexprs $N d -> (ind += (i_d - 1)*stride[d])) : 0
    end
  end
end

"""
    \_voronoift!(F::AbstractArray{Int, N}, I::AbstractArray{Bool, N}, stride::AbstractArray{Int}, sizeI::Tuple, g::AbstractArray{Int})

Compute the partial Voronoi diagram along the first dimension of F,
using g as a temporary array, following Maurer et al. 2003.
"""
@generated function _voronoift!{N}(F::AbstractArray{Int, N}, I::AbstractArray{Bool, N}, stride::AbstractArray{Int}, sizeI::Tuple, g::AbstractArray{Int})
  quote
    @inbounds @nloops $N d j -> (j == 1 ? 0 : 1:sizeI[j]) begin
      Fstar = (@nref $N F j -> (j == 1 ? (:) : d_j))
      l = 0
      setindex!(g, 0)

      @inbounds for i = 1:size(Fstar, 1)
        f = Fstar[i]
        if f != 0
          if l < 2
            l += 1
            g[l] = f
          else
            while l >= 2 && removeft(g[l-1], g[l], f, d_2, sizeI)
              l -= 1
            end
            l += 1
            g[l] = f
          end
        end
      end

      n_s = l
      if n_s == 0
      else
        l = 1
        @inbounds for d_1 = 1:length(Fstar)
          # This makes a vector x containing the appropriate subscripts
          x = 1
          (@nexprs $N j -> (x += (d_j - 1)*stride[j]))
          while l < n_s && (euclidean(x, g[l], sizeI) > euclidean(x, g[l+1], sizeI))
            l += 1
          end
          (@nref $N F d) = g[l]
        end
      end
    end
  end
end

"""
    removeft(u::Int, v::Int, w::INt, r::Int, dims)

Calculate whether we should remove a feature pixel from the Voronoi diagram.
"""
function removeft(u::Int, v::Int, w::Int, r::Int, dims)
  u1 = ind2sub(dims, u)[1]
  v1 = ind2sub(dims, v)[1]
  w1 = ind2sub(dims, w)[1]
  a = v1 - u1
  b = w1 - v1
  c = a + b

  return c*distance2(v, r, dims) - b*distance2(u, r, dims) - a*distance2(w, r, dims) - a*b*c > 0
end

"""
    distance2(u::Int, r::Int, dims)

Calculate the Squared Euclidean distance of u, given by a linear index, to a row index.
"""
function distance2(u::Int, r::Int, dims)
  u2 = ind2sub(dims, u)[2]
  return (u2 - r)^2
end

"""
    sqeuclidean(x::Int, g::Int, dims)

Calculate the Squared Euclidean Distance between linear indices.
"""
function sqeuclidean(x::Int, g::Int, dims)
  x_subs = ind2sub(dims, x)
  g_subs = ind2sub(dims, g)

  s = 0
  @inbounds for i = 1:length(dims)
    s += (x_subs[i] - g_subs[i])^2
  end
  return s
end

# Euclidean distance between linear indices
euclidean(x::Int, g::Int, dims) = sqrt(sqeuclidean(x, g, dims))
