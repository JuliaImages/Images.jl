"""
    Percentile(x)

Indicate that `x` should be interpreted as a [percentile](https://en.wikipedia.org/wiki/Percentile) rather than an absolute value. For example,

- `canny(img, 1.4, (80, 20))` uses absolute thresholds on the edge magnitude image
- `canny(img, 1.4, (Percentile(80), Percentile(20)))` uses percentiles of the edge magnitude image as threshold
"""
struct Percentile{T} <: Real p::T end


"""
HomogeneousPoint(x::NTuple{N, T})

In projective geometry [homogeneous coordinates](https://en.wikipedia.org/wiki/Homogeneous_coordinates) are the
natural coordinates for describing points and lines.

For instance, the homogeneous coordinates for a planar point are a triplet of real numbers ``(u, v ,w)``, with ``w \\neq 0``.
This triple can be associated with a point ``P = (x,y)`` in Cartesian coordinates, where ``x = \\frac{u}{w}`` and ``y = \\frac{v}{w}``
[(more details)](http://www.geom.uiuc.edu/docs/reference/CRC-formulas/node6.html#SECTION01140000000000000000).

In particular, the `HomogeneousPoint((10.0,5.0,1.0))` is the standardised projective representation of the Cartesian
point `(10.0,5.0)`.
"""
struct HomogeneousPoint{T <: AbstractFloat,N}
    coords::NTuple{N, T}
end

# By overwriting Base.to_indices we can define how to index into an N-dimensional array
# given an (N+1)-dimensional [`HomogeneousPoint`](@ref) type.
# We do this by converting the homogeneous coordinates to Cartesian coordinates
# and rounding to nearest integer.
#
# For homogeneous coordinates of a planar point we return
# a tuple of permuted Cartesian coordinates, (y,x), since matrices
# are indexed  according to row and then column.
# For homogeneous coordinates of other dimensions we do not permute
# the corresponding Cartesian coordinates.
Base.to_indices(A::AbstractArray, p::Tuple{<: HomogeneousPoint}) = homogeneous_point_to_indices(p[1])

function homogeneous_point_to_indices(p::HomogeneousPoint{T,3}) where T
    if  p.coords[end] == 1
        return round(Int,  p.coords[2]), round(Int, p.coords[1])
    else
        return round(Int,  p.coords[2] / p.coords[end]), round(Int, p.coords[1] / p.coords[end])
    end
end

function homogeneous_point_to_indices(p::HomogeneousPoint)
    if  p.coords[end] == 1
        return round.(Int, p.coords)
    else
        return round.(Int, p.coords ./ p.coords[end])
    end
end
