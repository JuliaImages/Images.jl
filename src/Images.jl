__precompile__(true)  # because of ImageAxes/ImageMeta

module Images

import Base.Iterators.take
import Base: +, -, *
import Base: abs, atan2, clamp, convert, copy, copy!, ctranspose, delete!,
             eltype, fft, get, getindex, haskey, hypot,
             ifft, imag, length, linearindexing, map, map!, maximum, mimewritable,
             minimum, ndims, one, parent, permutedims, real, reinterpret,
             reshape, resize!,
             setindex!, show, showcompact, similar, size, sqrt, squeeze,
             strides, sum, write, zero

export float32, float64
export HomogeneousPoint

using Base: depwarn
using Base.Order: Ordering, ForwardOrdering, ReverseOrdering

using Compat
using StaticArrays

# "deprecated imports" are below

using Reexport
@reexport using FixedPointNumbers
@reexport using Colors
using ColorVectorSpace, FileIO
export load, save
import Colors: Fractional, red, green, blue
const AbstractGray{T}                    = Color{T,1}
const TransparentRGB{C<:AbstractRGB,T}   = TransparentColor{C,T,4}
const TransparentGray{C<:AbstractGray,T} = TransparentColor{C,T,2}
const NumberLike = Union{Number,AbstractGray}
const RealLike = Union{Real,AbstractGray}
import Graphics
import Graphics: width, height, Point
using StatsBase  # TODO: eliminate this dependency
using IndirectArrays, MappedArrays
using Compat.TypeUtils

const is_little_endian = ENDIAN_BOM == 0x04030201

@reexport using ImageCore
@reexport using ImageTransformations
@reexport using ImageAxes
@reexport using ImageMetadata
@reexport using ImageFiltering
@reexport using ImageMorphology
@reexport using ImageDistances

using ImageMetadata: ImageMetaAxis
import ImageMorphology: dilate, erode
import ImageTransformations: restrict
using TiledIteration: EdgeIterator
using CoordinateTransformations: Translation

using Base.Cartesian  # TODO: delete this

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

include("labeledarrays.jl")
include("algorithms.jl")
include("exposure.jl")
include("connected.jl")
include("edge.jl")
include("showmime.jl")
include("juno.jl")
include("corner.jl")
include("bwdist.jl")
using .FeatureTransform
include("convexhull.jl")
include("deprecated.jl")

export # types
    BlobLoG,
    ColorizedArray,
    Percentile,

    # macros
    @test_approx_eq_sigma_eps,

    # core functions
    assert2d,
    assert_scalar_color,
    assert_timedim_last,
    assert_xfirst,
    assert_yfirst,
    colordim,
    colorspace,
    coords_spatial,
    copyproperties,
    data,
    getindexim,
    grayim,
    colorim,
    height,
    isdirect,
    isxfirst,
    isyfirst,
    limits,
    maxabsfinite,
    maxfinite,
    minfinite,
    nchannels,
    ncolorelem,
    nimages,
    pixelspacing,
    properties,
    rerange!,
    reslice!,
    restrict,
    sdims,
    size_spatial,
    shareproperties,
    sliceim,
    spacedirections,
    spatialorder,
    spatialpermutation,
    spatialproperties,
    storageorder,
    subim,
    timedim,
    width,
    widthheight,
    raw,

    # color-related functions
    separate,

    # Scaling of intensity
    sc,
    scale,
    mapinfo,
    ufixed8sc,
    ufixedsc,

    # algorithms
    backdiffx,
    backdiffy,
    dilate,
    erode,
    opening,
    closing,
    tophat,
    bothat,
    morphogradient,
    morpholaplace,
    forwarddiffx,
    forwarddiffy,
    imcorner,
    imcorner_subpixel,
    corner2subpixel,
    harris,
    shi_tomasi,
    kitchen_rosenfeld,
    fastcorners,
    meancovs,
    gammacovs,
    imedge,  # TODO: deprecate?
    imfilter_LoG,
    blob_LoG,
    findlocalmaxima,
    findlocalminima,
    imgaussiannoise,
    imlineardiffusion,
    imROF,
    feature_transform,
    distance_transform,
    convexhull,
    otsu_threshold,
    yen_threshold,
    clearborder,

    #Exposure
    complement,
    imhist,
    histeq,
    adjust_gamma,
    histmatch,
    clahe,
    imadjustintensity,
    imstretch,
    cliphist,


#     imthresh,
    label_components,
    label_components!,
    component_boxes,
    component_lengths,
    component_indices,
    component_subscripts,
    component_centroids,
    magnitude,
    magnitude_phase,
    meanfinite,
    entropy,
    ncc,
    orientation,
    padarray,
    phase,
    sad,
    sadn,
    ssd,
    ssdn,
    thin_edges,
    thin_edges_subpix,
    thin_edges_nonmaxsup,
    thin_edges_nonmaxsup_subpix,
    canny,
    integral_image,
    boxdiff,
    bilinear_interpolation,
    gaussian_pyramid,

    # phantoms
    shepp_logan

_length(A::AbstractArray) = length(linearindices(A))
_length(A) = length(A)

"""
Constructors, conversions, and traits:

    - Construction: use constructors of specialized packages, e.g., `AxisArray`, `ImageMeta`, etc.
    - "Conversion": `colorview`, `channelview`, `rawview`, `normedview`, `permuteddimsview`
    - Traits: `pixelspacing`, `sdims`, `timeaxis`, `timedim`, `spacedirections`

Contrast/coloration:

    - `clamp01`, `clamp01nan`, `scaleminmax`, `colorsigned`, `scalesigned`

Algorithms:

    - Reductions: `maxfinite`, `maxabsfinite`, `minfinite`, `meanfinite`, `sad`, `ssd`, `integral_image`, `boxdiff`, `gaussian_pyramid`
    - Resizing: `restrict`, `imresize` (not yet exported)
    - Filtering: `imfilter`, `imfilter!`, `imfilter_LoG`, `mapwindow`, `imROF`, `padarray`
    - Filtering kernels: `Kernel.` or `KernelFactors.`, followed by `ando[345]`, `guassian2d`, `imaverage`, `imdog`, `imlaplacian`, `prewitt`, `sobel`
    - Exposure : `imhist`, `histeq`, `adjust_gamma`, `histmatch`, `imadjustintensity`, `imstretch`, `imcomplement`, `clahe`, `cliphist`
    - Gradients: `backdiffx`, `backdiffy`, `forwarddiffx`, `forwarddiffy`, `imgradients`
    - Edge detection: `imedge`, `imgradients`, `thin_edges`, `magnitude`, `phase`, `magnitudephase`, `orientation`, `canny`
    - Corner detection: `imcorner`,`imcorner_subpixel`, `harris`, `shi_tomasi`, `kitchen_rosenfeld`, `meancovs`, `gammacovs`, `fastcorners`
    - Blob detection: `blob_LoG`, `findlocalmaxima`, `findlocalminima`
    - Morphological operations: `dilate`, `erode`, `closing`, `opening`, `tophat`, `bothat`, `morphogradient`, `morpholaplace`, `feature_transform`, `distance_transform`, `convexhull`
    - Connected components: `label_components`, `component_boxes`, `component_lengths`, `component_indices`, `component_subscripts`, `component_centroids`
    - Interpolation: `bilinear_interpolation`

Test images and phantoms (see also TestImages.jl):

    - `shepp_logan`
"""
Images

end
