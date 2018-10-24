VERSION < v"0.7.0-beta2.199" && __precompile__()

module Images

import Base.Iterators.take
import Base: +, -, *
import Base: abs, atan, clamp, convert, copy, copy!, delete!,
             eltype, get, getindex, haskey, hypot,
             imag, length, map, map!, maximum,
             minimum, ndims, one, parent, permutedims, real, reinterpret,
             reshape, resize!,
             setindex!, show, similar, size, sqrt,
             strides, sum, write, zero

export float32, float64
export HomogeneousPoint

using Base: depwarn
using Base.Order: Ordering, ForwardOrdering, ReverseOrdering

using StaticArrays
using Base64: Base64EncodePipe

# CHECKME: use this or follow deprecation and substitute?
using SparseArrays: findnz

# "deprecated imports" are below

using Reexport
@reexport using FixedPointNumbers
@reexport using Colors
using ColorVectorSpace, FileIO
export load, save
import Colors: Fractional, red, green, blue
import Graphics
import Graphics: width, height, Point
using StatsBase  # TODO: eliminate this dependency
using IndirectArrays, MappedArrays

# TODO: can we get rid of these definitions?
const NumberLike = Union{Number,AbstractGray}
const RealLike = Union{Real,AbstractGray}

const is_little_endian = ENDIAN_BOM == 0x04030201

@reexport using ImageCore
@reexport using ImageTransformations
@reexport using ImageAxes
@reexport using ImageMetadata
@reexport using ImageFiltering
@reexport using ImageMorphology
@reexport using ImageDistances

import ImageShow
using ImageMetadata: ImageMetaAxis
import ImageMorphology: dilate, erode
import ImageTransformations: restrict
using TiledIteration: EdgeIterator

using Base.Cartesian  # TODO: delete this

include("misc.jl")
include("labeledarrays.jl")
include("algorithms.jl")
include("exposure.jl")
include("connected.jl")
include("corner.jl")
include("edge.jl")
include("bwdist.jl")
using .FeatureTransform
include("convexhull.jl")

export
    # types
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
    Equalization,
    complement,
    imhist,
    histeq,
    build_histogram,
    adjust_histogram,
    adjust_histogram!,
    adjust_gamma,
    histmatch,
    clahe,
    imadjustintensity,
    imstretch,
    cliphist,

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

_length(A::AbstractArray) = length(eachindex(A))
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
