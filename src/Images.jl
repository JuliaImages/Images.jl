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
@reexport using ImageCore
if isdefined(ImageCore, :permuteddimsview)
    export permuteddimsview
end

using FileIO
export load, save
import .Colors: Fractional
import Graphics
import Graphics: width, height, Point
using StatsBase  # TODO: eliminate this dependency
using IndirectArrays, ImageCore.MappedArrays

# TODO: can we get rid of these definitions?
const NumberLike = Union{Number,AbstractGray}
const RealLike = Union{Real,AbstractGray}

const is_little_endian = ENDIAN_BOM == 0x04030201

@reexport using ImageTransformations
@reexport using ImageAxes
@reexport using ImageMetadata
@reexport using ImageFiltering
@reexport using ImageMorphology
@reexport using ImageDistances
@reexport using ImageContrastAdjustment
@reexport using ImageQualityIndexes

if isdefined(ImageQualityIndexes, :assess_psnr)
    # deprecated since ImageQualityIndexes v0.1.4
    Base.@deprecate_binding psnr assess_psnr
    Base.@deprecate_binding ssim assess_ssim
else
    const psnr = ImageQualityIndexes.psnr
    export psnr
end

# Both ImageMetadata v0.9.0 and ImageAxes v0.6.0 deprecate the symbol data and
# this causes a name conflict
if isdefined(ImageMetadata, :arraydata)
    Base.@deprecate_binding data arraydata
end
# ImageMetadata < v0.9.0 compatibility
if !hasmethod(arraydata, (ImageMeta, ) )
    ImageAxes.arraydata(img::ImageMeta) = ImageMetadata.data(img)
end

# Non-exported symbol bindings to ImageShow
import ImageShow
if isdefined(ImageShow, :play)
    @doc (@doc ImageShow.play)
    const play = ImageShow.play
else
    play(args...; kwargs...) = error("The `Images.play` function requires ImageShow at least 0.3.0.")
end
if isdefined(ImageShow, :explore)
    @doc (@doc ImageShow.explore)
    const explore = ImageShow.explore
else
    explore(args...; kwargs...) = error("The `Images.explore` function requires ImageShow at least 0.3.0.")
end
if isdefined(ImageShow, :gif)
    @doc (@doc ImageShow.gif)
    const gif = ImageShow.gif
else
    gif(args...; kwargs...) = error("The `Images.gif` function requires ImageShow at least 0.3.0.")
end

# While we are bridging the old API and the new API in ImageContrastAdjustment
# we need to import these functions because we make new definitions for them
# in deprecations.jl
import ImageContrastAdjustment: build_histogram, adjust_histogram, adjust_histogram!

using ImageMetadata: ImageMetaAxis
import ImageMorphology: dilate, erode
import ImageTransformations: restrict
using TiledIteration: EdgeIterator

include("compat.jl")
include("misc.jl")
include("labeledarrays.jl")
include("algorithms.jl")
include("exposure.jl")
include("deprecations.jl")
include("corner.jl")
include("edge.jl")

export
    # types
    BlobLoG,
    ColorizedArray,
    Percentile,

    # macros
    @test_approx_eq_sigma_eps,

    # core functions
    assert_timedim_last,
    colordim,
    coords_spatial,
    copyproperties,
    data,
    height,
    maxabsfinite,
    maxfinite,
    minfinite,
    nimages,
    pixelspacing,
    properties,
    restrict,
    sdims,
    size_spatial,
    shareproperties,
    spacedirections,
    spatialorder,
    spatialproperties,
    timedim,
    width,
    widthheight,

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
    blob_LoG,
    findlocalmaxima,
    findlocalminima,
    imgaussiannoise,
    imlineardiffusion,
    imROF,
    otsu_threshold,
    yen_threshold,

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

    magnitude,
    magnitude_phase,
    meanfinite,
    entropy,
    orientation,
    padarray,
    phase,
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

    - Reductions: `maxfinite`, `maxabsfinite`, `minfinite`, `meanfinite`, `integral_image`, `boxdiff`, `gaussian_pyramid`
    - Resizing: `restrict`, `imresize` (not yet exported)
    - Filtering: `imfilter`, `imfilter!`, `mapwindow`, `imROF`, `padarray`
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
