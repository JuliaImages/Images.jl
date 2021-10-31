module Images

export HomogeneousPoint

using StaticArrays

using Reexport
@reexport using ImageCore
@reexport using ImageBase

@reexport using FileIO: load, save
import Graphics # TODO: eliminate this direct dependency
using StatsBase  # TODO: eliminate this dependency
using IndirectArrays, ImageCore.MappedArrays

# TODO: can we get rid of these definitions?
const NumberLike = Union{Number,AbstractGray}
const RealLike = Union{Real,AbstractGray}

const is_little_endian = ENDIAN_BOM == 0x04030201 # CHECKME(johnnychen94): is this still used?

@reexport using ImageTransformations
@reexport using ImageAxes
@reexport using ImageMetadata
@reexport using ImageFiltering
@reexport using ImageMorphology
@reexport using ImageDistances
@reexport using ImageContrastAdjustment
@reexport using ImageQualityIndexes
@reexport using IntegralArrays
@reexport using IntegralArrays.IntervalSets.EllipsisNotation

# Non-exported symbol bindings to ImageShow so that we can use, e.g., `Images.gif`
import ImageShow: play, explore, gif

# While we are bridging the old API and the new API in ImageContrastAdjustment
# we need to import these functions because we make new definitions for them
# in deprecations.jl
import ImageContrastAdjustment: build_histogram, adjust_histogram, adjust_histogram!

using TiledIteration: EdgeIterator

# TODO(johnnychen94): (v1.0.0) remove these entry points
# Entry points that isn't used by JuliaImages at all (except for deprecations)
# They used to be accessible by, e.g., `Images.metadata`
import .Colors: Fractional
import FileIO: metadata
import Graphics: Point

include("compat.jl")
include("misc.jl")
include("labeledarrays.jl")
include("algorithms.jl")
include("deprecations.jl")
include("corner.jl")
include("edge.jl")

export
    # types
    ColorizedArray,
    Percentile,

    # macros
    @test_approx_eq_sigma_eps,

    # algorithms
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
    imgaussiannoise,
    otsu_threshold,
    yen_threshold,

    #Exposure
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
    entropy,
    orientation,
    phase,
    thin_edges,
    thin_edges_subpix,
    thin_edges_nonmaxsup,
    thin_edges_nonmaxsup_subpix,
    canny,
    gaussian_pyramid,

    # phantoms
    shepp_logan

"""
Constructors, conversions, and traits:

    - Construction: use constructors of specialized packages, e.g., `AxisArray`, `ImageMeta`, etc.
    - "Conversion": `colorview`, `channelview`, `rawview`, `normedview`, `permuteddimsview`
    - Traits: `pixelspacing`, `sdims`, `timeaxis`, `timedim`, `spacedirections`

Contrast/coloration:

    - `clamp01`, `clamp01nan`, `scaleminmax`, `colorsigned`, `scalesigned`

Algorithms:

    - Reductions: `maxfinite`, `maxabsfinite`, `minfinite`, `meanfinite`, `IntegralArray`, `gaussian_pyramid`
    - Resizing: `restrict`, `imresize` (not yet exported)
    - Filtering: `imfilter`, `imfilter!`, `mapwindow`, `imROF`, `padarray`
    - Filtering kernels: `Kernel.` or `KernelFactors.`, followed by `ando[345]`, `guassian`, `Laplacian', `DoG`, `prewitt`, `sobel`, etc.
    - Exposure : `imhist`, `histeq`, `adjust_gamma`, `histmatch`, `imadjustintensity`, `imstretch`, `imcomplement`, `clahe`, `cliphist`
    - Gradients: `backdiffx`, `backdiffy`, `forwarddiffx`, `forwarddiffy`, `imgradients`
    - Edge detection: `imedge`, `imgradients`, `thin_edges`, `magnitude`, `phase`, `magnitudephase`, `orientation`, `canny`
    - Corner detection: `imcorner`,`imcorner_subpixel`, `harris`, `shi_tomasi`, `kitchen_rosenfeld`, `meancovs`, `gammacovs`, `fastcorners`
    - Blob detection: `blob_LoG`, `findlocalmaxima`, `findlocalminima`
    - Morphological operations: `dilate`, `erode`, `closing`, `opening`, `tophat`, `bothat`, `morphogradient`, `morpholaplace`, `feature_transform`, `distance_transform`, `convexhull`
    - Connected components: `label_components`, `component_boxes`, `component_lengths`, `component_indices`, `component_subscripts`, `component_centroids`

Test images and phantoms (see also TestImages.jl):

    - `shepp_logan`
"""
Images

end
