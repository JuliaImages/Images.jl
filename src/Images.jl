__precompile__(true)  # because of ImageAxes/ImageMeta

module Images

if VERSION >= v"0.6.0-dev.1024"
    import Base.Iterators.take
else
    import Base.take
end
import Base.Order: Ordering, ForwardOrdering, ReverseOrdering
import Base: ==, .==, +, -, *, /, .+, .-, .*, ./, .^, .<, .>
import Base: abs, atan2, clamp, convert, copy, copy!, ctranspose, delete!, done,
             eltype, fft, get, getindex, haskey, hypot,
             ifft, imag, length, linearindexing, map, map!, maximum, mimewritable,
             minimum, next, ndims, one, parent, permutedims, real, reinterpret,
             reshape, resize!,
             setindex!, show, showcompact, similar, size, slice, sqrt, squeeze,
             start, strides, sub, sum, write, zero
if VERSION < v"0.5.0-dev+4490"
    import Base: float32, float64
else
    export float32, float64
end
using Base: depwarn

using Compat
import Compat.view

# "deprecated imports" are below

using Reexport
@reexport using FixedPointNumbers
@reexport using Colors
using ColorVectorSpace, FileIO
export load, save
import Colors: Fractional, red, green, blue
typealias AbstractGray{T}                    Color{T,1}
typealias TransparentRGB{C<:AbstractRGB,T}   TransparentColor{C,T,4}
typealias TransparentGray{C<:AbstractGray,T} TransparentColor{C,T,2}
typealias NumberLike                         Union{Number,AbstractGray}
typealias RealLike                           Union{Real,AbstractGray}
import Graphics
import Graphics: width, height, Point
using StatsBase  # TODO: eliminate this dependency
using IndirectArrays, MappedArrays

const is_little_endian = ENDIAN_BOM == 0x04030201

@reexport using ImageCore
@reexport using ImageAxes
@reexport using ImageMetadata
@reexport using ImageFiltering

using ImageMetadata: ImageMetaAxis

using Base.Cartesian  # TODO: delete this

include("map-deprecated.jl")
include("overlays-deprecated.jl")
include("labeledarrays.jl")
include("algorithms.jl")
include("exposure.jl")
include("connected.jl")
include("edge.jl")
include("showmime.jl")
include("corner.jl")
include("distances.jl")
include("deprecated.jl")

export # types
    ColorizedArray,

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

    #Exposure
    imhist,
    histeq,
    adjust_gamma,
    histmatch,
    clahe,
    imcomplement,
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

    # distances
    hausdorff_distance,

    # phantoms
    shepp_logan

_length(A::AbstractArray) = length(linearindices(A))
_length(A) = length(A)

"""
`Images` is a package for representing and processing images.

Constructors, conversions, and traits:

    - Construction: `Image`, `ImageCmap`, `grayim`, `colorim`, `convert`, `copyproperties`, `shareproperties`
    - Traits: `colordim`, `colorspace`, `coords_spatial`, `data`, `isdirect`, `isxfirst`, `isyfirst`, `pixelspacing`, `properties`, `sdims`, `spacedirections`, `spatialorder`, `storageorder`, `timedim`
    - Size-related traits: `height`, `nchannels`, `ncolorelem`, `nimages`, `size_spatial`, `width`, `widthheight`
    - Trait assertions: `assert_2d`, `assert_scalar_color`, `assert_timedim_last`, `assert_xfirst`, `assert_yfirst`
    - Indexing operations: `getindexim`, `sliceim`, `subim`
    - Conversions: `convert`, `raw`, `reinterpret`, `separate`

Contrast/coloration:

    - `MapInfo`: `MapNone`, `BitShift`, `ClampMinMax`, `ScaleMinMax`, `ScaleAutoMinMax`, `sc`, etc.

Algorithms:

    - Reductions: `maxfinite`, `maxabsfinite`, `minfinite`, `meanfinite`, `sad`, `ssd`, `integral_image`, `boxdiff`, `gaussian_pyramid`
    - Resizing: `restrict`, `imresize` (not yet exported)
    - Filtering: `imfilter`, `imfilter_fft`, `imfilter_gaussian`, `imfilter_LoG`, `imROF`, `ncc`, `padarray`
    - Filtering kernels: `ando[345]`, `guassian2d`, `imaverage`, `imdog`, `imlaplacian`, `prewitt`, `sobel`
    - Exposure : `imhist`, `histeq`, `adjust_gamma`, `histmatch`, `imadjustintensity`, `imstretch`, `imcomplement`, `clahe`, `cliphist`
    - Gradients: `backdiffx`, `backdiffy`, `forwarddiffx`, `forwarddiffy`, `imgradients`
    - Edge detection: `imedge`, `imgradients`, `thin_edges`, `magnitude`, `phase`, `magnitudephase`, `orientation`, `canny`
    - Corner detection: `imcorner`, `harris`, `shi_tomasi`, `kitchen_rosenfeld`, `meancovs`, `gammacovs`, `fastcorners`
    - Blob detection: `blob_LoG`, `findlocalmaxima`, `findlocalminima`
    - Morphological operations: `dilate`, `erode`, `closing`, `opening`, `tophat`, `bothat`, `morphogradient`, `morpholaplace`
    - Connected components: `label_components`, `component_boxes`, `component_lengths`, `component_indices`, `component_subscripts`, `component_centroids`
    - Interpolation: `bilinear_interpolation`

Test images and phantoms (see also TestImages.jl):

    - `shepp_logan`
"""
Images

end
