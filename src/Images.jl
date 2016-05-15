__precompile__(true)

module Images

VERSION >= v"0.4.0" && import Base.take
import Base.Order: Ordering, ForwardOrdering, ReverseOrdering
import Base: ==, .==, +, -, *, /, .+, .-, .*, ./, .^, .<, .>
import Base: atan2, clamp, convert, copy, copy!, ctranspose, delete!, done,
             eltype, fft, float32, float64, get, getindex, haskey, hypot,
             ifft, length, linearindexing, map, map!, maximum, mimewritable,
             minimum, next, ndims, one, parent, permutedims, reinterpret,
             reshape, resize!,
             setindex!, show, showcompact, similar, size, slice, sqrt, squeeze,
             start, strides, sub, sum, write, writemime, zero
# "deprecated imports" are below

using Colors, ColorVectorSpace, FixedPointNumbers, FileIO
import Colors: Fractional, red, green, blue
typealias AbstractGray{T}                    Color{T,1}
typealias TransparentRGB{C<:AbstractRGB,T}   TransparentColor{C,T,4}
typealias TransparentGray{C<:AbstractGray,T} TransparentColor{C,T,2}
using Graphics
import Graphics: width, height, Point
import FixedPointNumbers: ufixed8, ufixed10, ufixed12, ufixed14, ufixed16
using Compat; import Compat.String
import Compat.String

using Base.Cartesian
include("compatibility/forcartesian.jl")

# if isdefined(module_parent(Images), :Grid)
#     import ..Grid.restrict
# end

const is_little_endian = ENDIAN_BOM == 0x04030201
immutable TypeConst{N} end  # for passing compile-time constants to functions

include("core.jl")
include("map.jl")
include("overlays.jl")
include("labeledarrays.jl")
include("algorithms.jl")
include("connected.jl")
include("edge.jl")
include("writemime.jl")
include("corner.jl")
include("distances.jl")


function precompile()
    for T in (UInt8, UInt16, Int, Float32, Float64)
        Tdiv = typeof(one(T)/2)
        for N = 2:3
            precompile(restrict!, (Array{Tdiv, N}, Array{T,N}, Int))
            precompile(imfilter, (Array{T,N}, Array{Float64,N}))
            precompile(imfilter, (Array{T,N}, Array{Float64,2}))
            precompile(imfilter, (Array{T,N}, Array{Float32,N}))
            precompile(imfilter, (Array{T,N}, Array{Float32,2}))
        end
    end
    for T in (Float32, Float64)
        for N = 2:3
            precompile(_imfilter_gaussian!, (Array{T,N}, Vector{Float64}))
            precompile(_imfilter_gaussian!, (Array{T,N}, Vector{Int}))
            precompile(imfilter_gaussian_no_nans!, (Array{T,N}, Vector{Float64}))
            precompile(imfilter_gaussian_no_nans!, (Array{T,N}, Vector{Int}))
            precompile(fft, (Array{T,N},))
        end
    end
    for T in (Complex{Float32}, Complex{Float64})
        for N = 2:3
            precompile(ifft, (Array{T,N},))
        end
    end
end

export # types
    AbstractImage,
    AbstractImageDirect,
    AbstractImageIndexed,
    Image,
    ImageCmap,
    BitShift,
    ClampMin,
    ClampMax,
    ClampMinMax,
    Clamp,
    Clamp01NaN,
    LabeledArray,
    MapInfo,
    MapNone,
    Overlay,
    OverlayImage,
    ScaleAutoMinMax,
    ScaleMinMax,
    ScaleMinMaxNaN,
    ScaleSigned,
    SliceData,

    # macros
    @test_approx_eq_sigma_eps,

    # constants
    palette_fire,
    palette_gray32,
    palette_gray64,
    palette_rainbow,

    # core functions
    assert2d,
    assert_scalar_color,
    assert_timedim_last,
    assert_xfirst,
    assert_yfirst,
    colordim,
    colorspace,
    coords,
    coords_spatial,
    copyproperties,
    data,
    dimindex,
    dimindexes,
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

    # iterator functions
    first_index,
    iterate_spatial,
    parent,

    # color-related functions
    imadjustintensity,
    indexedcolor,
    lut,
    separate,
    uint32color,
    uint32color!,

    # Scaling of intensity
    sc,
    scale,
    mapinfo,
    uint8sc,
    uint16sc,
    uint32sc,
    ufixed8sc,
    ufixedsc,

    # flip dimensions
    flipx,
    flipy,
    flipz,

    # algorithms
    ando3,
    ando4,
    ando5,
    backdiffx,
    backdiffy,
    dilate,
    erode,
    extrema_filter,
    opening,
    closing,
    tophat,
    bothat,
    morphogradient,
    morpholaplace,
    forwarddiffx,
    forwarddiffy,
    gaussian2d,
    imaverage,
    imcomplement,
    imcorner,
    imdog,
    imedge,
    imfilter,
    imfilter!,
    imfilter_fft,
    imfilter_gaussian,
    imfilter_gaussian!,
    imfilter_LoG,
    blob_LoG,
    findlocalmaxima,
    findlocalminima,
    imgaussiannoise,
    imgradients,
    imhist,
    imlaplacian,
    imlineardiffusion,
    imlog,
    imROF,
    imstretch,
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
    prewitt,
    sad,
    sadn,
    sobel,
    ssd,
    ssdn,
    thin_edges,
    thin_edges_subpix,
    thin_edges_nonmaxsup,
    thin_edges_nonmaxsup_subpix,

    # distances
    hausdorff_distance,

    # phantoms
    shepp_logan


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

    - `MapInfo`: `MapNone`, `BitShift`, `ClampMinMax`, `ScaleMinMax`, `ScaleAutoMinMax`, etc.
    - `imadjustintensity`, `sc`, `imstretch`, `imcomplement`


Algorithms:

    - Reductions: `maxfinite`, `maxabsfinite`, `minfinite`, `meanfinite`, `sad`, `ssd`
    - Resizing: `restrict`, `imresize` (not yet exported)
    - Filtering: `imfilter`, `imfilter_fft`, `imfilter_gaussian`, `imfilter_LoG`, `imROF`, `ncc`, `padarray`
    - Filtering kernels: `ando[345]`, `guassian2d`, `imaverage`, `imdog`, `imlaplacian`, `prewitt`, `sobel`
    - Exposure : `imhist`
    - Gradients: `backdiffx`, `backdiffy`, `forwarddiffx`, `forwarddiffy`, `imgradients`
    - Edge detection: `imedge`, `imgradients`, `thin_edges`, `magnitude`, `phase`, `magnitudephase`, `orientation`
    - Corner detection: `imcorner`
    - Blob detection: `blob_LoG`, `findlocalmaxima`, `findlocalminima`
    - Morphological operations: `dilate`, `erode`, `closing`, `opening`, `tophat`, `bothat`, `morphogradient`, `morpholaplace`
    - Connected components: `label_components`, `component_boxes`, `component_lengths`, `component_indices`, `component_subscripts`, `component_centroids`

Test images and phantoms (see also TestImages.jl):

    - `shepp_logan`
"""
Images

import FileIO: load, save
@deprecate imread(filename; kwargs...) load(filename; kwargs...)
@deprecate imwrite(img, filename; kwargs...) save(filename, img; kwargs...)
export load, save

function limits(img)
    Base.depwarn("limits is deprecated, all limits are (0,1)", :limits)
    oldlimits(img)
end

end
