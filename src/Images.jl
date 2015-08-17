VERSION >= v"0.4.0-dev+6521" && __precompile__(@unix? true : false)

module Images

import Base.Order: Ordering, ForwardOrdering, ReverseOrdering
import Base: ==, .==, +, -, *, /, .+, .-, .*, ./, .^, .<, .>
import Base: atan2, clamp, convert, copy, copy!, ctranspose, delete!, done, eltype,
             fft, float32, float64, get, getindex, haskey, hypot, ifft, length, map, map!,
             maximum, mimewritable, minimum, next, ndims, one, parent, permutedims, reinterpret,
             setindex!, show, showcompact, similar, size, slice, sqrt, squeeze,
             start, strides, sub, sum, write, writemime, zero
# "deprecated imports" are below

using Colors, ColorVectorSpace, FixedPointNumbers, Compat
import Colors: Fractional, red, green, blue
typealias TransparentRGB{C<:AbstractRGB,T}   Transparent{C,T,4}
typealias TransparentGray{C<:AbstractGray,T} Transparent{C,T,2}
if VERSION < v"0.4.0-dev+3275"
    using Base.Graphics
    import Base.Graphics: width, height, Point
else
    using Graphics
    import Graphics: width, height, Point
end
import FixedPointNumbers: ufixed8, ufixed10, ufixed12, ufixed14, ufixed16

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
const have_imagemagick = include("ioformats/libmagickwand.jl")
@osx_only include("ioformats/OSXnative.jl")
include("io.jl")
include("labeledarrays.jl")
include("algorithms.jl")
include("connected.jl")
include("edge.jl")

__init__() = LibMagick.init()

function precompile()
    for T in (Uint8, Uint16, Int, Float32, Float64)
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
    LabeledArray,
    MapInfo,
    MapNone,
    Overlay,
    OverlayImage,
    ScaleAutoMinMax,
    ScaleMinMax,
    ScaleSigned,
    SliceData,

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

    # io functions
    add_image_file_format,
    imread,
    imwrite,
    loadformat,

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

    # algorithms
    ando3,
    ando4,
    ando5,
    backdiffx,
    backdiffy,
    dilate,
    erode,
    opening,
    closing,
    forwarddiffx,
    forwarddiffy,
    gaussian2d,
    imaverage,
    imcomplement,
    imdog,
    imedge,
    imfilter,
    imfilter!,
    imfilter_fft,
    imfilter_gaussian,
    imfilter_gaussian!,
    imfilter_LoG,
    imgaussiannoise,
    imgradients,
    imlaplacian,
    imlineardiffusion,
    imlog,
    imROF,
    imstretch,
#     imthresh,
    label_components,
    label_components!,
    magnitude,
    magnitude_phase,
    meanfinite,
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

    # phantoms
    shepp_logan

export # Deprecated exports
    ClipMin,
    ClipMax,
    ClipMinMax,
    ScaleInfo,
    climdefault,
    float32sc,
    float64sc,
    ntsc2rgb,
    rgb2gray,
    rgb2ntsc,
    scaleinfo,
    scaleminmax,
    scalesigned


import Base: scale, scale!  # delete when deprecations are removed
@deprecate scaleminmax  ScaleMinMax
@deprecate scaleminmax(img::AbstractArray, min::Real, max::Real)  ScaleMinMax(RGB24, img, min, max)
@deprecate float32sc    float32
@deprecate float64sc    float64
@deprecate uint8sc      ufixed8sc
@deprecate uint16sc(img)  ufixedsc(Ufixed16, img)
@deprecate ClipMin      ClampMin
@deprecate ClipMax      ClampMax
@deprecate ClipMinMax   ClampMinMax
@deprecate climdefault(img) zero(eltype(img)), one(eltype(img))
@deprecate ScaleMinMax{T<:Real}(img::AbstractArray{T}, mn, mx) ScaleMinMax(Ufixed8, img, mn, mx)
@deprecate ScaleMinMax{T<:Color}(img::AbstractArray{T}, mn, mx) ScaleMinMax(RGB{Ufixed8}, img, mn, mx)
@deprecate scaleinfo    mapinfo
@deprecate scale(mapi::MapInfo, A) map(mapi, A)                # delete imports above when eliminated
@deprecate scale!(dest, mapi::MapInfo, A) map!(mapi, dest, A)  #   "
@deprecate copy(A::AbstractArray, B::AbstractArray) copyproperties(A, B)
@deprecate share(A::AbstractArray, B::AbstractArray) shareproperties(A, B)


const ScaleInfo = MapInfo  # can't deprecate types?

end
