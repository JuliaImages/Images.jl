module Images

import Base.Order: Ordering, ForwardOrdering, ReverseOrdering
import Base.Graphics: width, height
import Base: convert, copy, copy!, ctranspose, delete!, eltype, float32, float64,
             get, getindex, haskey, length, maximum, minimum, ndims, one, parent,
             permutedims, reinterpret, scale, scale!, setindex!, show, similar,
             size, slice, squeeze, strides, sub, sum, write, writemime, zero

using Color, FixedPointNumbers
import Color: Fractional
import FixedPointNumbers: ufixed8, ufixed10, ufixed12, ufixed14, ufixed16

if VERSION.minor < 3
    using Cartesian
else
    using Base.Cartesian
    include("compatibility/forcartesian.jl")
end

# if isdefined(module_parent(Images), :Grid)
#     import ..Grid.restrict
# end

const is_little_endian = ENDIAN_BOM == 0x04030201

include("colortypes.jl")
using .ColorTypes

include("core.jl")
include("iterator.jl")
const have_imagemagick = include("ioformats/libmagickwand.jl")
@osx_only include("ioformats/OSXnative.jl")
include("io.jl")
include("scaling.jl")
include("labeledarrays.jl")
include("uint32color.jl")
include("algorithms.jl")
include("connected.jl")

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
    Gray,
    GrayAlpha,
    ARGB,
    BGRA,
    RGBA,
    RGB1,
    RGB4,

    AbstractImage,
    AbstractImageDirect,
    AbstractImageIndexed,
    Image,
    ImageCmap,
    BitShift,
    ClipMin,
    ClipMax,
    ClipMinMax,
    LabeledArray,
    Overlay,
    OverlayImage,
    ScaleAutoMinMax,
    ScaleInfo,
    ScaleMinMax,
    ScaleNone,
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
    ncolorelem,
    nimages,
    pixelspacing,
    properties,
    refim,
    rerange!,
    reslice!,
    restrict,
    sdims,
    size_spatial,
    share,
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
    alphaval,
    blueval,
    greenval,
    redval,
    red,
    green,
    blue,
    hsi2rgb,
    imadjustintensity,
    indexedcolor,
    lut,
    ntsc2rgb,
    rgb2gray,
    rgb2hsi,
    rgb2ntsc,
    rgb2ycbcr,
    separate,
    uint32color,
    uint32color!,
    ycbcr2rgb,
    
    # Scaling of intensity
    climdefault,
    float32sc,
    float64sc,
    sc,
    scale,
    scaleinfo,
    scaleminmax,
    scalesigned,
    uint8sc,
    uint16sc,
    uint32sc,

    # algorithms
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
    imfilter_fft,
    imfilter_gaussian,
    imfilter_gaussian!,
    imfilter_LoG,
    imgaussiannoise,
    imlaplacian,
    imlineardiffusion,
    imlog,
    imROF,
    imstretch,
#     imthresh,
    label_components,
    label_components!,
    ncc,
    padarray,
    prewitt,
    sad,
    sadn,
    sobel,
    ssd,
    ssdn,

    # phantoms
    shepp_logan


@deprecate cairoRGB     uint32color!
@deprecate refim        getindexim
@deprecate scaledefault climdefault
@deprecate scaleminmax  ScaleMinMax
@deprecate float32sc    float32
@deprecate float64sc    float64
@deprecate uint8sc      ufixed8sc
@deprecate uint16sc(img)  ufixedsc(Ufixed16, img)
@deprecate ClipMin      ClampMin
@deprecate ClipMax      ClampMax
@deprecate ClipMinMax   ClampMinMax
@deprecate climdefault(img) zero(eltype(img)), one(eltype(img))

if VERSION < v"0.3-"
  __init__()
end

end
