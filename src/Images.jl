module Images

using Color

importall Base

include("core.jl")
include("config.jl")
include("iterator.jl")
include("io.jl")
include("display.jl")
include("algorithms.jl")

export # types
    AbstractImage,
    AbstractImageDirect,
    AbstractImageIndexed,
    Image,
    ImageCmap,
    ClipMin,
    ClipMax,
    ClipMinMax,
    ScaleMinMax,
    ScaleNone,

    # constants
    palette_fire,
    palette_gray32,
    palette_gray64,
    palette_rainbow,

    # core functions
    assert2d,
    assert_scalar_color,
    assert_xfirst,
    colordim,
    colorspace,
    coords_spatial,
    data,
    isdirect,
    isxfirst,
    isyfirst,
    limits,
    nimages,
    pixelspacing,
    refim,
    sdims,
    size_spatial,
    share,
    sliceim,
    spatialorder,
    spatialpermutation,
    spatialproperties,
    storageorder,
    subim,
    timedim,
    widthheight,
    
    # io functions
    add_image_file_format,
    imread,
    imwrite,
    
    # display functions
    display,
    ftshow,
    imshow,
    update,
    
    # iterator functions
    first_index,
    iterate_spatial,
    parent,
    
    # color-related functions
    alphaval,
    blueval,
    greenval,
    redval,
    hsi2rgb,
    imadjustintensity,
    indexedcolor,
    lut,
    ntsc2rgb,
    rgb2gray,
    rgb2hsi,
    rgb2ntsc,
    rgb2ycbcr,
    ycbcr2rgb,
    
    # algorithms
    backdiffx,
    backdiffy,
    forwarddiffx,
    forwarddiffy,
    gaussian2d,
    imaverage,
    imcomplement,
    imdog,
    imedge,
    imfilter,
    imgaussiannoise,
    imlaplacian,
    imlineardiffusion,
    imlog,
    imROF,
    imstretch,
    imthresh,
    ncc,
    prewitt,
    sad,
    sadn,
    sc,
    scale,
    scaleinfo,
    scaleminmax,
    sobel,
    ssd,
    ssdn

end
