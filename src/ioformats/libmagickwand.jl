module LibMagick

using Colors, FixedPointNumbers, ..ColorVectorSpace, Compat

import Base: error, size

export MagickWand,
    constituteimage,
    exportimagepixels!,
    getblob,
    getimagealphachannel,
    getimagecolorspace,
    getimagedepth,
    getnumberimages,
    importimagepixels,
    readimage,
    resetiterator,
    setimagecolorspace,
    setimagecompression,
    setimagecompressionquality,
    setimageformat,
    writeimage


# Find the library
depsfile = joinpath(dirname(@__FILE__),"..","..","deps","deps.jl")
versionfile = joinpath(dirname(@__FILE__),"..","..","deps","versioninfo.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("Images not properly installed. Please run Pkg.build(\"Images\") then restart Julia.")
end
if isfile(versionfile)
    include(versionfile)
end
<<<<<<< HEAD
<<<<<<< HEAD

const have_imagemagick = isdefined(:libwand)

# Initialize the library
if !have_imagemagick
    function __init__()
=======
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
const have_imagemagick = isdefined(:libwand)

# Initialize the library
function init()
    global libwand
    if have_imagemagick
        eval(:(ccall((:MagickWandGenesis, $libwand), Void, ())))
    else
<<<<<<< HEAD
>>>>>>> adcbb46... started removing files that go into FileIO
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
        warn("ImageMagick utilities not found. Install for more file format support.")
    end
end


# Constants
# Storage types
const CHARPIXEL = 1
const DOUBLEPIXEL = 2
const FLOATPIXEL = 3
const INTEGERPIXEL = 4
const SHORTPIXEL = 7
<<<<<<< HEAD
<<<<<<< HEAD
@compat IMStorageTypes = Union{UInt8, UInt16, UInt32, Float32, Float64}
storagetype(::Type{UInt8}) = CHARPIXEL
storagetype(::Type{UInt16}) = SHORTPIXEL
storagetype(::Type{UInt32}) = INTEGERPIXEL
=======
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
IMStorageTypes = Union(Uint8, Uint16, Uint32, Float32, Float64)
storagetype(::Type{Uint8}) = CHARPIXEL
storagetype(::Type{Uint16}) = SHORTPIXEL
storagetype(::Type{Uint32}) = INTEGERPIXEL
<<<<<<< HEAD
>>>>>>> adcbb46... started removing files that go into FileIO
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
storagetype(::Type{Float32}) = FLOATPIXEL
storagetype(::Type{Float64}) = DOUBLEPIXEL
storagetype{T<:Ufixed}(::Type{T}) = storagetype(FixedPointNumbers.rawtype(T))
storagetype{CV<:Colorant}(::Type{CV}) = storagetype(eltype(CV))

# Channel types
type ChannelType
<<<<<<< HEAD
<<<<<<< HEAD
    value::UInt32
=======
    value::Uint32
>>>>>>> adcbb46... started removing files that go into FileIO
=======
    value::Uint32
>>>>>>> 0611ac0... Revert "started rewriting SIF"
end
const UndefinedChannel = ChannelType(0x00000000)
const RedChannel = ChannelType(0x00000001)
const GrayChannel = ChannelType(0x00000001)
const CyanChannel = ChannelType(0x00000001)
const GreenChannel = ChannelType(0x00000002)
const MagentaChannel = ChannelType(0x00000002)
const BlueChannel = ChannelType(0x00000004)
const YellowChannel = ChannelType(0x00000004)
const AlphaChannel = ChannelType(0x00000008)
const MatteChannel = ChannelType(0x00000008)
const OpacityChannel = ChannelType(0x00000008)
const BlackChannel = ChannelType(0x00000020)
const IndexChannel = ChannelType(0x00000020)
const CompositeChannels = ChannelType(0x0000002F)
const TrueAlphaChannel = ChannelType(0x00000040)
const RGBChannels = ChannelType(0x00000080)
const GrayChannels = ChannelType(0x00000080)
const SyncChannels = ChannelType(0x00000100)
const AllChannels = ChannelType(0x7fffffff)
const DefaultChannels = ChannelType( (AllChannels.value | SyncChannels.value) &~ OpacityChannel.value )


# Image type
const IMType = ["BilevelType", "GrayscaleType", "GrayscaleMatteType", "PaletteType", "PaletteMatteType", "TrueColorType", "TrueColorMatteType", "ColorSeparationType", "ColorSeparationMatteType", "OptimizeType", "PaletteBilevelMatteType"]
const IMTypedict = Dict([(IMType[i], i) for i = 1:length(IMType)])

<<<<<<< HEAD
<<<<<<< HEAD
const CStoIMTypedict = @compat Dict("Gray" => "GrayscaleType", "GrayA" => "GrayscaleMatteType", "RGB" => "TrueColorType", "ARGB" => "TrueColorMatteType", "CMYK" => "ColorSeparationType")
=======
const CStoIMTypedict = @compat Dict("Gray" => "GrayscaleType", "GrayAlpha" => "GrayscaleMatteType", "RGB" => "TrueColorType", "ARGB" => "TrueColorMatteType", "CMYK" => "ColorSeparationType")
>>>>>>> adcbb46... started removing files that go into FileIO
=======
const CStoIMTypedict = @compat Dict("Gray" => "GrayscaleType", "GrayAlpha" => "GrayscaleMatteType", "RGB" => "TrueColorType", "ARGB" => "TrueColorMatteType", "CMYK" => "ColorSeparationType")
>>>>>>> 0611ac0... Revert "started rewriting SIF"

# Colorspace
const IMColorspace = ["RGB", "Gray", "Transparent", "OHTA", "Lab", "XYZ", "YCbCr", "YCC", "YIQ", "YPbPr", "YUV", "CMYK", "sRGB"]
const IMColordict = Dict([(IMColorspace[i], i) for i = 1:length(IMColorspace)])

<<<<<<< HEAD
<<<<<<< HEAD
function nchannels(imtype::AbstractString, cs::AbstractString, havealpha = false)
    n = 3
    if startswith(imtype, "Grayscale") || startswith(imtype, "Bilevel")
        n = 1
        cs = havealpha ? "GrayA" : "Gray"
=======
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
function nchannels(imtype::String, cs::String, havealpha = false)
    n = 3
    if startswith(imtype, "Grayscale") || startswith(imtype, "Bilevel")
        n = 1
        cs = havealpha ? "GrayAlpha" : "Gray"
<<<<<<< HEAD
>>>>>>> adcbb46... started removing files that go into FileIO
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    elseif cs == "CMYK"
        n = 4
    else
        cs = havealpha ? "ARGB" : "RGB" # only remaining variants supported by exportimagepixels
    end
    n + havealpha, cs
end

<<<<<<< HEAD
<<<<<<< HEAD
# channelorder = ["Gray" => "I", "GrayA" => "IA", "RGB" => "RGB", "ARGB" => "ARGB", "RGBA" => "RGBA", "CMYK" => "CMYK"]
=======
# channelorder = ["Gray" => "I", "GrayAlpha" => "IA", "RGB" => "RGB", "ARGB" => "ARGB", "RGBA" => "RGBA", "CMYK" => "CMYK"]
>>>>>>> adcbb46... started removing files that go into FileIO
=======
# channelorder = ["Gray" => "I", "GrayAlpha" => "IA", "RGB" => "RGB", "ARGB" => "ARGB", "RGBA" => "RGBA", "CMYK" => "CMYK"]
>>>>>>> 0611ac0... Revert "started rewriting SIF"

# Compression
const NoCompression = 1

type MagickWand
    ptr::Ptr{Void}
end

function MagickWand()
    wand = MagickWand(ccall((:NewMagickWand, libwand), Ptr{Void}, ()))
    finalizer(wand, destroymagickwand)
    wand
end

destroymagickwand(wand::MagickWand) = ccall((:DestroyMagickWand, libwand), Ptr{Void}, (Ptr{Void},), wand.ptr)

type PixelWand
    ptr::Ptr{Void}
end

function PixelWand()
    wand = PixelWand(ccall((:NewPixelWand, libwand), Ptr{Void}, ()))
    finalizer(wand, destroypixelwand)
    wand
end

destroypixelwand(wand::PixelWand) = ccall((:DestroyPixelWand, libwand), Ptr{Void}, (Ptr{Void},), wand.ptr)

const IMExceptionType = Array(Cint, 1)
function error(wand::MagickWand)
<<<<<<< HEAD
<<<<<<< HEAD
    pMsg = ccall((:MagickGetException, libwand), Ptr{UInt8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
=======
    pMsg = ccall((:MagickGetException, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
    pMsg = ccall((:MagickGetException, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    msg = bytestring(pMsg)
    relinquishmemory(pMsg)
    error(msg)
end
function error(wand::PixelWand)
<<<<<<< HEAD
<<<<<<< HEAD
    pMsg = ccall((:PixelGetException, libwand), Ptr{UInt8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
=======
    pMsg = ccall((:PixelGetException, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
    pMsg = ccall((:PixelGetException, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    msg = bytestring(pMsg)
    relinquishmemory(pMsg)
    error(msg)
end

function getsize(buffer, channelorder)
    if channelorder == "I"
        return size(buffer, 1), size(buffer, 2), size(buffer, 3)
    else
        return size(buffer, 2), size(buffer, 3), size(buffer, 4)
    end
end
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
getsize{C<:Colorant}(buffer::AbstractArray{C}, channelorder) = size(buffer, 1), size(buffer, 2), size(buffer, 3)

colorsize(buffer, channelorder) = channelorder == "I" ? 1 : size(buffer, 1)
colorsize{C<:Colorant}(buffer::AbstractArray{C}, channelorder) = 1
=======
=======
>>>>>>> adcbb46... started removing files that go into FileIO
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
getsize{C<:Union(Color,TransparentColor)}(buffer::AbstractArray{C}, channelorder) = size(buffer, 1), size(buffer, 2), size(buffer, 3)

colorsize(buffer, channelorder) = channelorder == "I" ? 1 : size(buffer, 1)
colorsize{C<:Union(Color,TransparentColor)}(buffer::AbstractArray{C}, channelorder) = 1
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 49bdda1... started removing files that go into FileIO
=======
>>>>>>> adcbb46... started removing files that go into FileIO
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"

bitdepth{C<:Colorant}(buffer::AbstractArray{C}) = 8*eltype(C)
bitdepth{T}(buffer::AbstractArray{T}) = 8*sizeof(T)

# colorspace is included for consistency with constituteimage, but it is not used
function exportimagepixels!{T}(buffer::AbstractArray{T}, wand::MagickWand,  colorspace::ASCIIString, channelorder::ASCIIString; x = 0, y = 0)
    cols, rows, nimages = getsize(buffer, channelorder)
    ncolors = colorsize(buffer, channelorder)
    p = pointer(buffer)
    for i = 1:nimages
        nextimage(wand)
<<<<<<< HEAD
<<<<<<< HEAD
        status = ccall((:MagickExportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{UInt8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder, storagetype(T), p)
=======
        status = ccall((:MagickExportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder, storagetype(T), p)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
        status = ccall((:MagickExportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder, storagetype(T), p)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
        status == 0 && error(wand)
        p += sizeof(T)*cols*rows*ncolors
    end
    buffer
end

# function importimagepixels{T}(buffer::AbstractArray{T}, wand::MagickWand, colorspace::ASCIIString; x = 0, y = 0)
#     cols, rows = getsize(buffer, colorspace)
<<<<<<< HEAD
<<<<<<< HEAD
#     status = ccall((:MagickImportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{UInt8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder[colorspace], storagetype(T), buffer)
=======
#     status = ccall((:MagickImportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder[colorspace], storagetype(T), buffer)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
#     status = ccall((:MagickImportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder[colorspace], storagetype(T), buffer)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
#     status == 0 && error(wand)
#     nothing
# end

function constituteimage{T<:Unsigned}(buffer::AbstractArray{T}, wand::MagickWand, colorspace::ASCIIString, channelorder::ASCIIString; x = 0, y = 0)
    cols, rows, nimages = getsize(buffer, channelorder)
    ncolors = colorsize(buffer, channelorder)
    p = pointer(buffer)
    depth = bitdepth(buffer)
    for i = 1:nimages
<<<<<<< HEAD
<<<<<<< HEAD
        status = ccall((:MagickConstituteImage, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Ptr{UInt8}, Cint, Ptr{Void}), wand.ptr, cols, rows, channelorder, storagetype(T), p)
=======
        status = ccall((:MagickConstituteImage, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, cols, rows, channelorder, storagetype(T), p)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
        status = ccall((:MagickConstituteImage, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, cols, rows, channelorder, storagetype(T), p)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
        status == 0 && error(wand)
        setimagecolorspace(wand, colorspace)
        status = ccall((:MagickSetImageDepth, libwand), Cint, (Ptr{Void}, Csize_t), wand.ptr, depth)
        status == 0 && error(wand)
        p += sizeof(T)*cols*rows*ncolors
    end
    nothing
end

<<<<<<< HEAD
<<<<<<< HEAD
function getblob(wand::MagickWand, format::AbstractString)
    setimageformat(wand, format)
    len = Array(Csize_t, 1)
    ptr = ccall((:MagickGetImagesBlob, libwand), Ptr{UInt8}, (Ptr{Void}, Ptr{Csize_t}), wand.ptr, len)
=======
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
function getblob(wand::MagickWand, format::String)
    setimageformat(wand, format)
    len = Array(Csize_t, 1)
    ptr = ccall((:MagickGetImagesBlob, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Csize_t}), wand.ptr, len)
<<<<<<< HEAD
>>>>>>> adcbb46... started removing files that go into FileIO
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    blob = pointer_to_array(ptr, convert(Int, len[1]))
    finalizer(blob, relinquishmemory)
    blob
end

<<<<<<< HEAD
<<<<<<< HEAD
function pingimage(wand::MagickWand, filename::AbstractString)
    status = ccall((:MagickPingImage, libwand), Cint, (Ptr{Void}, Ptr{UInt8}), wand.ptr, filename)
=======
function pingimage(wand::MagickWand, filename::String)
    status = ccall((:MagickPingImage, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, filename)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
function pingimage(wand::MagickWand, filename::String)
    status = ccall((:MagickPingImage, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, filename)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    status == 0 && error(wand)
    nothing
end

<<<<<<< HEAD
<<<<<<< HEAD
function readimage(wand::MagickWand, filename::AbstractString)
    status = ccall((:MagickReadImage, libwand), Cint, (Ptr{Void}, Ptr{UInt8}), wand.ptr, filename)
=======
function readimage(wand::MagickWand, filename::String)
    status = ccall((:MagickReadImage, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, filename)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
function readimage(wand::MagickWand, filename::String)
    status = ccall((:MagickReadImage, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, filename)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    status == 0 && error(wand)
    nothing
end

function readimage(wand::MagickWand, stream::IO)
    status = ccall((:MagickReadImageFile, libwand), Cint, (Ptr{Void}, Ptr{Void}), wand.ptr, Libc.FILE(stream).ptr)
    status == 0 && error(wand)
    nothing
end

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
function readimage(wand::MagickWand, stream::Vector{UInt8})
    status = ccall((:MagickReadImageBlob, libwand), Cint, (Ptr{Void}, Ptr{Void}, Cint), wand.ptr, stream, length(stream)*sizeof(eltype(stream)))
    status == 0 && error(wand)
    nothing
end

>>>>>>> origin/master
function writeimage(wand::MagickWand, filename::AbstractString)
    status = ccall((:MagickWriteImages, libwand), Cint, (Ptr{Void}, Ptr{UInt8}, Cint), wand.ptr, filename, true)
=======
function writeimage(wand::MagickWand, filename::String)
    status = ccall((:MagickWriteImages, libwand), Cint, (Ptr{Void}, Ptr{Uint8}, Cint), wand.ptr, filename, true)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
function writeimage(wand::MagickWand, filename::String)
    status = ccall((:MagickWriteImages, libwand), Cint, (Ptr{Void}, Ptr{Uint8}, Cint), wand.ptr, filename, true)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    status == 0 && error(wand)
    nothing
end

function size(wand::MagickWand)
    height = ccall((:MagickGetImageHeight, libwand), Csize_t, (Ptr{Void},), wand.ptr)
    width = ccall((:MagickGetImageWidth, libwand), Csize_t, (Ptr{Void},), wand.ptr)
    return convert(Int, width), convert(Int, height)
end

getnumberimages(wand::MagickWand) = convert(Int, ccall((:MagickGetNumberImages, libwand), Csize_t, (Ptr{Void},), wand.ptr))

nextimage(wand::MagickWand) = ccall((:MagickNextImage, libwand), Cint, (Ptr{Void},), wand.ptr) == 1

resetiterator(wand::MagickWand) = ccall((:MagickResetIterator, libwand), Void, (Ptr{Void},), wand.ptr)

newimage(wand::MagickWand, cols::Integer, rows::Integer, pw::PixelWand) = ccall((:MagickNewImage, libwand), Cint, (Ptr{Void}, Csize_t, Csize_t, Ptr{Void}), wand.ptr, cols, rows, pw.ptr) == 0 && error(wand)

# test whether image has an alpha channel
getimagealphachannel(wand::MagickWand) = ccall((:MagickGetImageAlphaChannel, libwand), Cint, (Ptr{Void},), wand.ptr) == 1


<<<<<<< HEAD
<<<<<<< HEAD
function getimageproperties(wand::MagickWand,patt::AbstractString)
    numbProp = Csize_t[0]
    p = ccall((:MagickGetImageProperties, libwand),Ptr{Ptr{UInt8}},(Ptr{Void},Ptr{UInt8},Ptr{Csize_t}),wand.ptr,patt,numbProp)
=======
function getimageproperties(wand::MagickWand,patt::String)
    numbProp = Csize_t[0]
    p = ccall((:MagickGetImageProperties, libwand),Ptr{Ptr{Uint8}},(Ptr{Void},Ptr{Uint8},Ptr{Csize_t}),wand.ptr,patt,numbProp)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
function getimageproperties(wand::MagickWand,patt::String)
    numbProp = Csize_t[0]
    p = ccall((:MagickGetImageProperties, libwand),Ptr{Ptr{Uint8}},(Ptr{Void},Ptr{Uint8},Ptr{Csize_t}),wand.ptr,patt,numbProp)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    if p == C_NULL
        error("Pattern not in property names")
    else
        nP = convert(Int, numbProp[1])
        ret = Array(ASCIIString, nP)
        for i = 1:nP
            ret[i] = bytestring(unsafe_load(p,i))
        end
        ret

    end
end

<<<<<<< HEAD
<<<<<<< HEAD
function getimageproperty(wand::MagickWand,prop::AbstractString)
    p = ccall((:MagickGetImageProperty, libwand),Ptr{UInt8},(Ptr{Void},Ptr{UInt8}),wand.ptr,prop)
    if p == convert(Ptr{UInt8}, C_NULL)
=======
function getimageproperty(wand::MagickWand,prop::String)
    p = ccall((:MagickGetImageProperty, libwand),Ptr{Uint8},(Ptr{Void},Ptr{Uint8}),wand.ptr,prop)
    if p == C_NULL
>>>>>>> adcbb46... started removing files that go into FileIO
=======
function getimageproperty(wand::MagickWand,prop::String)
    p = ccall((:MagickGetImageProperty, libwand),Ptr{Uint8},(Ptr{Void},Ptr{Uint8}),wand.ptr,prop)
    if p == C_NULL
>>>>>>> 0611ac0... Revert "started rewriting SIF"
        possib = getimageproperties(wand,"*")
        warn("Undefined property, possible names are \"$(join(possib,"\",\""))\"")
        nothing
    else
        bytestring(p)
    end
end

# # get number of colors in the image
# magickgetimagecolors(wand::MagickWand) = ccall((:MagickGetImageColors, libwand), Csize_t, (Ptr{Void},), wand.ptr)

# get the type
function getimagetype(wand::MagickWand)
    t = ccall((:MagickGetImageType, libwand), Cint, (Ptr{Void},), wand.ptr)
    # Apparently the following is necessary, because the type is "potential"
    ccall((:MagickSetImageType, libwand), Void, (Ptr{Void}, Cint), wand.ptr, t)
    1 <= t <= length(IMType) || error("Image type ", t, " not recognized")
    IMType[t]
end

# get the colorspace
function getimagecolorspace(wand::MagickWand)
    cs = ccall((:MagickGetImageColorspace, libwand), Cint, (Ptr{Void},), wand.ptr)
    1 <= cs <= length(IMColorspace) || error("Colorspace ", cs, " not recognized")
    IMColorspace[cs]
end

# set the colorspace
function setimagecolorspace(wand::MagickWand, cs::ASCIIString)
    status = ccall((:MagickSetImageColorspace, libwand), Cint, (Ptr{Void},Cint), wand.ptr, IMColordict[cs])
    status == 0 && error(wand)
    nothing
end

# set the compression
function setimagecompression(wand::MagickWand, compression::Integer)
    status = ccall((:MagickSetImageCompression, libwand), Cint, (Ptr{Void},Cint), wand.ptr, int32(compression))
    status == 0 && error(wand)
    nothing
end

function setimagecompressionquality(wand::MagickWand, quality::Integer)
    0 < quality <= 100 || error("quality setting must be in the (inclusive) range 1-100.\nSee http://www.imagemagick.org/script/command-line-options.php#quality for details")
    status = ccall((:MagickSetImageCompressionQuality, libwand), Cint, (Ptr{Void}, Cint), wand.ptr, quality)
    status == 0 && error(wand)
    nothing
end

# set the image format
function setimageformat(wand::MagickWand, format::ASCIIString)
<<<<<<< HEAD
<<<<<<< HEAD
    status = ccall((:MagickSetImageFormat, libwand), Cint, (Ptr{Void}, Ptr{UInt8}), wand.ptr, format)
=======
    status = ccall((:MagickSetImageFormat, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, format)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
    status = ccall((:MagickSetImageFormat, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, format)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    status == 0 && error(wand)
    nothing
end

# get the pixel depth
getimagedepth(wand::MagickWand) = convert(Int, ccall((:MagickGetImageDepth, libwand), Csize_t, (Ptr{Void},), wand.ptr))

# pixel depth for given channel type
<<<<<<< HEAD
<<<<<<< HEAD
getimagechanneldepth(wand::MagickWand, channelType::ChannelType) = convert(Int, ccall((:MagickGetImageChannelDepth, libwand), Csize_t, (Ptr{Void},UInt32), wand.ptr, channelType.value ))

pixelsetcolor(wand::PixelWand, colorstr::ByteString) = ccall((:PixelSetColor, libwand), Csize_t, (Ptr{Void},Ptr{UInt8}), wand.ptr, colorstr) == 0 && error(wand)

relinquishmemory(p) = ccall((:MagickRelinquishMemory, libwand), Ptr{UInt8}, (Ptr{UInt8},), p)

# get library information
# If you pass in "*", you get the full list of options
function queryoptions(pattern::AbstractString)
    nops = Cint[0]
    pops = ccall((:MagickQueryConfigureOptions, libwand), Ptr{Ptr{UInt8}}, (Ptr{UInt8}, Ptr{Cint}), pattern, nops)
=======
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
getimagechanneldepth(wand::MagickWand, channelType::ChannelType) = convert(Int, ccall((:MagickGetImageChannelDepth, libwand), Csize_t, (Ptr{Void},Uint32), wand.ptr, channelType.value ))

pixelsetcolor(wand::PixelWand, colorstr::ByteString) = ccall((:PixelSetColor, libwand), Csize_t, (Ptr{Void},Ptr{Uint8}), wand.ptr, colorstr) == 0 && error(wand)

relinquishmemory(p) = ccall((:MagickRelinquishMemory, libwand), Ptr{Uint8}, (Ptr{Uint8},), p)

# get library information
# If you pass in "*", you get the full list of options
function queryoptions(pattern::String)
    nops = Cint[0]
    pops = ccall((:MagickQueryConfigureOptions, libwand), Ptr{Ptr{Uint8}}, (Ptr{Uint8}, Ptr{Cint}), pattern, nops)
<<<<<<< HEAD
>>>>>>> adcbb46... started removing files that go into FileIO
=======
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    ret = Array(ASCIIString, nops[1])
    for i = 1:nops[1]
        ret[i] = bytestring(unsafe_load(pops, i))
    end
    ret
end

# queries the value of a particular option
<<<<<<< HEAD
<<<<<<< HEAD
function queryoption(option::AbstractString)
    p = ccall((:MagickQueryConfigureOption, libwand), Ptr{UInt8}, (Ptr{UInt8},), option)
=======
function queryoption(option::String)
    p = ccall((:MagickQueryConfigureOption, libwand), Ptr{Uint8}, (Ptr{Uint8},), option)
>>>>>>> adcbb46... started removing files that go into FileIO
=======
function queryoption(option::String)
    p = ccall((:MagickQueryConfigureOption, libwand), Ptr{Uint8}, (Ptr{Uint8},), option)
>>>>>>> 0611ac0... Revert "started rewriting SIF"
    bytestring(p)
end

end

LibMagick.have_imagemagick
