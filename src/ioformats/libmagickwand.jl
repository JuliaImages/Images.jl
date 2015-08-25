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
const have_imagemagick = isdefined(:libwand)

# Initialize the library
function init()
    global libwand
    if have_imagemagick
        eval(:(ccall((:MagickWandGenesis, $libwand), Void, ())))
    else
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
IMStorageTypes = Union(Uint8, Uint16, Uint32, Float32, Float64)
storagetype(::Type{Uint8}) = CHARPIXEL
storagetype(::Type{Uint16}) = SHORTPIXEL
storagetype(::Type{Uint32}) = INTEGERPIXEL
storagetype(::Type{Float32}) = FLOATPIXEL
storagetype(::Type{Float64}) = DOUBLEPIXEL
storagetype{T<:Ufixed}(::Type{T}) = storagetype(FixedPointNumbers.rawtype(T))
storagetype{CV<:Colorant}(::Type{CV}) = storagetype(eltype(CV))

# Channel types
type ChannelType
    value::Uint32
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

const CStoIMTypedict = @compat Dict("Gray" => "GrayscaleType", "GrayAlpha" => "GrayscaleMatteType", "RGB" => "TrueColorType", "ARGB" => "TrueColorMatteType", "CMYK" => "ColorSeparationType")

# Colorspace
const IMColorspace = ["RGB", "Gray", "Transparent", "OHTA", "Lab", "XYZ", "YCbCr", "YCC", "YIQ", "YPbPr", "YUV", "CMYK", "sRGB"]
const IMColordict = Dict([(IMColorspace[i], i) for i = 1:length(IMColorspace)])

function nchannels(imtype::String, cs::String, havealpha = false)
    n = 3
    if startswith(imtype, "Grayscale") || startswith(imtype, "Bilevel")
        n = 1
        cs = havealpha ? "GrayAlpha" : "Gray"
    elseif cs == "CMYK"
        n = 4
    else
        cs = havealpha ? "ARGB" : "RGB" # only remaining variants supported by exportimagepixels
    end
    n + havealpha, cs
end

# channelorder = ["Gray" => "I", "GrayAlpha" => "IA", "RGB" => "RGB", "ARGB" => "ARGB", "RGBA" => "RGBA", "CMYK" => "CMYK"]

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
    pMsg = ccall((:MagickGetException, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
    msg = bytestring(pMsg)
    relinquishmemory(pMsg)
    error(msg)
end
function error(wand::PixelWand)
    pMsg = ccall((:PixelGetException, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
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
getsize{C<:Colorant}(buffer::AbstractArray{C}, channelorder) = size(buffer, 1), size(buffer, 2), size(buffer, 3)

colorsize(buffer, channelorder) = channelorder == "I" ? 1 : size(buffer, 1)
colorsize{C<:Colorant}(buffer::AbstractArray{C}, channelorder) = 1

bitdepth{C<:Colorant}(buffer::AbstractArray{C}) = 8*eltype(C)
bitdepth{T}(buffer::AbstractArray{T}) = 8*sizeof(T)

# colorspace is included for consistency with constituteimage, but it is not used
function exportimagepixels!{T}(buffer::AbstractArray{T}, wand::MagickWand,  colorspace::ASCIIString, channelorder::ASCIIString; x = 0, y = 0)
    cols, rows, nimages = getsize(buffer, channelorder)
    ncolors = colorsize(buffer, channelorder)
    p = pointer(buffer)
    for i = 1:nimages
        nextimage(wand)
        status = ccall((:MagickExportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder, storagetype(T), p)
        status == 0 && error(wand)
        p += sizeof(T)*cols*rows*ncolors
    end
    buffer
end

# function importimagepixels{T}(buffer::AbstractArray{T}, wand::MagickWand, colorspace::ASCIIString; x = 0, y = 0)
#     cols, rows = getsize(buffer, colorspace)
#     status = ccall((:MagickImportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, cols, rows, channelorder[colorspace], storagetype(T), buffer)
#     status == 0 && error(wand)
#     nothing
# end

function constituteimage{T<:Unsigned}(buffer::AbstractArray{T}, wand::MagickWand, colorspace::ASCIIString, channelorder::ASCIIString; x = 0, y = 0)
    cols, rows, nimages = getsize(buffer, channelorder)
    ncolors = colorsize(buffer, channelorder)
    p = pointer(buffer)
    depth = bitdepth(buffer)
    for i = 1:nimages
        status = ccall((:MagickConstituteImage, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, cols, rows, channelorder, storagetype(T), p)
        status == 0 && error(wand)
        setimagecolorspace(wand, colorspace)
        status = ccall((:MagickSetImageDepth, libwand), Cint, (Ptr{Void}, Csize_t), wand.ptr, depth)
        status == 0 && error(wand)
        p += sizeof(T)*cols*rows*ncolors
    end
    nothing
end

function getblob(wand::MagickWand, format::String)
    setimageformat(wand, format)
    len = Array(Csize_t, 1)
    ptr = ccall((:MagickGetImagesBlob, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Csize_t}), wand.ptr, len)
    blob = pointer_to_array(ptr, convert(Int, len[1]))
    finalizer(blob, relinquishmemory)
    blob
end

function pingimage(wand::MagickWand, filename::String)
    status = ccall((:MagickPingImage, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, filename)
    status == 0 && error(wand)
    nothing
end

function readimage(wand::MagickWand, filename::String)
    status = ccall((:MagickReadImage, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, filename)
    status == 0 && error(wand)
    nothing
end

function readimage(wand::MagickWand, stream::IO)
    status = ccall((:MagickReadImageFile, libwand), Cint, (Ptr{Void}, Ptr{Void}), wand.ptr, Libc.FILE(stream).ptr)
    status == 0 && error(wand)
    nothing
end

function writeimage(wand::MagickWand, filename::String)
    status = ccall((:MagickWriteImages, libwand), Cint, (Ptr{Void}, Ptr{Uint8}, Cint), wand.ptr, filename, true)
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


function getimageproperties(wand::MagickWand,patt::String)
    numbProp = Csize_t[0]
    p = ccall((:MagickGetImageProperties, libwand),Ptr{Ptr{Uint8}},(Ptr{Void},Ptr{Uint8},Ptr{Csize_t}),wand.ptr,patt,numbProp)
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

function getimageproperty(wand::MagickWand,prop::String)
    p = ccall((:MagickGetImageProperty, libwand),Ptr{Uint8},(Ptr{Void},Ptr{Uint8}),wand.ptr,prop)
    if p == C_NULL
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
    status = ccall((:MagickSetImageFormat, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, format)
    status == 0 && error(wand)
    nothing
end

# get the pixel depth
getimagedepth(wand::MagickWand) = convert(Int, ccall((:MagickGetImageDepth, libwand), Csize_t, (Ptr{Void},), wand.ptr))

# pixel depth for given channel type
getimagechanneldepth(wand::MagickWand, channelType::ChannelType) = convert(Int, ccall((:MagickGetImageChannelDepth, libwand), Csize_t, (Ptr{Void},Uint32), wand.ptr, channelType.value ))

pixelsetcolor(wand::PixelWand, colorstr::ByteString) = ccall((:PixelSetColor, libwand), Csize_t, (Ptr{Void},Ptr{Uint8}), wand.ptr, colorstr) == 0 && error(wand)

relinquishmemory(p) = ccall((:MagickRelinquishMemory, libwand), Ptr{Uint8}, (Ptr{Uint8},), p)

# get library information
# If you pass in "*", you get the full list of options
function queryoptions(pattern::String)
    nops = Cint[0]
    pops = ccall((:MagickQueryConfigureOptions, libwand), Ptr{Ptr{Uint8}}, (Ptr{Uint8}, Ptr{Cint}), pattern, nops)
    ret = Array(ASCIIString, nops[1])
    for i = 1:nops[1]
        ret[i] = bytestring(unsafe_load(pops, i))
    end
    ret
end

# queries the value of a particular option
function queryoption(option::String)
    p = ccall((:MagickQueryConfigureOption, libwand), Ptr{Uint8}, (Ptr{Uint8},), option)
    bytestring(p)
end

end

LibMagick.have_imagemagick
