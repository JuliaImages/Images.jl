module LibMagick

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
    setimageformat,
    writeimage


# Find the library
mpath = get(ENV, "MAGICK_HOME", "") # If MAGICK_HOME is defined, add to library search path
mpaths = isempty(mpath) ? ASCIIString[] : [mpath, joinpath(mpath, "lib")]
libnames = ["libMagickWand", "CORE_RL_wand_"]
suffixes = ["", "-Q16", "-6.Q16", "-Q8"]
options = ["", "HDRI"]
const libwand = find_library(vec(libnames.*transpose(suffixes).*reshape(options,(1,1,length(options)))), mpaths)
have_imagemagick = !isempty(libwand)

# Initialize the library
function init()
    if have_imagemagick
        ccall((:MagickWandGenesis, libwand), Void, ())
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

# Image type
const IMType = ["BilevelType", "GrayscaleType", "GrayscaleMatteType", "PaletteType", "PaletteMatteType", "TrueColorType", "TrueColorMatteType", "ColorSeparationType", "ColorSeparationMatteType", "OptimizeType", "PaletteBilevelMatteType"]
const IMTypedict = Dict(IMType, 1:length(IMType))

const CStoIMTypedict = ["Gray" => "GrayscaleType", "GrayAlpha" => "GrayscaleMatteType", "RGB" => "TrueColorType", "ARGB" => "TrueColorMatteType", "CMYK" => "ColorSeparationType"]

# Colorspace
const IMColorspace = ["RGB", "Gray", "Transparent", "OHTA", "Lab", "XYZ", "YCbCr", "YCC", "YIQ", "YPbPr", "YUV", "CMYK", "sRGB"]
const IMColordict = Dict(IMColorspace, 1:length(IMColorspace))

function nchannels(imtype::String, cs::String, havealpha = false)
    n = 3
    if beginswith(imtype, "Grayscale") || beginswith(imtype, "Bilevel")
        n = 1
        cs = havealpha ? "GrayAlpha" : "Gray"
    elseif cs == "CMYK"
        n = 4
    else
        cs = havealpha ? "ARGB" : "RGB" # only remaining variants supported by exportimagepixels
    end
    n + havealpha, cs
end

channelorder = ["Gray" => "I", "GrayAlpha" => "IA", "RGB" => "RGB", "ARGB" => "ARGB", "CMYK" => "CMYK"]

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

const IMExceptionType = Array(Cint, 1)
function error(wand::MagickWand)
    pMsg = ccall((:MagickGetException, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Cint}), wand.ptr, IMExceptionType)
    msg = bytestring(pMsg)
    relinquishmemory(pMsg)
    error(msg)
end



function exportimagepixels!{T}(buffer::AbstractArray{T}, wand::MagickWand,  colorspace::ASCIIString; x = 0, y = 0)
    status = ccall((:MagickExportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, size(buffer, 1+(ndims(buffer)>2)), size(buffer, 2+(ndims(buffer)>2)), channelorder[colorspace], storagetype(T), buffer)
    status == 0 && error(wand)
    buffer
end

function importimagepixels{T}(buffer::AbstractArray{T}, wand::MagickWand, colorspace::ASCIIString; x = 0, y = 0)
    status = ccall((:MagickImportImagePixels, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Csize_t, Csize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, x, y, size(buffer, 1+(ndims(buffer)>2)), size(buffer, 2+(ndims(buffer)>2)), channelorder[colorspace], storagetype(T), buffer)
    status == 0 && error(wand)
    nothing
end

function constituteimage{T<:Unsigned}(buffer::AbstractArray{T}, wand::MagickWand, colorspace::ASCIIString; x = 0, y = 0)
    status = ccall((:MagickConstituteImage, libwand), Cint, (Ptr{Void}, Cssize_t, Cssize_t, Ptr{Uint8}, Cint, Ptr{Void}), wand.ptr, size(buffer, 1+(ndims(buffer)>2)), size(buffer, 2+(ndims(buffer)>2)), channelorder[colorspace], storagetype(T), buffer)
    status == 0 && error(wand)
    status = ccall((:MagickSetImageDepth, libwand), Cint, (Ptr{Void}, Csize_t), wand.ptr, 8*sizeof(T))
    status == 0 && error(wand)
    nothing
end

function getblob(wand::MagickWand, format::String)
    setimageformat(wand, format)
    len = Array(Csize_t, 1)
    ptr = ccall((:MagickGetImagesBlob, libwand), Ptr{Uint8}, (Ptr{Void}, Ptr{Csize_t}), wand.ptr, len)
    blob = pointer_to_array(ptr, int(len[1]))
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
    status = ccall((:MagickReadImageFile, libwand), Cint, (Ptr{Void}, Ptr{Void}), wand.ptr, fdopen(stream))
    status == 0 && error(wand)
    nothing
end

function writeimage(wand::MagickWand, filename::String)
    status = ccall((:MagickWriteImage, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, filename)
    status == 0 && error(wand)
    nothing
end

function size(wand::MagickWand)
    height = ccall((:MagickGetImageHeight, libwand), Csize_t, (Ptr{Void},), wand.ptr)
    width = ccall((:MagickGetImageWidth, libwand), Csize_t, (Ptr{Void},), wand.ptr)
    return int(width), int(height)
end

getnumberimages(wand::MagickWand) = int(ccall((:MagickGetNumberImages, libwand), Csize_t, (Ptr{Void},), wand.ptr))

resetiterator(wand::MagickWand) = ccall((:MagickResetIterator, libwand), Void, (Ptr{Void},), wand.ptr)

# test whether image has an alpha channel
getimagealphachannel(wand::MagickWand) = ccall((:MagickGetImageAlphaChannel, libwand), Cint, (Ptr{Void},), wand.ptr) == 1

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

# set the image format
function setimageformat(wand::MagickWand, format::ASCIIString)
    status = ccall((:MagickSetImageFormat, libwand), Cint, (Ptr{Void}, Ptr{Uint8}), wand.ptr, format)
    status == 0 && error(wand)
    nothing
end

# get the pixel depth
getimagedepth(wand::MagickWand) = int(ccall((:MagickGetImageDepth, libwand), Csize_t, (Ptr{Void},), wand.ptr))


relinquishmemory(p) = ccall((:MagickRelinquishMemory, libwand), Ptr{Uint8}, (Ptr{Uint8},), p)

# Implementation of fdopen
# Delete this once 0.3 or greater is common, and use the CFILE mechanism
function fdopen(s::IO)
    @unix_only FILEp = ccall(:fdopen, Ptr{Void}, (Cint, Ptr{Uint8}), convert(Cint, fd(s)), "r")
    @windows_only FILEp = ccall(:_fdopen, Ptr{Void}, (Cint, Ptr{Uint8}), convert(Cint, fd(s)), "r")
    systemerror("fdopen failed", FILEp == 0)
    status = ccall(:fseek, Cint, (Ptr{Void}, Clong, Cint), FILEp, convert(Clong, position(s)), int32(0))
    systemerror("fseek failed", status == 0)
    FILEp
end


end

LibMagick.have_imagemagick
