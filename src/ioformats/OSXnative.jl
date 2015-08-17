module LibOSXNative

#import Base: error, size
using Images, Color, Images.ColorVectorSpace, FixedPointNumbers, Compat

export imread

imread(io::IOStream) = imread(readbytes(io))

function imread(b::Array{Uint8,1})
  data = CFDataCreate(b)
  imgsrc = CGImageSourceCreateWithData(data)
  CFRelease(data)
  read_and_release_imgsrc(imgsrc)
end

function imread(filename)
    myURL = CFURLCreateWithFileSystemPath(abspath(filename))
    imgsrc = CGImageSourceCreateWithURL(myURL)
    CFRelease(myURL)
    read_and_release_imgsrc(imgsrc)
end

## core, internal function
function read_and_release_imgsrc(imgsrc)
    imgsrc == C_NULL && return nothing

    # Get image information
    imframes = convert(Int,CGImageSourceGetCount(imgsrc))
    if imframes == 0
        # Bail out to ImageMagick
        warn("OSX reader found no frames")
        CFRelease(imgsrc)
        return nothing
    end
    dict = CGImageSourceCopyPropertiesAtIndex(imgsrc, 0)
    imheight = CFNumberGetValue(CFDictionaryGetValue(dict, "PixelHeight"), Int16)
    imwidth = CFNumberGetValue(CFDictionaryGetValue(dict, "PixelWidth"), Int16)
    isindexed = CFBooleanGetValue(CFDictionaryGetValue(dict, "IsIndexed"))
    if isindexed
        # Bail out to ImageMagick
        warn("OSX reader: indexed color images not implemented")
        CFRelease(imgsrc)
        return nothing
    end
    hasalpha = CFBooleanGetValue(CFDictionaryGetValue(dict, "HasAlpha"))

    pixeldepth = CFNumberGetValue(CFDictionaryGetValue(dict, "Depth"), Int16)
    # Colormodel is one of: "RGB", "Gray", "CMYK", "Lab"
    colormodel = CFStringGetCString(CFDictionaryGetValue(dict, "ColorModel"))
    if colormodel == ""
        # Bail out to ImageMagick
        warn("OSX reader found empty colormodel string")
        CFRelease(imgsrc)
        return nothing
    end
    imtype = CFStringGetCString(CGImageSourceGetType(imgsrc))
    alphacode, storagedepth = alpha_and_depth(imgsrc)

    # Get image description string
    imagedescription = ""
    if imtype == "public.tiff"
        tiffdict = CFDictionaryGetValue(dict, "{TIFF}")
        imagedescription = tiffdict != C_NULL ?
            CFStringGetCString(CFDictionaryGetValue(tiffdict, "ImageDescription")) : nothing
    end
    CFRelease(dict)

    # Allocate the buffer and get the pixel data
    sz = imframes > 1 ? (convert(Int,imwidth), convert(Int,imheight), convert(Int,imframes)) : (convert(Int,imwidth), convert(Int,imheight))
    T = pixeldepth <= 8 ? Ufixed8 : Images.ufixedtype[pixeldepth]
    if colormodel == "Gray" && alphacode == 0 && storagedepth == 1
        buf = Array(Gray{T}, sz)
        fillgray!(reinterpret(T, buf, tuple(sz...)), imgsrc)
    elseif colormodel == "Gray" && in(alphacode, [1, 3])
        buf = Array(GrayAlpha{T}, sz)
        fillgrayalpha!(reinterpret(T, buf, tuple(2, sz...)), imgsrc)
    elseif colormodel == "Gray" && in(alphacode, [2, 4])
        # Not sure these exist out in the wild
        error("Unexpected GrayAlpha image layout: alpha is before intensity")
    elseif colormodel == "RGB" && in(alphacode, [1, 3])
        buf = Array(RGBA{T}, sz)
        fillcolor!(reinterpret(T, buf, tuple(4, sz...)), imgsrc, storagedepth)
    elseif colormodel == "RGB" && in(alphacode, [2, 4])
        buf = Array(ARGB{T}, sz)
        fillcolor!(reinterpret(T, buf, tuple(4, sz...)), imgsrc, storagedepth)
    elseif colormodel == "RGB" && alphacode == 0
        buf = Array(RGB{T}, sz)
        fillcolor!(reinterpret(T, buf, tuple(3, sz...)), imgsrc, storagedepth)
    elseif colormodel == "RGB" && in(alphacode, [5, 6])
        buf = alphacode == 5 ? Array(RGB4{T}, sz) : Array(RGB1{T}, sz)
        fillcolor!(reinterpret(T, buf, tuple(4, sz...)), imgsrc, storagedepth)
    else
        warn("Unknown colormodel ($colormodel) and alphacode ($alphacode) found by OSX reader")
        CFRelease(imgsrc)
        return nothing
    end
    CFRelease(imgsrc)

    # Set the image properties
    prop = @compat Dict(
        "spatialorder" => ["x", "y"],
        "pixelspacing" => [1, 1],
        "imagedescription" => imagedescription,
        "suppress" => Set(Any["imagedescription"])
    )
    if imframes > 1
        prop["timedim"] = ndims(buf)
    end
    Image(buf, prop)
end

function alpha_and_depth(imgsrc)
    CGimg = CGImageSourceCreateImageAtIndex(imgsrc, 0)  # Check only first frame
    alphacode = CGImageGetAlphaInfo(CGimg)
    bitspercomponent = CGImageGetBitsPerComponent(CGimg)
    bitsperpixel = CGImageGetBitsPerPixel(CGimg)
    CGImageRelease(CGimg)
    # Alpha codes documented here:
    # https://developer.apple.com/library/mac/documentation/graphicsimaging/reference/CGImage/Reference/reference.html#//apple_ref/doc/uid/TP30000956-CH3g-459700
    # Dividing bits per pixel by bits per component tells us how many
    # color + alpha slices we have in the file.
    alphacode, convert(Int,div(bitsperpixel, bitspercomponent))
end

function fillgray!{T}(buffer::AbstractArray{T, 2}, imgsrc)
    imwidth, imheight = size(buffer, 1), size(buffer, 2)
    CGimg = CGImageSourceCreateImageAtIndex(imgsrc, 0)
    imagepixels = CopyImagePixels(CGimg)
    pixelptr = CFDataGetBytePtr(imagepixels, eltype(buffer))
    imbuffer = pointer_to_array(pixelptr, (imwidth, imheight), false)
    buffer[:, :] = imbuffer
    CFRelease(imagepixels)
    CGImageRelease(CGimg)
end

# Image stack
function fillgray!{T}(buffer::AbstractArray{T, 3}, imgsrc)
    imwidth, imheight, nimages = size(buffer, 1), size(buffer, 2), size(buffer, 3)
    for i in 1:nimages
        CGimg = CGImageSourceCreateImageAtIndex(imgsrc, i - 1)
        imagepixels = CopyImagePixels(CGimg)
        pixelptr = CFDataGetBytePtr(imagepixels, T)
        imbuffer = pointer_to_array(pixelptr, (imwidth, imheight), false)
        buffer[:, :, i] = imbuffer
        CFRelease(imagepixels)
        CGImageRelease(CGimg)
    end
end

function fillgrayalpha!(buffer::AbstractArray{Uint8, 3}, imgsrc)
    imwidth, imheight = size(buffer, 2), size(buffer, 3)
    CGimg = CGImageSourceCreateImageAtIndex(imgsrc, 0)
    imagepixels = CopyImagePixels(CGimg)
    pixelptr = CFDataGetBytePtr(imagepixels, Uint16)
    imbuffer = pointer_to_array(pixelptr, (imwidth, imheight), false)
    buffer[1, :, :] = imbuffer & 0xff
    buffer[2, :, :] = div(imbuffer & 0xff00, 256)
    CFRelease(imagepixels)
    CGImageRelease(CGimg)
end
fillgrayalpha!(buffer::AbstractArray{Ufixed8, 3}, imgsrc) = fillgrayalpha!(reinterpret(Uint8, buffer), imgsrc)

function fillcolor!{T}(buffer::AbstractArray{T, 3}, imgsrc, nc)
    imwidth, imheight = size(buffer, 2), size(buffer, 3)
    CGimg = CGImageSourceCreateImageAtIndex(imgsrc, 0)
    imagepixels = CopyImagePixels(CGimg)
    pixelptr = CFDataGetBytePtr(imagepixels, T)
    imbuffer = pointer_to_array(pixelptr, (nc, imwidth, imheight), false)
    buffer[:, :, :] = imbuffer
    CFRelease(imagepixels)
    CGImageRelease(CGimg)
end

function fillcolor!{T}(buffer::AbstractArray{T, 4}, imgsrc, nc)
    imwidth, imheight, nimages = size(buffer, 2), size(buffer, 3), size(buffer, 4)
    for i in 1:nimages
        CGimg = CGImageSourceCreateImageAtIndex(imgsrc, i - 1)
        imagepixels = CopyImagePixels(CGimg)
        pixelptr = CFDataGetBytePtr(imagepixels, T)
        imbuffer = pointer_to_array(pixelptr, (nc, imwidth, imheight), false)
        buffer[:, :, :, i] = imbuffer
        CFRelease(imagepixels)
        CGImageRelease(CGimg)
    end
end


## OSX Framework Wrappers ######################################################

const foundation = Libdl.find_library(["/System/Library/Frameworks/Foundation.framework/Resources/BridgeSupport/Foundation"])
const imageio = Libdl.find_library(["/System/Library/Frameworks/ImageIO.framework/ImageIO"])

const kCFNumberSInt8Type = 1
const kCFNumberSInt16Type = 2
const kCFNumberSInt32Type = 3
const kCFNumberSInt64Type = 4
const kCFNumberFloat32Type = 5
const kCFNumberFloat64Type = 6
const kCFNumberCharType = 7
const kCFNumberShortType = 8
const kCFNumberIntType = 9
const kCFNumberLongType = 10
const kCFNumberLongLongType = 11
const kCFNumberFloatType = 12
const kCFNumberDoubleType = 13
const kCFNumberCFIndexType = 14
const kCFNumberNSIntegerType = 15
const kCFNumberCGFloatType = 16
const kCFNumberMaxType = 16

# Objective-C and NS wrappers
oms{T}(id, uid, ::Type{T}=Ptr{Void}) =
    ccall(:objc_msgSend, T, (Ptr{Void}, Ptr{Void}), id, selector(uid))

ogc{T}(id, ::Type{T}=Ptr{Void}) =
    ccall((:objc_getClass, "Cocoa.framework/Cocoa"), Ptr{Void}, (Ptr{Uint8}, ), id)

selector(sel::String) = ccall(:sel_getUid, Ptr{Void}, (Ptr{Uint8}, ), sel)

NSString(init::String) = ccall(:objc_msgSend, Ptr{Void},
                               (Ptr{Void}, Ptr{Void}, Ptr{Uint8}, Uint64),
                               oms(ogc("NSString"), "alloc"),
                               selector("initWithCString:encoding:"), init, 1)

NSLog(str::String, obj) = ccall((:NSLog, foundation), Ptr{Void},
                                (Ptr{Void}, Ptr{Void}), NSString(str), obj)

NSLog(str::String) = ccall((:NSLog, foundation), Ptr{Void},
                           (Ptr{Void}, ), NSString(str))

NSLog(obj::Ptr) = ccall((:NSLog, foundation), Ptr{Void}, (Ptr{Void}, ), obj)

# Core Foundation
# General
CFRetain(CFTypeRef::Ptr{Void}) = CFTypeRef != C_NULL &&
    ccall(:CFRetain, Void, (Ptr{Void}, ), CFTypeRef)

CFRelease(CFTypeRef::Ptr{Void}) = CFTypeRef != C_NULL &&
    ccall(:CFRelease, Void, (Ptr{Void}, ), CFTypeRef)

function CFGetRetainCount(CFTypeRef::Ptr{Void})
    CFTypeRef == C_NULL && return 0
    ccall(:CFGetRetainCount, Clonglong, (Ptr{Void}, ), CFTypeRef)
end

CFShow(CFTypeRef::Ptr{Void}) = CFTypeRef != C_NULL &&
    ccall(:CFShow, Void, (Ptr{Void}, ), CFTypeRef)

function CFCopyDescription(CFTypeRef::Ptr{Void})
    CFTypeRef == C_NULL && return C_NULL
    ccall(:CFCopyDescription, Ptr{Void}, (Ptr{Void}, ), CFTypeRef)
end

#CFCopyTypeIDDescription(CFTypeID::Cint) = CFTypeRef != C_NULL &&
#    ccall(:CFCopyTypeIDDescription, Ptr{Void}, (Cint, ), CFTypeID)

function CFGetTypeID(CFTypeRef::Ptr{Void})
    CFTypeRef == C_NULL && return nothing
    ccall(:CFGetTypeID, Culonglong, (Ptr{Void}, ), CFTypeRef)
end

CFURLCreateWithString(filename) =
    ccall(:CFURLCreateWithString, Ptr{Void},
          (Ptr{Void}, Ptr{Void}, Ptr{Void}), C_NULL, NSString(filename), C_NULL)

CFURLCreateWithFileSystemPath(filename::String) =
    ccall(:CFURLCreateWithFileSystemPath, Ptr{Void},
          (Ptr{Void}, Ptr{Void}, Cint, Bool), C_NULL, NSString(filename), 0, false)

# CFDictionary
CFDictionaryGetKeysAndValues(CFDictionaryRef::Ptr{Void}, keys, values) =
    CFDictionaryRef != C_NULL &&
    ccall(:CFDictionaryGetKeysAndValues, Void,
          (Ptr{Void}, Ptr{Ptr{Void}}, Ptr{Ptr{Void}}), CFDictionaryRef, keys, values)

function CFDictionaryGetValue(CFDictionaryRef::Ptr{Void}, key)
    CFDictionaryRef == C_NULL && return C_NULL
    ccall(:CFDictionaryGetValue, Ptr{Void},
          (Ptr{Void}, Ptr{Void}), CFDictionaryRef, key)
end

CFDictionaryGetValue(CFDictionaryRef::Ptr{Void}, key::String) =
    CFDictionaryGetValue(CFDictionaryRef::Ptr{Void}, NSString(key))

# CFNumber
function CFNumberGetValue(CFNum::Ptr{Void}, numtype)
    CFNum == C_NULL && return nothing
    out = Cint[0]
    ccall(:CFNumberGetValue, Bool, (Ptr{Void}, Cint, Ptr{Cint}), CFNum, numtype, out)
    out[1]
end

CFNumberGetValue(CFNum::Ptr{Void}, ::Type{Int8}) =
    CFNumberGetValue(CFNum, kCFNumberSInt8Type)

CFNumberGetValue(CFNum::Ptr{Void}, ::Type{Int16}) =
    CFNumberGetValue(CFNum, kCFNumberSInt16Type)

CFNumberGetValue(CFNum::Ptr{Void}, ::Type{Int32}) =
    CFNumberGetValue(CFNum, kCFNumberSInt32Type)

CFNumberGetValue(CFNum::Ptr{Void}, ::Type{Int64}) =
    CFNumberGetValue(CFNum, kCFNumberSInt64Type)

CFNumberGetValue(CFNum::Ptr{Void}, ::Type{Float32}) =
    CFNumberGetValue(CFNum, kCFNumberFloat32Type)

CFNumberGetValue(CFNum::Ptr{Void}, ::Type{Float64}) =
    CFNumberGetValue(CFNum, kCFNumberFloat64Type)

CFNumberGetValue(CFNum::Ptr{Void}, ::Type{Uint8}) =
    CFNumberGetValue(CFNum, kCFNumberCharType)

#CFBoolean
CFBooleanGetValue(CFBoolean::Ptr{Void}) =
    CFBoolean != C_NULL &&
    ccall(:CFBooleanGetValue, Bool, (Ptr{Void}, ), CFBoolean)

# CFString
function CFStringGetCString(CFStringRef::Ptr{Void})
    CFStringRef == C_NULL && return ""
    buffer = Array(Uint8, 1024)  # does this need to be bigger for Open Microscopy TIFFs?
    res = ccall(:CFStringGetCString, Bool, (Ptr{Void}, Ptr{Uint8}, Uint, Uint16),
                CFStringRef, buffer, length(buffer), 0x0600)
    res == C_NULL && return ""
    return bytestring(pointer(buffer))
end

# These were unsafe, can return null pointers at random times.
# See Apple Developer Docs
#CFStringGetCStringPtr(CFStringRef::Ptr{Void}) =
#    ccall(:CFStringGetCStringPtr, Ptr{Uint8}, (Ptr{Void}, Uint16), CFStringRef, 0x0600)
#
#getCFString(CFStr::Ptr{Void}) = CFStringGetCStringPtr(CFStr) != C_NULL ?
#    bytestring(CFStringGetCStringPtr(CFStr)) : ""

# Core Graphics
# CGImageSource
CGImageSourceCreateWithURL(myURL::Ptr{Void}) =
    ccall((:CGImageSourceCreateWithURL, imageio), Ptr{Void}, (Ptr{Void}, Ptr{Void}), myURL, C_NULL)

CGImageSourceCreateWithData(data::Ptr{Void}) =
    ccall((:CGImageSourceCreateWithData, imageio), Ptr{Void}, (Ptr{Void}, Ptr{Void}), data, C_NULL)

CGImageSourceGetType(CGImageSourceRef::Ptr{Void}) =
    ccall(:CGImageSourceGetType, Ptr{Void}, (Ptr{Void}, ), CGImageSourceRef)

CGImageSourceGetStatus(CGImageSourceRef::Ptr{Void}) =
    ccall(:CGImageSourceGetStatus, Uint32, (Ptr{Void}, ), CGImageSourceRef)

CGImageSourceGetStatusAtIndex(CGImageSourceRef::Ptr{Void}, n) =
    ccall(:CGImageSourceGetStatusAtIndex, Int32,
          (Ptr{Void}, Csize_t), CGImageSourceRef, n) #Int32?

CGImageSourceCopyProperties(CGImageSourceRef::Ptr{Void}) =
    ccall(:CGImageSourceCopyProperties, Ptr{Void},
          (Ptr{Void}, Ptr{Void}), CGImageSourceRef, C_NULL)

CGImageSourceCopyPropertiesAtIndex(CGImageSourceRef::Ptr{Void}, n) =
    ccall(:CGImageSourceCopyPropertiesAtIndex, Ptr{Void},
          (Ptr{Void}, Csize_t, Ptr{Void}), CGImageSourceRef, n, C_NULL)

CGImageSourceGetCount(CGImageSourceRef::Ptr{Void}) =
    ccall(:CGImageSourceGetCount, Csize_t, (Ptr{Void}, ), CGImageSourceRef)

CGImageSourceCreateImageAtIndex(CGImageSourceRef::Ptr{Void}, i) =
    ccall(:CGImageSourceCreateImageAtIndex, Ptr{Void},
          (Ptr{Void}, Uint64, Ptr{Void}), CGImageSourceRef, i, C_NULL)


# CGImageGet
CGImageGetAlphaInfo(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetAlphaInfo, Uint32, (Ptr{Void}, ), CGImageRef)

CGImageGetBitmapInfo(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetBitmapInfo, Uint32, (Ptr{Void}, ), CGImageRef)

CGImageGetBitsPerComponent(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetBitsPerComponent, Csize_t, (Ptr{Void}, ), CGImageRef)

CGImageGetBitsPerPixel(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetBitsPerPixel, Csize_t, (Ptr{Void}, ), CGImageRef)

CGImageGetBytesPerRow(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetBytesPerRow, Csize_t, (Ptr{Void}, ), CGImageRef)

CGImageGetColorSpace(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetColorSpace, Uint32, (Ptr{Void}, ), CGImageRef)

CGImageGetDecode(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetDecode, Ptr{Float64}, (Ptr{Void}, ), CGImageRef)

CGImageGetHeight(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetHeight, Csize_t, (Ptr{Void}, ), CGImageRef)

CGImageGetRenderingIntent(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetRenderingIntent, Uint32, (Ptr{Void}, ), CGImageRef)

CGImageGetShouldInterpolate(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetShouldInterpolate, Bool, (Ptr{Void}, ), CGImageRef)

CGImageGetTypeID() =
    ccall(:CGImageGetTypeID, Culonglong, (),)

CGImageGetWidth(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetWidth, Csize_t, (Ptr{Void}, ), CGImageRef)

CGImageRelease(CGImageRef::Ptr{Void}) =
    ccall(:CGImageRelease, Void, (Ptr{Void}, ), CGImageRef)

# Get pixel data
# See: https://developer.apple.com/library/mac/qa/qa1509/_index.html
CGImageGetDataProvider(CGImageRef::Ptr{Void}) =
    ccall(:CGImageGetDataProvider, Ptr{Void}, (Ptr{Void}, ), CGImageRef)

CGDataProviderCopyData(CGDataProviderRef::Ptr{Void}) =
    ccall(:CGDataProviderCopyData, Ptr{Void}, (Ptr{Void}, ), CGDataProviderRef)

CopyImagePixels(inImage::Ptr{Void}) =
    CGDataProviderCopyData(CGImageGetDataProvider(inImage))

CFDataGetBytePtr{T}(CFDataRef::Ptr{Void}, ::Type{T}) =
    ccall(:CFDataGetBytePtr, Ptr{T}, (Ptr{Void}, ), CFDataRef)

CFDataGetLength(CFDataRef::Ptr{Void}) =
    ccall(:CFDataGetLength, Ptr{Int64}, (Ptr{Void}, ), CFDataRef)

CFDataCreate(bytes::Array{Uint8,1}) =
  ccall(:CFDataCreate,Ptr{Void},(Ptr{Void},Ptr{Uint8},Csize_t),C_NULL,bytes,length(bytes))

end # Module
