module LibOSXNative

#import Base: error, size
using Images

export imread

function imread(filename)
    myURL = CFURLCreateWithFileSystemPath(abspath(filename))
    imgsrc = CGImageSourceCreateWithURL(myURL)
    CFRelease(myURL)
    imgsrc == C_NULL && error("Could not find file at URL: $filename")

    # Get image information
    imtype = getCFString(CGImageSourceGetType(imgsrc))
    imframes = int(CGImageSourceGetCount(imgsrc))
    dict = CGImageSourceCopyPropertiesAtIndex(imgsrc, 0)
    tiffdict = CFDictionaryGetValue(dict, "{TIFF}")
    #CFShow(dict)
    #CFShow(tiffdict)
    imheight = CFNumberGetValue(CFDictionaryGetValue(dict, "PixelHeight"), Int16)
    imwidth = CFNumberGetValue(CFDictionaryGetValue(dict, "PixelWidth"), Int16)
    pixeldepth = CFNumberGetValue(CFDictionaryGetValue(dict, "Depth"), Int16)
    colormodel = getCFString(CFDictionaryGetValue(dict, "ColorModel"))
    imagedescription = tiffdict != C_NULL ?
        getCFString(CFDictionaryGetValue(tiffdict, "ImageDescription")) : nothing

#    i = 0
#    CGimg = CGImageSourceCreateImageAtIndex(imgsrc, i)
#    @show CGImageGetAlphaInfo(CGimg)
#    @show CGImageGetBitmapInfo(CGimg)
#    @show int(CGImageGetBitsPerComponent(CGimg))
#    @show int(CGImageGetBitsPerPixel(CGimg))
#    @show int(CGImageGetBytesPerRow(CGimg))
#    @show CGImageGetColorSpace(CGimg)
#    @show CGImageGetDecode(CGimg)
#    @show CGImageGetRenderingIntent(CGimg)
#    @show CGImageGetShouldInterpolate(CGimg)
#    @show CGImageGetTypeID()
#    @show CFGetTypeID(CGimg)
#    @show height = int(CGImageGetHeight(CGimg))
#    @show width = int(CGImageGetWidth(CGimg))
#    @show width*height
#    CGImageRelease(CGimg)

    myimg = Array(Uint16, imwidth, imheight, imframes)

    for i in 1:imframes
        CGimg = CGImageSourceCreateImageAtIndex(imgsrc, i - 1)
        imagepixels = CopyImagePixels(CGimg)
        pixelptr = CFDataGetBytePtr(imagepixels, Uint16)
        imbuffer = pointer_to_array(pixelptr, (int(imwidth), int(imheight)), false)
        myimg[:, :, i] = imbuffer
        CFRelease(imagepixels)
        CGImageRelease(CGimg)
    end
    CFRelease(imgsrc)

    prop = {"colorspace" => "Gray",
            "spatialorder" => ["x", "y"],
            "pixelspacing" => [1, 1]}
    if imframes > 1
        prop["timedim"] = 3
    end
    Image(myimg, prop)
end


## OSX Framework Wrappers ######################################################

const foundation = find_library(["/System/Library/Frameworks/Foundation.framework/Resources/BridgeSupport/Foundation"])
const imageio = find_library(["/System/Library/Frameworks/ImageIO.framework/ImageIO"])

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
    ccall((:objc_getClass, "Cocoa.framework/Cocoa"), Ptr{Void}, (Ptr{Uint8},), id)

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
    CFTypeRef != C_NULL ?
        ccall(:CFGetRetainCount, Clonglong, (Ptr{Void}, ), CFTypeRef) : 0
end

CFShow(CFTypeRef::Ptr{Void}) = CFTypeRef != C_NULL && 
    ccall(:CFShow, Void, (Ptr{Void}, ), CFTypeRef)

CFCopyDescription(CFTypeRef::Ptr{Void}) = CFTypeRef != C_NULL && 
    ccall(:CFCopyDescription, Ptr{Void}, (Ptr{Void}, ), CFTypeRef)

#CFCopyTypeIDDescription(CFTypeID::Cint) = CFTypeRef != C_NULL && 
#    ccall(:CFCopyTypeIDDescription, Ptr{Void}, (Cint, ), CFTypeID)

CFGetTypeID(CFTypeRef::Ptr{Void}) = CFTypeRef != C_NULL && 
    ccall(:CFGetTypeID, Culonglong, (Ptr{Void}, ), CFTypeRef)

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

CFDictionaryGetValue(CFDictionaryRef::Ptr{Void}, key) =
    CFDictionaryRef != C_NULL && 
    ccall(:CFDictionaryGetValue, Ptr{Void},
          (Ptr{Void}, Ptr{Void}), CFDictionaryRef, key)

CFDictionaryGetValue(CFDictionaryRef::Ptr{Void}, key::String) = 
    CFDictionaryGetValue(CFDictionaryRef::Ptr{Void}, NSString(key))

# CFNumber
function CFNumberGetValue(CFNum::Ptr{Void}, numtype)
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

# CFString
CFStringGetCStringPtr(CFStringRef::Ptr{Void}) = 
    ccall(:CFStringGetCStringPtr, Ptr{Uint8}, (Ptr{Void}, Uint16), CFStringRef, 0x0600)

getCFString(CFStr::Ptr{Void}) = bytestring(CFStringGetCStringPtr(CFStr))

# Core Graphics
# CGImageSource
CGImageSourceCreateWithURL(myURL::Ptr{Void}) = 
    ccall((:CGImageSourceCreateWithURL, imageio), Ptr{Void}, (Ptr{Void}, Ptr{Void}), myURL, C_NULL)

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

#CFDataGetBytePtr(CFDataRef::Ptr{Void}) =
#    ccall(:CFDataGetBytePtr, Ptr{Uint8}, (Ptr{Void}, ), CFDataRef)

CFDataGetBytePtr(CFDataRef::Ptr{Void}, T) =
    ccall(:CFDataGetBytePtr, Ptr{T}, (Ptr{Void}, ), CFDataRef)

CFDataGetLength(CFDataRef::Ptr{Void}) =
    ccall(:CFDataGetLength, Ptr{Int64}, (Ptr{Void}, ), CFDataRef)


end # Module
