module JPEGs

import Images
import Images: JPEG, Image, imread, imwrite, dimindex, add_image_file_format

const turbojpeg = "libturbojpeg" # TODO... BinDeps?
const DEFAULT_SUBSAMPLING = "S420"

include("libturbojpeg.jl")

function imread{S<:IO}(stream::S, ::Type{JPEG};
                       colorspace = "RGB", jpeg_flags = 0)

    # Set some input parameters
    if (colorspace == "RGB")
        pixelFormat = TurboJPEG.PixelFormat.RGB
    elseif (colorspace == "Gray")
        pixelFormat = TurboJPEG.PixelFormat.Gray
    else
        error("Unsupported input pixel format: ", colorspace)
    end

    pixelSize = TurboJPEG.PixelSize[pixelFormat]
    # Other decompression parameters
    pitch = 0 # Use scaledWidth
 
    # seek to stream start, because the magic has already been skipped
    # TODO: mmap for large files?
    seek(stream,0)
    inbuf = readbytes(stream)

    # Allocate output variables
    width = Cint[0]; height = Cint[0]; subsampling = Cint[-1]
    
    # Get header information
    TJ = tjInitDecompress()
    err = tjDecompressHeader2(TJ, inbuf, sizeof(inbuf),
                              width, height, subsampling)
    
    if (err != 0)
        error("Error reading JPEG header: ", bytestring(tjGetErrorStr()))
    end
    
    width,height,subsampling = width[1], height[1], subsampling[1]
    bufsize = width*height*pixelSize
     
    # Allocate output buffer
    local outbuf
    try
        outbuf = zeros(Uint8, pixelSize, width, height)
    catch
        error("Unable to allocate ", bufsize, " bytes for JPEG output")
    end
    err = tjDecompress2(TJ, pointer(inbuf),
                        uint64(bufsize), pointer(outbuf),
                        width, int32(pitch), height, int32(pixelFormat),
                        int32(jpeg_flags))
    
    err != 0 && error("Error decompressing image: ", bytestring(tjGetErrorStr()))
    
    props = Dict{ASCIIString,Any}()
    props["colorspace"] = colorspace
    props["spatialorder"] = ["x", "y"]
    props["colordim"] = 1 
    return Image(outbuf, props)
end

function imwrite(img, sheader::IO, ::Type{JPEG};
                 quality = 100, subsampling = TurboJPEG.Sampling.S420,
                 jpeg_flags = 0)

    props = img.properties

    TJc = tjInitCompress()

    # Set some properties
    cs = get(props, "colorspace", nothing)
    if (cs == "RGB")
        pixelFormat = TurboJPEG.PixelFormat.RGB
        cs = "RGB"
    elseif (cs == "Gray")
        pixelFormat = TurboJPEG.PixelFormat.Gray
    else
        warn("Undefined \"colorspace\", using RGB")
        pixelFormat = "RGB"
    end

    # Get the dimension indices
    xind = dimindex(img, "x")
    yind = dimindex(img, "y")
    width,height = size(img)[xind],size(img)[yind]
    
    # tell TJ to calculate pitch
    pitch = 0

    # output size, will be set by compressor
    jpegbuf_size = Clong[0]

    # passing C_NULL tells TJ to allocate
    jpegbuf = convert(Ptr{Ptr{Cuchar}}, [C_NULL])
    srcbuf = convert(Ptr{Cuchar}, img.data)
    
    err = tjCompress2(TJc, srcbuf, width, pitch, height, pixelFormat,
                      jpegbuf, jpegbuf_size, subsampling, quality,
                      jpeg_flags)

    if (err != 0)
        error("Error compressing JPEG: ", bytestring(tjGetErrorStr()))
    end
    try
        write(sheader, unsafe_load(jpegbuf), jpegbuf_size[1]*WORD_SIZE)
    catch errmsg
        error("Unable to write JPEG image to output stream: ", errmsg)
    end
    tjFree(unsafe_load(jpegbuf))
end

end
