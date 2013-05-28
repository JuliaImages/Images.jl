##########   I/O   ###########

abstract ImageFileType

# Database of image file extension, magic bytes, and file type stubs
# Add new items to this database using add_image_file_format (below)
#
# Filename extensions do not uniquely specify the image type, so we
# have extensions pointing to a list of candidate formats. On reading,
# this list is really just a hint to improve efficiency: in the end,
# it's the set of magic bytes in the file that determine the file
# format.
fileext = Dict{ByteString, Vector{Int}}()
filemagic = Array(Vector{Uint8}, 0)
filetype = Array(Any, 0)
filesrcloaded = Array(Bool, 0)
filesrc = Array(String, 0)

function loadformat(index::Int)
    filename = joinpath("ioformats", filesrc[index])
    if !isfile(filename)
        filename = joinpath(Pkg.dir(), "Images", "src", "ioformats", filesrc[index])
    end
    include(filename)
    filesrcloaded[index] = true
end

function add_image_file_format{ImageType<:ImageFileType}(ext::ByteString, magic::Vector{Uint8}, ::Type{ImageType}, filecode::ASCIIString)
    # Check to see whether these magic bytes are already in the database
    for i in 1:length(filemagic)
        if magic == filemagic[i]
            fileext[ext] = [i]  # update/add extension lookup
            filetype[i] = ImageType
            return
        end
    end
    # Add a new entry on the list
    push!(filemagic, magic)
    len = length(filemagic)
    push!(filetype, ImageType)
    if haskey(fileext, ext)
        push!(fileext[ext], len)
    else
        fileext[ext] = [len]
    end
    push!(filesrcloaded, isempty(filecode))
    push!(filesrc, filecode)
end
add_image_file_format{ImageType<:ImageFileType}(ext::ByteString, magic::Vector{Uint8}, ::Type{ImageType}) = add_image_file_format(ext, magic, ImageType, "")

# Define our fallback file format now, because we need it in generic imread.
# This has no extension (and is not added to the database), because it is always used as a stream.
type ImageMagick <: ImageFileType end

function imread(filename::String)
    _, ext = splitext(filename)
    ext = lowercase(ext)
    stream = open(filename, "r")
    magicbuf = Array(Uint8, 0)
    # Use the extension as a hint to determine file type
    if haskey(fileext, ext)
        candidates = fileext[ext]
        index = image_decode_magic(stream, magicbuf, candidates)
        if index > 0
            # Position to end of this type's magic bytes
            seek(stream, length(filemagic[index]))
            if !filesrcloaded[index]
                loadformat(index)
            end
            return imread(stream, filetype[index])
        end
    end
    # Extension wasn't helpful, look at all known magic bytes
    index = image_decode_magic(stream, magicbuf, 1:length(filemagic))
    if index > 0
        seek(stream, length(filemagic[index]))
        if !filesrcloaded[index]
            loadformat(index)
        end
        return imread(stream, filetype[index])
    end
    # There are no registered readers for this type. Try using ImageMagick if available.
    if have_imagemagick
        # Fall back on ImageMagick's convert & identify
        return imread(filename, ImageMagick)
    else
        error("Do not know how to read file ", filename)
    end
end

# Identify via magic bytes
function image_decode_magic{S<:IO}(stream::S, magicbuf::Vector{Uint8}, candidates::AbstractVector{Int})
    maxlen = 0
    for i in candidates
        len = length(filemagic[i])
        maxlen = (len > maxlen) ? len : maxlen
    end
    # If there are no magic bytes, simply use the file extension
    if maxlen == 0 && length(candidates) == 1
        return candidates[1]
    end
    while length(magicbuf) < maxlen && !eof(stream)
        push!(magicbuf, read(stream, Uint8))
    end
    for i in candidates
        ret = ccall(:memcmp, Int32, (Ptr{Uint8}, Ptr{Uint8}, Int), magicbuf, filemagic[i], min(length(filemagic[i]), length(magicbuf)))
        if ret == 0
            return i
        end
    end
    return -1
end

function imwrite(img, filename::String)
    _, ext = splitext(filename)
    ext = lowercase(ext)
    if haskey(fileext, ext)
        # Write using specific format
        candidates = fileext[ext]
        index = candidates[1]  # TODO?: use options, don't default to first
        if !filesrcloaded[index]
            loadformat(index)
        end
        imwrite(img, filename, filetype[index])
    elseif have_imagemagick
        # Fall back on ImageMagick
        imwrite(img, filename, ImageMagick)
    else
        error("Do not know how to write file ", filename)
    end
end

#### Implementation of specific formats ####

## ImageMagick's convert ##
classdict = [
  "DirectClassGray" => (true, false, "Gray"),
  "DirectClassGrayMatte" => (true, true, "GrayAlpha"),
  "DirectClassCMYK" => (true, false, "CMYK"),
  "DirectClassRGB" => (true, false, "RGB"),
  "DirectClasssRGB" => (true, false, "RGB"),
  "DirectClassRGBMatte" => (true, true, "RGBA"),
  "DirectClasssRGBMatte" => (true, true, "RGBA"),
  "PseudoClassGray" => (false, false, "Gray"),
  "PseudoClassRGB" => (false, false, "RGB"),
  "PseudoClasssRGB" => (false, false, "RGB")
]

tfdict = ["True" => true, "False" => false]

typedict = [8 => Uint8, 16 => Uint16, 32 => Uint32]

function imread(filename::String, ::Type{ImageMagick})
    # Determine what we need to know about the image format
    cmd = `identify -format "%r\n%z\n%w %h %n\n" $filename`
    stream, _ = readsfrom(cmd)
    imclass = strip(readline(stream))
    imclass = replace(imclass, " ", "")  # on some platforms spaces are inserted
    isdirect, hasalpha, colorspace = classdict[imclass]
    bitdepth = parseint(strip(readline(stream)))
    szline = strip(readline(stream))
    w, h, n = parseints(szline, 3)
    local sz
    local spatialorder
    if n > 1
        sz = (w, h, n)
        spatialorder = ["x", "y", "z"]  # might be time, but user will have to say
    else
        sz = (w, h)
        spatialorder = ["x", "y"]
    end
    prop = ["colorspace" => colorspace, "spatialorder" => spatialorder]
    # Extract the data
    isdirect = true   # haven't figured out how to get indexed image data yet
    if isdirect
        local data
        if colorspace[1:min(length(colorspace),4)] == "Gray"
            # TODO: figure out how to directly extract GrayAlpha
            cmd = `convert $filename -depth $bitdepth gray:-`
            stream, _ = readsfrom(cmd)
            data = read(stream, typedict[bitdepth], sz...)
            if hasalpha
                cmd = `convert $filename -alpha extract -depth $bitdepth gray:-`
                stream, _ = readsfrom(cmd)
                alpha = read(stream, typedict[bitdepth], sz...)
                data = cat(ndims(data)+1, data, alpha)
                prop["colordim"] = ndims(data)
            end
        else
            cmd = `convert $filename -depth $bitdepth $colorspace:-`
            stream, _ = readsfrom(cmd)
            nchannels = length(colorspace)
            data = read(stream, typedict[bitdepth], nchannels, sz...)
            prop["colordim"] = 1
        end
        return Image(data, prop)
    else
        # Indexed image
        # Note: this doesn't work, IM doesn't really return the index
        println("Reading indexed image ", filename)
#         cmd = `convert $filename -channel Index -separate -depth $bitdepth gray:-`
        cmd = `convert $filename -channel Index -separate -`
        stream, _ = readsfrom(cmd)
        data = read(stream, typedict[bitdepth], sz...)
#         error("Haven't figured out yet how to get the colormap")
        return ImageCmap(data, [], prop)
    end
end

function imwrite(img, filename::String, ::Type{ImageMagick})
    cs = colorspace(img)
    bitdepth = 8*sizeof(eltype(img))
    csout = cs
    if cs == "RGB24"
        csout = "rgb"
        bitdepth = 8
    elseif cs == "ARGB32"
        csout = "rgba"
        bitdepth = 8
    elseif cs == "ARGB"
        csout = "rgba"
    end
    if eltype(img) <: FloatingPoint
        bitdepth = 8  # TODO?: allow this to be specified
    end
    T = typedict[bitdepth]
    scalei = scaleinfo(typedict[bitdepth], img)
    w, h = widthheight(img)
    local cmd
    # For now, convert to direct
    if isa(img, AbstractImageIndexed)
        img = convert(Image, img)
    end
    if true # isdirect(img)
        if cs[1:min(4,length(cs))] == "Gray"
            if cs == "GrayAlpha"
                error("Not yet implemented")
                # Need to figure out how to write separate channels to the file twice
            end
            cmd = `convert -size $(w)x$(h) -depth $bitdepth gray: $filename`
            stream, proc = writesto(cmd)
            spawn(cmd)
            writegray(stream, img, scalei)
            close(stream)
        else
            csparsed = lowercase(csout)
            if !(csparsed == "rgb" || csparsed == "rgba")
                error("Output format ", csparsed, " not recognized")
            end
            if csparsed == "rgba"
                error("ImageMagick doesn't do the right thing here, sorry")
            end
            cmd = `convert -size $(w)x$(h) -depth $bitdepth $csparsed:- $filename`
            stream, proc = writesto(cmd)
            spawn(cmd)
            writecolor(stream, img, scalei)
            close(stream)
        end
    end
end

# Write grayscale values in horizontal-major order
function writegray(stream, img, scalei::ScaleInfo)
    assert2d(img)
    xfirst = isxfirst(img)
    firstindex, spsz, spstride, csz, cstride = iterate_spatial(img)
    isz, jsz = spsz
    istride, jstride = spstride
    A = parent(img)
    if xfirst
        for j = 1:jsz
            k = firstindex + (j-1)*jstride
            for i = 0:istride:(isz-1)*istride
                write(stream, scale(scalei, A[k+i]))
            end
        end
    else
        for i = 1:isz
            for j = 1:jsz
                k = firstindex+(i-1)*istride+(j-1)*jstride
                write(stream, scale(scalei,A[k]))
            end
        end
    end
end

# Write RGB or RGBA values in horizontal-major order
# The A value is written if present, otherwise skipped
function writecolor(stream, img, scalei::ScaleInfo)
    assert2d(img)
    xfirst = isxfirst(img)
    firstindex, spsz, spstride, csz, cstride = iterate_spatial(img)
    isz, jsz = spsz
    istride, jstride = spstride
    A = parent(img)
    cs = colorspace(img)
    if cs == "Gray"
        if xfirst
            for j = 1:jsz
                k = firstindex + (j-1)*jstride
                for i = 0:istride:(isz-1)*istride
                    v = scale(scalei, A[k+i])
                    write(stream, v)
                    write(stream, v)
                    write(stream, v)
                end
            end
        else
            for i = 1:isz
                for j = 1:jsz
                    k = firstindex+(i-1)*istride+(j-1)*jstride
                    v = scale(scalei, A[k])
                    write(stream, v)
                    write(stream, v)
                    write(stream, v)
                end
            end
        end
    elseif cs == "RGB"
        if xfirst
            for j = 1:jsz
                k = firstindex + (j-1)*jstride
                for i = 0:istride:(isz-1)*istride
                    ki = k+i
                    write(stream, scale(scalei, A[ki]))
                    write(stream, scale(scalei, A[ki+cstride]))
                    write(stream, scale(scalei, A[ki+2cstride]))
                end
            end
        else
            for i = 1:isz
                for j = 1:jsz
                    k = firstindex+(i-1)*istride+(j-1)*jstride
                    write(stream, scale(scalei, A[k]))
                    write(stream, scale(scalei, A[k+cstride]))
                    write(stream, scale(scalei, A[k+2cstride]))
                end
            end
        end
    elseif cs == "ARGB"
        if xfirst
            for j = 1:jsz
                k = firstindex + (j-1)*jstride
                for i = 0:istride:(isz-1)*istride
                    ki = k+i
                    write(stream,scale(scalei, A[ki+cstride]))
                    write(stream,scale(scalei, A[ki+2cstride]))
                    write(stream,scale(scalei, A[ki+3cstride]))
                    write(stream,scale(scalei, A[ki]))
                end
            end
        else
            for i = 1:isz
                for j = 1:jsz
                    k = firstindex+(i-1)*istride+(j-1)*jstride
                    write(stream,scale(scalei, A[k+cstride]))
                    write(stream,scale(scalei, A[k+2cstride]))
                    write(stream,scale(scalei, A[k+3cstride]))
                    write(stream,scale(scalei, A[k]))
                end
            end
        end
    elseif cs == "RGBA"
        if xfirst
            for j = 1:jsz
                k = firstindex + (j-1)*jstride
                for i = 0:istride:(isz-1)*istride
                    ki = k+i
                    write(stream,scale(scalei, A[ki]))
                    write(stream,scale(scalei, A[ki+cstride]))
                    write(stream,scale(scalei, A[ki+2cstride]))
                    write(stream,scale(scalei, A[ki+3cstride]))
                end
            end
        else
            for i = 1:isz
                for j = 1:jsz
                    k = firstindex+(i-1)*istride+(j-1)*jstride
                    write(stream,scale(scalei, A[k]))
                    write(stream,scale(scalei, A[k+cstride]))
                    write(stream,scale(scalei, A[k+2cstride]))
                    write(stream,scale(scalei, A[k+3cstride]))
                end
            end
        end
    elseif cs == "RGB24"
        if xfirst
            for j = 1:jsz
                k = firstindex + (j-1)*jstride
                for i = 0:istride:(isz-1)*istride
                    v = A[k+i]
                    write(stream, redval(v))
                    write(stream, greenval(v))
                    write(stream, blueval(v))
                end
            end
        else
            for i = 1:isz
                for j = 1:jsz
                    k = firstindex+(i-1)*istride+(j-1)*jstride
                    v = A[k]
                    write(stream, redval(v))
                    write(stream, greenval(v))
                    write(stream, blueval(v))
                end
            end
        end
    elseif cs == "ARGB32"
        if xfirst
            for j = 1:jsz
                k = firstindex + (j-1)*jstride
                for i = 0:istride:(isz-1)*istride
                    v = A[k+i]
                    write(stream, redval(v))
                    write(stream, greenval(v))
                    write(stream, blueval(v))
                    write(stream, alphaval(v))
                end
            end
        else
            for i = 1:isz
                for j = 1:jsz
                    k = firstindex+(i-1)*istride+(j-1)*jstride
                    v = A[k]
                    write(stream, redval(v))
                    write(stream, greenval(v))
                    write(stream, blueval(v))
                    write(stream, alphaval(v))
                end
            end
        end
    else
        error("colorspace ", cs, " not yet supported")
    end
end

## PPM, PGM, and PBM ##
type PPMBinary <: ImageFileType end
type PGMBinary <: ImageFileType end
type PBMBinary <: ImageFileType end

add_image_file_format(".ppm", b"P6", PPMBinary)
#add_image_file_format(".ppm", b"P3", PPMASCII)
add_image_file_format(".pgm", b"P5", PGMBinary)
#add_image_file_format(".pgm", b"P2", PGMASCII)
add_image_file_format(".pbm", b"P4", PBMBinary)
#add_image_file_format(".pbm", b"P1", PBMASCII)

function parse_netpbm_size(stream::IO)
    szline = strip(readline(stream))
    while isempty(szline) || szline[1] == '#'
        szline = strip(readline(stream))
    end
    parseints(szline, 2)
end

function parse_netpbm_maxval(stream::IO)
    eatwspace_comment(stream, '#')
    maxvalline = strip(readline(stream))
    parseint(maxvalline)
end

function imread{S<:IO}(stream::S, ::Type{PPMBinary})
    w, h = parse_netpbm_size(stream)
    maxval = parse_netpbm_maxval(stream)
    local dat
    if maxval <= 255
        dat = read(stream, Uint8, 3, w, h)
    elseif maxval <= typemax(Uint16)
        dat = Array(Uint16, 3, w, h)
        # there is no endian standard, but netpbm is big-endian
        if ENDIAN_BOM == 0x01020304
            for indx = 1:3*w*h
                dat[indx] = read(stream, Uint16)
            end
        else
            for indx = 1:3*w*h
                dat[indx] = bswap(read(stream, Uint16))
            end
        end
    else
        error("Image file may be corrupt. Are there really more than 16 bits in this image?")
    end
    Image(dat, ["colormap" => "RGB", "colordim" => 1, "storageorder" => ["c", "x", "y"], "limits" => (0,maxval)])
end

function imread{S<:IO}(stream::S, ::Type{PGMBinary})
    w, h = parse_netpbm_size(stream)
    maxval = parse_netpbm_maxval(stream)
    local dat
    if maxval <= 255
        dat = read(stream, Uint8, w, h)
    elseif maxval <= typemax(Uint16)
        dat = Array(Uint16, w, h)
        if ENDIAN_BOM == 0x01020304
            for indx = 1:w*h
                dat[indx] = read(stream, Uint16)
            end
        else
            for indx = 1:w*h
                dat[indx] = bswap(read(stream, Uint16))
            end
        end
    else
        error("Image file may be corrupt. Are there really more than 16 bits in this image?")
    end
    Image(dat, ["colormap" => "Gray", "storageorder" => ["x", "y"], "limits" => (0,maxval)])
end

function imread{S<:IO}(stream::S, ::Type{PBMBinary})
    w, h = parse_netpbm_size(stream)
    dat = BitArray(w, h)
    nbytes_per_row = iceil(w/8)
    for irow = 1:h, j = 1:nbytes_per_row
        tmp = read(stream, Uint8)
        offset = (j-1)*8
        for k = 1:min(8, w-offset)
            dat[offset+k, irow] = (tmp>>>(8-k))&0x01
        end
    end
    Image(dat, ["storageorder" => ["x", "y"]])
end

function imwrite(img, filename::String, ::Type{PPMBinary})
    stream = open(filename, "w")
    imwrite(img, stream, PPMBinary)
end

function imwrite(img, s::IO, ::Type{PPMBinary})
    write(s, "P6\n")
    write(s, "# ppm file written by Julia\n")
    w, h = widthheight(img)
    bitdepth = 8*sizeof(eltype(img))
    if eltype(img) <: FloatingPoint
        bitdepth = 8
    end
    cs = colorspace(img)
    if cs == "RGB24"
        bitdepth = 8
    elseif cs == "ARGB32"
        bitdepth = 8
    end
    T = typedict[bitdepth]
    mx = int(typemax(T))
    scalei = scaleinfo(T, img)
    write(s, "$w $h\n$mx\n")
    writecolor(s, img, scalei)
    close(s)
end

# ## PNG ##
# type PNGFile <: ImageFileType end
# 
# add_image_file_format(".png", [0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A], PNGFile)
# 
# libpng = dlopen("libpng")
# libimage_jl = dlopen(strcat(JULIA_HOME, "/../../extras/libjlimage"))
# png_major   = ccall(dlsym(libimage_jl, :jl_png_libpng_ver_major), Int32, ())
# png_minor   = ccall(dlsym(libimage_jl, :jl_png_libpng_ver_minor), Int32, ())
# png_release = ccall(dlsym(libimage_jl, :jl_png_libpng_ver_release), Int32, ())
# jl_png_read_init           = dlsym(libimage_jl, :jl_png_read_init)
# jl_png_write_init          = dlsym(libimage_jl, :jl_png_write_init)
# png_set_sig_bytes          = dlsym(libpng, :png_set_sig_bytes)
# png_read_info              = dlsym(libpng, :png_read_info)
# png_write_info             = dlsym(libpng, :png_write_info)
# png_read_update_info       = dlsym(libpng, :png_read_update_info)
# png_get_image_width        = dlsym(libpng, :png_get_image_width)
# png_get_image_height       = dlsym(libpng, :png_get_image_height)
# png_get_bit_depth          = dlsym(libpng, :png_get_bit_depth)
# png_get_color_type         = dlsym(libpng, :png_get_color_type)
# #png_get_filter_type        = dlsym(libpng, :png_get_filter_type)
# #png_get_compression_type   = dlsym(libpng, :png_get_compression_type)
# #png_get_interlace_type     = dlsym(libpng, :png_get_interlace_type)
# #png_set_interlace_handling = dlsym(libpng, :png_set_interlace_handling)
# png_set_expand_gray_1_2_4_to_8 = convert(Ptr{Void}, 0)
# try
#     png_set_expand_gray_1_2_4_to_8 = dlsym(libpng, :png_set_expand_gray_1_2_4_to_8)
# catch
#     png_set_expand_gray_1_2_4_to_8 = dlsym(libpng, :png_set_gray_1_2_4_to_8)
# end
# png_set_swap               = dlsym(libpng, :png_set_swap)
# png_set_IHDR               = dlsym(libpng, :png_set_IHDR)
# png_get_valid              = dlsym(libpng, :png_get_valid)
# png_get_rowbytes           = dlsym(libpng, :png_get_rowbytes)
# jl_png_read_image          = dlsym(libimage_jl, :jl_png_read_image)
# jl_png_write_image          = dlsym(libimage_jl, :jl_png_write_image)
# jl_png_read_close          = dlsym(libimage_jl, :jl_png_read_close)
# jl_png_write_close          = dlsym(libimage_jl, :jl_png_write_close)
# const PNG_COLOR_MASK_PALETTE = 1
# const PNG_COLOR_MASK_COLOR   = 2
# const PNG_COLOR_MASK_ALPHA   = 4
# const PNG_COLOR_TYPE_GRAY = 0
# const PNG_COLOR_TYPE_PALETTE = PNG_COLOR_MASK_COLOR | PNG_COLOR_MASK_PALETTE
# const PNG_COLOR_TYPE_RGB = PNG_COLOR_MASK_COLOR
# const PNG_COLOR_TYPE_RGB_ALPHA = PNG_COLOR_MASK_COLOR | PNG_COLOR_MASK_ALPHA
# const PNG_COLOR_TYPE_GRAY_ALPHA = PNG_COLOR_MASK_ALPHA
# const PNG_INFO_tRNS = 0x0010
# const PNG_INTERLACE_NONE = 0
# const PNG_INTERLACE_ADAM7 = 1
# const PNG_COMPRESSION_TYPE_BASE = 0
# const PNG_FILTER_TYPE_BASE = 0
# 
# png_color_type(::Type{CSGray}) = PNG_COLOR_TYPE_GRAY
# png_color_type(::Type{RGB}) = PNG_COLOR_TYPE_RGB
# png_color_type(::Type{CSRGBA}) = PNG_COLOR_TYPE_RGB_ALPHA
# png_color_type(::Type{CSGrayAlpha}) = PNG_COLOR_TYPE_GRAPH_ALPHA
# 
# # Used by finalizer to close down resources
# png_read_cleanup(png_voidp) = ccall(jl_png_read_close, Void, (Ptr{Ptr{Void}},), png_voidp)
# png_write_cleanup(png_voidp) = ccall(jl_png_write_close, Void, (Ptr{Ptr{Void}},), png_voidp)
# 
# function imread{S<:IO}(stream::S, ::Type{PNGFile})
#     png_voidp = ccall(jl_png_read_init, Ptr{Ptr{Void}}, (Ptr{Void},), fd(stream))
#     if png_voidp == C_NULL
#         error("Error opening PNG file ", stream.name, " for reading")
#     end
#     finalizer(png_voidp, png_read_cleanup)  # gracefully handle errors
#     png_p = pointer_to_array(png_voidp, (3,))
#     # png_p[1] is the png_structp, png_p[2] is the png_infop, and
#     # png_p[3] is a FILE* created from stream
#     # Tell the library how many header bytes we've already read
#     ccall(png_set_sig_bytes, Void, (Ptr{Void}, Int32), png_p[1], position(stream))
#     # Read the rest of the header
#     ccall(png_read_info, Void, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#     # Determine image parameters
#     width = ccall(png_get_image_width, Uint32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#     height = ccall(png_get_image_height, Uint32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#     bit_depth = ccall(png_get_bit_depth, Int32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#     if bit_depth <= 8
#         T = Uint8
#     elseif bit_depth == 16
#         T = Uint16
#     else
#         error("Unknown pixel type")
#     end
#     color_type = ccall(png_get_color_type, Int32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#     if color_type == PNG_COLOR_TYPE_GRAY
#         n_color_channels = 1
#         cs = CSGray
#     elseif color_type == PNG_COLOR_TYPE_RGB
#         n_color_channels = 3
#         cs = RGB
#     elseif color_type == PNG_COLOR_TYPE_GRAY_ALPHA
#         n_color_channels = 2
#         cs = CSGrayAlpha
#     elseif color_type == PNG_COLOR_TYPE_RGB_ALPHA
#         n_color_channels = 4
#         cs = CSRGBA
#     else
#         error("Color type not recognized")
#     end
#     # There are certain data types we just don't want to handle in
#     # their raw format. Check for these, and if needed ask libpng to
#     # transform them.
#     if color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8
#         ccall(png_set_expand_gray_1_2_4_to_8, Void, (Ptr{Void},), png_p[1])
#         bit_depth = 8
#     end
#     if ccall(png_get_valid, Uint32, (Ptr{Void}, Ptr{Void}, Uint32), png_p[1], png_p[2], PNG_INFO_tRNS) > 0
#         call(png_set_tRNS_to_alpha, Void, (Ptr{Void},), png_p[1])
#     end
#     # TODO: paletted images
#     if bit_depth > 8 && ENDIAN_BOM == 0x04030201
#         call(png_set_swap, Void, (Ptr{Void},), png_p[1])
#     end
#     # TODO? set a background
#     ccall(png_read_update_info, Void, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#     # Allocate the buffer
#     pixbytes = sizeof(T) * n_color_channels
#     rowbytes = pixbytes*width
#     if rowbytes != ccall(png_get_rowbytes, Uint32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#         error("Number of bytes per row is wrong")
#     end
#     if n_color_channels == 1
#         data = Array(T, width, height)
#         order = "xy"
#     else
#         data = Array(T, n_color_channels, width, height)
#         order = "cxy"
#     end
#     # Read the image
#     status = ccall(jl_png_read_image, Int32, (Ptr{Void}, Ptr{Void}, Ptr{Uint8}), png_p[1], png_p[2], data)
#     if status < 0
#         error("PNG error: read failed")
#     end
#     # No need to clean up, the finalizer will do it
#     # Build the image
#     return ImageCS{T,2,cs}(data,order)
# end
# 
# function imwrite{S<:IO}(img, stream::S, ::Type{PNGFile}, opts::Options)
#     dat = uint(img)
#     cs = colorspace(img)
#     # Ensure PNG storage order
#     if cs != CSGray
#         dat = permute_c_first(dat, storageorder(img))
#     end
#     imwrite(dat, cs, stream, PNGFile, opts)
# end
# imwrite{S<:IO}(img, stream::S, ::Type{PNGFile}) = imwrite(img, stream, PNGFile, Options())
# 
# function imwrite(img, filename::ByteString, ::Type{PNGFile}, opts::Options)
#     s = open(filename, "w")
#     imwrite(img, s, PNGFile, opts)
#     gc()  # force the finalizer to run
# end
# imwrite(img, filename::ByteString, ::Type{PNGFile}) = imwrite(img, filename, PNGFile, Options())
# 
# function imwrite{T<:Unsigned, CS<:ColorSpace, S<:IO}(data::Array{T}, ::Type{CS}, stream::S, ::Type{PNGFile}, opts::Options)
#     @defaults opts interlace=PNG_INTERLACE_NONE compression=PNG_COMPRESSION_TYPE_BASE filter=PNG_FILTER_TYPE_BASE
#     png_voidp = ccall(jl_png_write_init, Ptr{Ptr{Void}}, (Ptr{Void},), fd(stream))
#     if png_voidp == C_NULL
#         error("Error opening PNG file ", stream.name, " for writing")
#     end
#     finalizer(png_voidp, png_write_cleanup)
#     png_p = pointer_to_array(png_voidp, (3,))
#     # Specify the parameters and write the header
#     if CS == CSGray
#         height = size(data, 2)
#         width = size(data, 1)
#     else
#         height = size(data, 3)
#         width = size(data, 2)
#     end
#     bit_depth = 8*sizeof(T)
#     ccall(png_set_IHDR, Void, (Ptr{Void}, Ptr{Void}, Uint32, Uint32, Int32, Int32, Int32, Int32, Int32), png_p[1], png_p[2], width, height, bit_depth, png_color_type(CS), interlace, compression, filter)
#     # TODO: support comments via a comments=Array(PNGComments, 0) option
#     ccall(png_write_info, Void, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
#     if bit_depth > 8 && ENDIAN_BOM == 0x04030201
#         call(png_set_swap, Void, (Ptr{Void},), png_p[1])
#     end
#     # Write the image
#     status = ccall(jl_png_write_image, Int32, (Ptr{Void}, Ptr{Void}, Ptr{Uint8}), png_p[1], png_p[2], data)
#     if status < 0
#         error("PNG error: read failed")
#     end
#     # Clean up now, to prevent problems that otherwise occur when we
#     # try to immediately use the file from an external program before
#     # png_voidp gets garbage-collected
# #    finalizer_del(png_voidp)  # delete the finalizer
# #    png_write_cleanup(png_voidp)
# end    
# imwrite(data, cs, stream, ::Type{PNGFile}) = imwrite(data, cs, stream, PNGFile, Options())

function parseints(line, n)
    ret = Array(Int, n)
    pos = 1
    for i = 1:n
        pos2 = search(line, ' ', pos)
        if pos2 == 0
            pos2 = length(line)+1
        end
        ret[i] = parseint(line[pos:pos2-1])
        pos = pos2+1
        if pos > length(line) && i < n
            error("Line terminated without finding all ", n, " integers")
        end
    end
    tuple(ret...)
    
end


### Register formats for later loading here
type Dummy <: ImageFileType; end
add_image_file_format(".dummy", b"Dummy Image", Dummy, "dummy.jl")

# Andor Technologies SIF file format  
type AndorSIF <: Images.ImageFileType end
add_image_file_format(".sif", b"Andor Technology Multi-Channel File", AndorSIF, "SIF.jl")

# Imagine file format (http://holylab.wustl.edu, "Software" tab)
type ImagineFile <: Images.ImageFileType end
add_image_file_format(".imagine", b"IMAGINE", ImagineFile, "Imagine.jl")
