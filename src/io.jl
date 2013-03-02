##########   I/O   ###########

abstract ImageFileType
type PPMBinary <: ImageFileType end  # the "fallback" type

# Database of image file extension, magic bytes, and file type stubs
# Add new items to this database using add_image_file_format (below)
#
# Filename extensions do not uniquely specify the image type, so we
# have extensions pointing to a list of candidate formats. On reading,
# this list is really just a hint to improve efficiency: in the end,
# it's the set of magic bytes in the file that determine the file
# format.
_image_file_ext_dict = Dict{ByteString, Vector{Int}}()
_image_file_magic_list = Array(Vector{Uint8}, 0)
_image_file_type_list = Array(CompositeKind, 0)

function add_image_file_format{ImageType<:ImageFileType}(ext::ByteString, magic::Vector{Uint8}, ::Type{ImageType})
    # Check to see whether these magic bytes are already in the database
    for i in 1:length(_image_file_magic_list)
        if magic == _image_file_magic_list[i]
            _image_file_ext_dict[ext] = i  # update/add extension lookup
            _image_file_type_list[i] = ImageType
            return
        end
    end
    # Add a new entry on the list
    push(_image_file_magic_list, magic)
    len = length(_image_file_magic_list)
    push(_image_file_type_list, ImageType)
    if has(_image_file_ext_dict, ext)
        push(_image_file_ext_dict[ext], len)
    else
        _image_file_ext_dict[ext] = [len]
    end
end

function imread(filename::String)
    pathname, basename, ext = fileparts(filename)
    ext = lowercase(ext)
    stream = open(filename, "r")
    magicbuf = Array(Uint8, 0)
    # Use the extension as a hint to determine file type
    if has(_image_file_ext_dict, ext)
        candidates = _image_file_ext_dict[ext]
#        println(candidates)
        index = image_decode_magic(stream, magicbuf, candidates)
        if index > 0
            # Position to end of this type's magic bytes
            seek(stream, length(_image_file_magic_list[index]))
            return imread(stream, _image_file_type_list[index])
        end
    end
    # Extension wasn't helpful, look at all known magic bytes
    index = image_decode_magic(stream, magicbuf, 1:length(_image_file_magic_list))
    if index > 0
        seek(stream, length(_image_file_magic_list[index]))
        return imread(stream, _image_file_type_list[index])
    end
    if have_imagemagick
        # Fall back on ImageMagick's convert & identify
        cmd = `convert -format "%w %h" -identify $filename rgb:-`
        stream_conv = fdio(read_from(cmd).fd, true)
        spawn(cmd)
        img = imread(stream_conv, PPMBinary)
    else
        error("Do not know how to read file ", filename)
    end
end

# Identify via magic bytes
function image_decode_magic{S<:IO}(stream::S, magicbuf::Vector{Uint8}, candidates::AbstractVector{Int})
    maxlen = 0
    for i in candidates
        len = length(_image_file_magic_list[i])
        maxlen = (len > maxlen) ? len : maxlen
    end
    while length(magicbuf) < maxlen && !eof(stream)
        push(magicbuf, read(stream, Uint8))
    end
    for i in candidates
        ret = ccall(:memcmp, Int32, (Ptr{Uint8}, Ptr{Uint8}, Int), magicbuf, _image_file_magic_list[i], min(length(_image_file_magic_list[i]), length(magicbuf)))
        if ret == 0
            return i
        end
    end
    return -1
end

function imwrite(img, filename::String, opts::Options)
    @defaults opts binary=true   # useful once implement PPMASCII, etc
    pathname, basename, ext = fileparts(filename)
    ext = lowercase(ext)
    stream = open(filename, "w")
    if has(_image_file_ext_dict, ext)
        candidates = _image_file_ext_dict[ext]
        index = candidates[1]  # TODO: use options, don't default to first
        imwrite(img, stream, _image_file_type_list[index], opts)
    else
        if colorspace(img) == CSGray
            imwrite(img, stream, PGMBinary)
        elseif colorspace(img) == CSsRGB
            imwrite(img, stream, PPMBinary)
        end
        error("Not implemented")
    end
end
imwrite(img, filename::String) = imwrite(img, filename, Options())

## PPM, PGM, and PBM ##

type PGMBinary <: ImageFileType end
type PBMBinary <: ImageFileType end

add_image_file_format(".ppm", b"P6", PPMBinary)
#add_image_file_format(".ppm", b"P3", PPMASCII)
add_image_file_format(".ppm", b"P5", PGMBinary)
#add_image_file_format(".ppm", b"P2", PGMASCII)
add_image_file_format(".ppm", b"P4", PBMBinary)
#add_image_file_format(".ppm", b"P1", PBMASCII)

function parse_netpbm_header{S<:IO}(stream::S, with_maxval::Bool)
    szline = strip(readline(stream))
    while isempty(szline) || szline[1] == "#"
        szline = strip(readline(stream))
    end
    spc = strchr(szline, ' ')
    w = parse_int(szline[1:spc-1])
    h = parse_int(szline[spc+1:end])
    if with_maxval
        maxvalline = strip(readline(stream))
        while isempty(maxvalline) || maxvalline[1] == "#"
            maxvalline = strip(readline(stream))
        end
        maxval = parse_int(maxvalline)
        return w, h, maxval
    else
        return w, h
    end
end

function imread{S<:IO}(stream::S, ::Type{PPMBinary})
    w, h, maxval = parse_netpbm_header(stream, true)
    if maxval <= 255
        dat = read(stream, Uint8, 3, w, h)
        return ImageCSMinMax{Uint8, 2, CSsRGB}(dat, "cxy", 0, maxval)
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
        return ImageCSMinMax{Uint16, 2, CSsRGB}(dat, "cxy", 0, maxval)
    else
        error("Image file may be corrupt. Are there really more than 16 bits in this image?")
    end
end

function imread{S<:IO}(stream::S, ::Type{PGMBinary})
    w, h, maxval = parse_netpbm_header(stream, true)
    if maxval <= 255
        dat = read(stream, Uint8, w, h)
        return ImageCSMinMax{Uint8, 2, CSgray}(dat, "xy", 0, maxval)
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
        return ImageCSMinMax{Uint16, 2, CSgray}(dat, "xy", 0, maxval)
    else
        error("Image file may be corrupt. Are there really more than 16 bits in this image?")
    end
end

function imread{S<:IO}(stream::S, ::Type{PBMBinary})
    w, h = parse_netpbm_header(stream, false)
    dat = Array(Uint8, w, h)  # TODO: use BitArray if the user has loaded it
    nbytes_per_row = iceil(w/8)
    for irow = 1:h, j = 1:nbytes_per_row
        tmp = read(stream, Uint8)
        offset = (j-1)*8
        for k = 1:min(8, w-offset)
            dat[offset+k, irow] = (tmp>>>(8-k))&0x01
        end
    end
    ImageCSMinMax{Uint8, 2, CSgray}(dat, "xy", 0, 1)
end

function imwrite(img, file::String, ::Type{PPMBinary})
    s = open(file, "w")
    write(s, "P6\n")
    write(s, "# ppm file written by julia\n")
    dat = pixeldata_cxy(img)
    if eltype(dat) <: Float
    m, n = size(dat)
    mx = int(clim_max(img))
    write(s, "$m $n\n$mx\n")
    write(s, dat)
    close(s)
end

## PNG ##
type PNGFile <: ImageFileType end

add_image_file_format(".png", [0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A], PNGFile)

libpng = dlopen("libpng")
libimage_jl = dlopen(strcat(JULIA_HOME, "/../../extras/libjlimage"))
png_major   = ccall(dlsym(libimage_jl, :jl_png_libpng_ver_major), Int32, ())
png_minor   = ccall(dlsym(libimage_jl, :jl_png_libpng_ver_minor), Int32, ())
png_release = ccall(dlsym(libimage_jl, :jl_png_libpng_ver_release), Int32, ())
jl_png_read_init           = dlsym(libimage_jl, :jl_png_read_init)
jl_png_write_init          = dlsym(libimage_jl, :jl_png_write_init)
png_set_sig_bytes          = dlsym(libpng, :png_set_sig_bytes)
png_read_info              = dlsym(libpng, :png_read_info)
png_write_info             = dlsym(libpng, :png_write_info)
png_read_update_info       = dlsym(libpng, :png_read_update_info)
png_get_image_width        = dlsym(libpng, :png_get_image_width)
png_get_image_height       = dlsym(libpng, :png_get_image_height)
png_get_bit_depth          = dlsym(libpng, :png_get_bit_depth)
png_get_color_type         = dlsym(libpng, :png_get_color_type)
#png_get_filter_type        = dlsym(libpng, :png_get_filter_type)
#png_get_compression_type   = dlsym(libpng, :png_get_compression_type)
#png_get_interlace_type     = dlsym(libpng, :png_get_interlace_type)
#png_set_interlace_handling = dlsym(libpng, :png_set_interlace_handling)
png_set_expand_gray_1_2_4_to_8 = convert(Ptr{Void}, 0)
try
    png_set_expand_gray_1_2_4_to_8 = dlsym(libpng, :png_set_expand_gray_1_2_4_to_8)
catch
    png_set_expand_gray_1_2_4_to_8 = dlsym(libpng, :png_set_gray_1_2_4_to_8)
end
png_set_swap               = dlsym(libpng, :png_set_swap)
png_set_IHDR               = dlsym(libpng, :png_set_IHDR)
png_get_valid              = dlsym(libpng, :png_get_valid)
png_get_rowbytes           = dlsym(libpng, :png_get_rowbytes)
jl_png_read_image          = dlsym(libimage_jl, :jl_png_read_image)
jl_png_write_image          = dlsym(libimage_jl, :jl_png_write_image)
jl_png_read_close          = dlsym(libimage_jl, :jl_png_read_close)
jl_png_write_close          = dlsym(libimage_jl, :jl_png_write_close)
const PNG_COLOR_MASK_PALETTE = 1
const PNG_COLOR_MASK_COLOR   = 2
const PNG_COLOR_MASK_ALPHA   = 4
const PNG_COLOR_TYPE_GRAY = 0
const PNG_COLOR_TYPE_PALETTE = PNG_COLOR_MASK_COLOR | PNG_COLOR_MASK_PALETTE
const PNG_COLOR_TYPE_RGB = PNG_COLOR_MASK_COLOR
const PNG_COLOR_TYPE_RGB_ALPHA = PNG_COLOR_MASK_COLOR | PNG_COLOR_MASK_ALPHA
const PNG_COLOR_TYPE_GRAY_ALPHA = PNG_COLOR_MASK_ALPHA
const PNG_INFO_tRNS = 0x0010
const PNG_INTERLACE_NONE = 0
const PNG_INTERLACE_ADAM7 = 1
const PNG_COMPRESSION_TYPE_BASE = 0
const PNG_FILTER_TYPE_BASE = 0

png_color_type(::Type{CSGray}) = PNG_COLOR_TYPE_GRAY
png_color_type(::Type{CSsRGB}) = PNG_COLOR_TYPE_RGB
png_color_type(::Type{CSRGBA}) = PNG_COLOR_TYPE_RGB_ALPHA
png_color_type(::Type{CSGrayAlpha}) = PNG_COLOR_TYPE_GRAPH_ALPHA

# Used by finalizer to close down resources
png_read_cleanup(png_voidp) = ccall(jl_png_read_close, Void, (Ptr{Ptr{Void}},), png_voidp)
png_write_cleanup(png_voidp) = ccall(jl_png_write_close, Void, (Ptr{Ptr{Void}},), png_voidp)

function imread{S<:IO}(stream::S, ::Type{PNGFile})
    png_voidp = ccall(jl_png_read_init, Ptr{Ptr{Void}}, (Ptr{Void},), fd(stream))
    if png_voidp == C_NULL
        error("Error opening PNG file ", stream.name, " for reading")
    end
    finalizer(png_voidp, png_read_cleanup)  # gracefully handle errors
    png_p = pointer_to_array(png_voidp, (3,))
    # png_p[1] is the png_structp, png_p[2] is the png_infop, and
    # png_p[3] is a FILE* created from stream
    # Tell the library how many header bytes we've already read
    ccall(png_set_sig_bytes, Void, (Ptr{Void}, Int32), png_p[1], position(stream))
    # Read the rest of the header
    ccall(png_read_info, Void, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
    # Determine image parameters
    width = ccall(png_get_image_width, Uint32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
    height = ccall(png_get_image_height, Uint32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
    bit_depth = ccall(png_get_bit_depth, Int32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
    if bit_depth <= 8
        T = Uint8
    elseif bit_depth == 16
        T = Uint16
    else
        error("Unknown pixel type")
    end
    color_type = ccall(png_get_color_type, Int32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
    if color_type == PNG_COLOR_TYPE_GRAY
        n_color_channels = 1
        cs = CSGray
    elseif color_type == PNG_COLOR_TYPE_RGB
        n_color_channels = 3
        cs = CSsRGB
    elseif color_type == PNG_COLOR_TYPE_GRAY_ALPHA
        n_color_channels = 2
        cs = CSGrayAlpha
    elseif color_type == PNG_COLOR_TYPE_RGB_ALPHA
        n_color_channels = 4
        cs = CSRGBA
    else
        error("Color type not recognized")
    end
    # There are certain data types we just don't want to handle in
    # their raw format. Check for these, and if needed ask libpng to
    # transform them.
    if color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8
        ccall(png_set_expand_gray_1_2_4_to_8, Void, (Ptr{Void},), png_p[1])
        bit_depth = 8
    end
    if ccall(png_get_valid, Uint32, (Ptr{Void}, Ptr{Void}, Uint32), png_p[1], png_p[2], PNG_INFO_tRNS) > 0
        call(png_set_tRNS_to_alpha, Void, (Ptr{Void},), png_p[1])
    end
    # TODO: paletted images
    if bit_depth > 8 && ENDIAN_BOM == 0x04030201
        call(png_set_swap, Void, (Ptr{Void},), png_p[1])
    end
    # TODO? set a background
    ccall(png_read_update_info, Void, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
    # Allocate the buffer
    pixbytes = sizeof(T) * n_color_channels
    rowbytes = pixbytes*width
    if rowbytes != ccall(png_get_rowbytes, Uint32, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
        error("Number of bytes per row is wrong")
    end
    if n_color_channels == 1
        data = Array(T, width, height)
        order = "xy"
    else
        data = Array(T, n_color_channels, width, height)
        order = "cxy"
    end
    # Read the image
    status = ccall(jl_png_read_image, Int32, (Ptr{Void}, Ptr{Void}, Ptr{Uint8}), png_p[1], png_p[2], data)
    if status < 0
        error("PNG error: read failed")
    end
    # No need to clean up, the finalizer will do it
    # Build the image
    return ImageCS{T,2,cs}(data,order)
end

function imwrite{S<:IO}(img, stream::S, ::Type{PNGFile}, opts::Options)
    dat = uint(img)
    cs = colorspace(img)
    # Ensure PNG storage order
    if cs != CSGray
        dat = permute_c_first(dat, storageorder(img))
    end
    imwrite(dat, cs, stream, PNGFile, opts)
end
imwrite{S<:IO}(img, stream::S, ::Type{PNGFile}) = imwrite(img, stream, PNGFile, Options())

function imwrite(img, filename::ByteString, ::Type{PNGFile}, opts::Options)
    s = open(filename, "w")
    imwrite(img, s, PNGFile, opts)
    gc()  # force the finalizer to run
end
imwrite(img, filename::ByteString, ::Type{PNGFile}) = imwrite(img, filename, PNGFile, Options())

function imwrite{T<:Unsigned, CS<:ColorSpace, S<:IO}(data::Array{T}, ::Type{CS}, stream::S, ::Type{PNGFile}, opts::Options)
    @defaults opts interlace=PNG_INTERLACE_NONE compression=PNG_COMPRESSION_TYPE_BASE filter=PNG_FILTER_TYPE_BASE
    png_voidp = ccall(jl_png_write_init, Ptr{Ptr{Void}}, (Ptr{Void},), fd(stream))
    if png_voidp == C_NULL
        error("Error opening PNG file ", stream.name, " for writing")
    end
    finalizer(png_voidp, png_write_cleanup)
    png_p = pointer_to_array(png_voidp, (3,))
    # Specify the parameters and write the header
    if CS == CSGray
        height = size(data, 2)
        width = size(data, 1)
    else
        height = size(data, 3)
        width = size(data, 2)
    end
    bit_depth = 8*sizeof(T)
    ccall(png_set_IHDR, Void, (Ptr{Void}, Ptr{Void}, Uint32, Uint32, Int32, Int32, Int32, Int32, Int32), png_p[1], png_p[2], width, height, bit_depth, png_color_type(CS), interlace, compression, filter)
    # TODO: support comments via a comments=Array(PNGComments, 0) option
    ccall(png_write_info, Void, (Ptr{Void}, Ptr{Void}), png_p[1], png_p[2])
    if bit_depth > 8 && ENDIAN_BOM == 0x04030201
        call(png_set_swap, Void, (Ptr{Void},), png_p[1])
    end
    # Write the image
    status = ccall(jl_png_write_image, Int32, (Ptr{Void}, Ptr{Void}, Ptr{Uint8}), png_p[1], png_p[2], data)
    if status < 0
        error("PNG error: read failed")
    end
    # Clean up now, to prevent problems that otherwise occur when we
    # try to immediately use the file from an external program before
    # png_voidp gets garbage-collected
#    finalizer_del(png_voidp)  # delete the finalizer
#    png_write_cleanup(png_voidp)
end    
imwrite(data, cs, stream, ::Type{PNGFile}) = imwrite(data, cs, stream, PNGFile, Options())
