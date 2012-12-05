require("color.jl")
require("options.jl")
import OptionsMod.*
require("setutils.jl")

# Plain arrays can be treated as images. Other types will have
# metadata associated, make yours a child of one of the following:
abstract ImageMeta                       # image with metadata
abstract ImageDirect <: ImageMeta        # each pixel has own value/color
abstract ImageIndexed <: ImageMeta       # indexed images (i.e., lookup table)

##### Treating plain arrays as images #####

# The following functions allow plain arrays to be treated as images
# for all image-processing functions:
pixelsdirect(img::Array) = img
storageorder(img::Matrix) = "yx"  # column-major storage order
function storageorder{T}(img::Array{T,3})
    if size(img, 3) == 3
        return "yxc"              # an rgb image
    else
        error("Cannot infer storage order of image::Array")
    end
end
# You could write the following generically in terms of the above, but
# for efficiency it's better to have a direct implementation:
colorspace(img::Matrix) = CSgray
function colorspace(img::Array)
    if size(img, 3) == 3
        return CSsRGB
    else
        error("Cannot infer colorspace of image::Array")
    end
end
function ncoords(img::Array)
    if size(img, 3) == 1 || size(img, 3) == 3
        return 2
    else
        error("Cannot infer # of coordinates of image::Array")
    end
end


# TODO: consider Array vs AbstractArray

# Image with a defined dimensionality and color space
type ImageCS{T,N,CS<:ColorSpace} <: ImageDirect
    data::Array{T}  # no N here, because a 2d image with color may be 3d array
    order::ASCIIString  # storage order, e.g., "yxc", c = color channel
end

# Image with color space and explicit pixel min/max values
type ImageCSMinMax{T,N,CS<:ColorSpace} <: ImageDirect
    data::Array{T}
    order::ASCIIString
    min::T
    max::T

    function ImageCSMinMax(data::Array{T}, order::ASCIIString, mn, mx)
        new(data, order, convert(T, mn), convert(T, mx))
    end
end

# Indexed image (colormap)
type ImageColormap{TInt,T,N,CS<:ColorSpace} <: ImageIndexed
    data::Array{TInt,N}
    order::ASCIIString
    cmap::Array{T}

    function ImageColormap(data::Array{TInt, N}, order::ASCIIString, cmap::Array{T})
        m = match(r"c", order)
        if m != nothing
            error("For an indexed image, there is no channel coordinate")
        end
        new(data, order, cmap)
    end
end

# Accessor functions provide _reasonable_ defaults for types that
# don't encode them in metadata.  Where possible, write algorithms
# using the accessor functions, not the fields.
# If there is no clean default, omit it. Instead, write specialized
# versions of the algorithms for that type.

# When using plain arrays, the default convention is the Matlab one:
# images are stored in "yx" order, or "yxc" for color (assumes
# sRGB). You can override this by redefining ncoords, storageorder,
# and colorspace for Array inputs.

pixeldata{IT<:ImageDirect}(img::IT) = img.data
# deliberately omit for ImageIndexed, because the proper way to
# address the data will depend on the algorithm

ncoords{T,N}(img::Union(ImageCS{T,N}, ImageCSMinMax{T,N}, ImageColormap{T,N})) = N

storageorder(img::Union(ImageCS, ImageCSMinMax, ImageColormap)) = img.order

colorspace{T, N, CS<:ColorSpace}(img::Union(ImageCS{T,N,CS}, ImageCSMinMax{T,N,CS}, ImageColormap{T,N,CS})) = CS

function size_spatial(img::Union(Array,ImageDirect))
    order = storageorder(img)
    data = pixeldata(img)
    m = match(r"c", order)
    if m == nothing
        return size(data)
    else
        return ntuple(ncoords(img), i->size(data, i+(i >= m.offset)))
    end
end
size_spatial(img::ImageColormap) = size(img.data)

# The number of color channels
function ncolors(img::Union(Array,ImageDirect))
    order = storageorder(img)
    data = pixeldata(img)
    m = match(r"c", order)
    if m == nothing
        return 1
    else
        return size(data, m.offset)
    end
end

clim_min{T<:Float}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = zero(T)
clim_max{T<:Float}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = one(T)
clim_min{T<:Integer}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = typemin(T)
clim_max{T<:Integer}(img::Union(Array{T}, ImageCS{T}, ImageColormap{T})) = typemax(T)
clim_min(img::ImageCSMinMax) = img.min
clim_max(img::ImageCSMinMax) = img.max

function chop(img::Union(Array, ImageDirect))
    data = pixeldata(img)
    mx = clim_max(img)
    data[data > mx] = mx
    mn = clim_min(img)
    data[data < mn] = mn
    return data
end

function uint8{T<:Float}(img::Union(Array{T}, ImageCS{T}, ImageCSMinMax{T}))
    data = chop(img)
    mx = clim_max(img)
    mn = clim_min(img)
    if mn != 0
        data = data-mn
    end
    udat = uint8(round((typemax(Uint8)/(mx-mn))*data))
end
function uint16{T<:Float}(img::Union(Array{T}, ImageCS{T}, ImageCSMinMax{T}))
    data = chop(img)
    mx = clim_max(img)
    mn = clim_min(img)
    if mn != 0
        dat = dat-mn
    end
    udat = uint16(round((typemax(Uint16)/(mx-mn))*data))
end
uint{T<:Float}(img::Union(Array{T}, ImageCS{T}, ImageCSMinMax{T})) = uint8(img)
uint{T<:Integer}(img::Union(Array{T}, ImageCS{T}, ImageCSMinMax{T})) = pixeldata(img)

# Put pixel data in target storage order
function pixeldata_permute(img::ImageDirect, targetorder::ASCIIString)
    p = permutation_to(storageorder(img), targetorder)
    data = pixeldata(img)
    return permute(data, p)
end

##########  Configuration   #########

have_imagemagick = false
if system("which convert > /dev/null") != 0
    println("Warning: ImageMagick utilities not found. Install for more file format support.")
else
    have_imagemagick = true
end
use_imshow_cmd = false
imshow_cmd = ""
use_gaston = false
# TODO: Windows
imshow_cmd_list = ["feh", "gwenview"]
for thiscmd in imshow_cmd_list
    if system("which $thiscmd > /dev/null") == 0
        use_imshow_cmd = true
        imshow_cmd = thiscmd
        break
    end
end
if !use_imshow_cmd
    try
        x = gnuplot_state
        use_gaston = true
    catch
        println("Warning: no image viewer found. You will not be able to see images.")
    end
end


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


#############################################################

function lut(pal::Vector, a)
    out = similar(a, eltype(pal))
    n = numel(pal)
    for i=1:numel(a)
        out[i] = pal[clamp(a[i], 1, n)]
    end
    out
end

function indexedcolor(data, pal)
    mn = min(data); mx = max(data)
    indexedcolor(data, pal, mx-mn, (mx+mn)/2)
end

function indexedcolor(data, pal, w, l)
    n = numel(pal)-1
    if n == 0
        return fill(pal[1], size(data))
    end
    w_min = l - w/2
    scale = w==0 ? 1 : w/n
    lut(pal, iround((data - w_min)./scale) + 1)
end

const palette_gray32 = uint32([4278190080, 4278716424, 4279242768, 4279769112, 4280295456, 4280887593, 4281413937, 4281940281, 4282466625, 4283058762, 4283585106, 4284111450, 4284637794, 4285164138, 4285756275, 4286282619, 4286808963, 4287335307, 4287927444, 4288453788, 4288980132, 4289506476, 4290032820, 4290624957, 4291151301, 4291677645, 4292203989, 4292796126, 4293322470, 4293848814, 4294375158, 4294967295])

const palette_fire = uint32([4284111450, 4284702808, 4285294423, 4285886038, 4286477397, 4287069012, 4287660627, 4288251985, 4288843600, 4289435215, 4290026574, 4290618189, 4291209804, 4291801162, 4292392777, 4292984392, 4293575751, 4294167366, 4294824517, 4294757442, 4294755904, 4294754365, 4294687291, 4294685752, 4294684214, 4294617139, 4294615601, 4294614062, 4294612524, 4294545449, 4294543911, 4294542372, 4294475298, 4294473759, 4294472221, 4294405146, 4294403608, 4294402069, 4294400531, 4294333456, 4294331918, 4294330379, 4294263305, 4294261766, 4294260228, 4294258946, 4293340673, 4292422401, 4291569665, 4290651649, 4289733377, 4288880641, 4287962369, 4287109889, 4286191617, 4285273344, 4284420864, 4283502592, 4282649856, 4281731584, 4280813568, 4279960832, 4279042560, 4278190080])

const palette_rainbow = uint32([4279125737, 4279064810, 4279004140, 4279009006, 4278948336, 4278953201, 4278892531, 4278897397, 4278836727, 4278841593, 4278644669, 4278513537, 4278382149, 4278251018, 4280086024, 4281921031, 4283756038, 4285591045, 4287491588, 4289326595, 4291161602, 4292996609, 4294897152, 4294759425, 4294687234, 4294615300, 4294543109, 4294471175, 4294398984, 4294327050, 4294254859, 4294182925])

# Permute an array to put the color channel first
function permute_c_first(dat::Array, storageorder::ASCIIString)
    m = match(r"c", storageorder)
    if m == nothing || m.offset == 1
        return dat
    end
    porder = [m.offset]
    for i = 1:ndims(dat)
        if i != m.offset
            push(porder, i)
        end
    end
    return permute(dat, porder)
end

function write_bitmap_data(s, img)
    n, m = size(img)
    if eltype(img) <: Integer
        if ndims(img) == 3 && size(img,3) == 3
            for i=1:n, j=1:m, k=1:3
                write(s, uint8(img[i,j,k]))
            end
        elseif ndims(img) == 2
            if is(eltype(img),Int32) || is(eltype(img),Uint32)
                for i=1:n, j=1:m
                    p = img[i,j]
                    write(s, uint8(redval(p)))
                    write(s, uint8(greenval(p)))
                    write(s, uint8(blueval(p)))
                end
            else
                for i=1:n, j=1:m, k=1:3
                    write(s, uint8(img[i,j]))
                end
            end
        else
            error("unsupported array dimensions")
        end
    elseif eltype(img) <: Float
        # prevent overflow
        a = copy(img)
        a[img .> 1] = 1
        a[img .< 0] = 0
        if ndims(a) == 3 && size(a,3) == 3
            for i=1:n, j=1:m, k=1:3
                write(s, uint8(255*a[i,j,k]))
            end
        elseif ndims(a) == 2
            for i=1:n, j=1:m, k=1:3
                write(s, uint8(255*a[i,j]))
            end
        else
            error("unsupported array dimensions")
        end
    else
        error("unsupported array type")
    end
end

# demo:
# m = [ mandel(complex(r,i)) for i=-1:.01:1, r=-2:.01:0.5 ];
# ppmwrite(indexedcolor(m, palette_fire), "mandel.ppm")

function imwrite(I, file::String)
    if length(file) > 3 && file[end-3:end]==".ppm"
        # fall back to built-in in case convert not available
        return imwrite(I, file, PPMBinary)
    end
    h, w = size(I)
    cmd = `convert -size $(w)x$(h) -depth 8 rgb: $file`
    stream = fdio(write_to(cmd).fd, true)
    spawn(cmd)
    write_bitmap_data(stream, I)
    close(stream)
    wait(cmd)
end

function imshow(img, range)
    if colorspace(img) == CSGray
        img = imadjustintensity(img, range)
    end
    if use_imshow_cmd
        tmp::String = "/tmp/tmp.png"
        imwrite(img, tmp)
        cmd = `$imshow_cmd $tmp`
        spawn(cmd)
    elseif use_gaston
        a = AxesConf()
        pdata, a.ylabel, a.xlabel = pixeldata_yxc_named(img)
        c = CurveConf()
        addcoords([],[],pdata,c)
        addconf(a)
        llplot()
    else
        error("Can't show image, no viewer is configured")
    end
end
imshow(img) = imshow(img, [])

function imadjustintensity{T}(img::Array{T,2}, range)
    if length(range) == 0
        range = [min(img) max(img)]
    elseif length(range) == 1
        error("incorrect range")
    end
    tmp = (img - range[1])/(range[2] - range[1])
    tmp[tmp .> 1] = 1
    tmp[tmp .< 0] = 0
    out = tmp
end

function rgb2gray{T}(img::Array{T,3})
    n, m = size(img)
    wr, wg, wb = 0.30, 0.59, 0.11
    out = Array(T, n, m)
    if ndims(img)==3 && size(img,3)==3
        for i=1:n, j=1:m
            out[i,j] = wr*img[i,j,1] + wg*img[i,j,2] + wb*img[i,j,3]
        end
    elseif is(eltype(img),Int32) || is(eltype(img),Uint32)
        for i=1:n, j=1:m
            p = img[i,j]
            out[i,j] = wr*redval(p) + wg*greenval(p) + wb*blueval(p)
        end
    else
        error("unsupported array type")
    end
    out
end

rgb2gray{T}(img::Array{T,2}) = img

function sobel()
    f = [1.0 2.0 1.0; 0.0 0.0 0.0; -1.0 -2.0 -1.0]
    return f, f'
end

function prewitt()
    f = [1.0 1.0 1.0; 0.0 0.0 0.0; -1.0 -1.0 -1.0]
    return f, f'
end

# average filter
function imaverage(filter_size)
    if length(filter_size) != 2
        error("wrong filter size")
    end
    m, n = filter_size[1], filter_size[2]
    if mod(m, 2) != 1 || mod(n, 2) != 1
        error("filter dimensions must be odd")
    end
    f = ones(Float64, m, n)/(m*n)
end

imaverage() = imaverage([3 3])

# laplacian filter kernel
function imlaplacian(diagonals::String)
    if diagonals == "diagonals"
        return [1.0 1.0 1.0; 1.0 -8.0 1.0; 1.0 1.0 1.0]
    elseif diagonals == "nodiagonals"
        return [0.0 1.0 0.0; 1.0 -4.0 1.0; 0.0 1.0 0.0]
    end
end

imlaplacian() = imlaplacian("nodiagonals")

# more general version
function imlaplacian(alpha::Number)
    lc = alpha/(1 + alpha)
    lb = (1 - alpha)/(1 + alpha)
    lm = -4/(1 + alpha)
    return [lc lb lc; lb lm lb; lc lb lc]
end

# 2D gaussian filter kernel
function gaussian2d(sigma::Number, filter_size)
    if length(filter_size) == 0
        # choose 'good' size 
        m = 4*ceil(sigma)+1
        n = m
    elseif length(filter_size) != 2
        error("wrong filter size")
    else
        m, n = filter_size[1], filter_size[2]
    end
    if mod(m, 2) != 1 || mod(n, 2) != 1
        error("filter dimensions must be odd")
    end
    g = [exp(-(X.^2+Y.^2)/(2*sigma.^2)) for X=-floor(m/2):floor(m/2), Y=-floor(n/2):floor(n/2)]
    return g/sum(g)
end

gaussian2d(sigma::Number) = gaussian2d(sigma, [])
gaussian2d() = gaussian2d(0.5, [])

# difference of gaussian
function imdog(sigma::Number)
    m = 4*ceil(sqrt(2)*sigma)+1
    return gaussian2d(sqrt(2)*sigma, [m m]) - gaussian2d(sigma, [m m])
end

imdog() = imdog(0.5)

# laplacian of gaussian
function imlog(sigma::Number)
    m = 4*ceil(sigma)+1
    return [((x^2+y^2-sigma^2)/sigma^4)*exp(-(x^2+y^2)/(2*sigma^2)) for x=-floor(m/2):floor(m/2), y=-floor(m/2):floor(m/2)]
end

imlog() = imlog(0.5)

# Sum of squared differences
function ssd{T}(A::Array{T}, B::Array{T})
    return sum((A-B).^2)
end

# normalized by Array size
ssdn{T}(A::Array{T}, B::Array{T}) = ssd(A, B)/numel(A)

# sum of absolute differences
function sad{T}(A::Array{T}, B::Array{T})
    return sum(abs(A-B))
end

# normalized by Array size
sadn{T}(A::Array{T}, B::Array{T}) = sad(A, B)/numel(A)

# normalized cross correlation
function ncc{T}(A::Array{T}, B::Array{T})
    Am = (A-mean(A))[:]
    Bm = (B-mean(B))[:]
    return dot(Am,Bm)/(norm(Am)*norm(Bm))
end

function imfilter{T}(img::Matrix{T}, filter::Matrix{T}, border::String, value)
    si, sf = size(img), size(filter)
    A = zeros(T, si[1]+sf[1]-1, si[2]+sf[2]-1)
    s1, s2 = int((sf[1]-1)/2), int((sf[2]-1)/2)
    # correlation instead of convolution
    filter = fliplr(fliplr(filter).')
    if border == "replicate"
        A[s1+1:end-s1, s2+1:end-s2] = img
        A[s1+1:end-s1, 1:s2] = repmat(img[:,1], 1, s2)
        A[s1+1:end-s1, end-s2+1:end] = repmat(img[:,end], 1, s2)
        A[1:s1, s2+1:end-s2] = repmat(img[1,:], s1, 1)
        A[end-s1+1:end, s2+1:end-s2] = repmat(img[end,:], s1, 1)
        A[1:s1, 1:s2] = fliplr(fliplr(img[1:s1, 1:s2])')
        A[end-s1+1:end, 1:s2] = img[end-s1+1:end, 1:s2]'
        A[1:s1, end-s2+1:end] = img[1:s1, end-s2+1:end]'
        A[end-s1+1:end, end-s2+1:end] = flipud(fliplr(img[end-s1+1:end, end-s2+1:end]))'
    elseif border == "circular"
        A[s1+1:end-s1, s2+1:end-s2] = img
        A[s1+1:end-s1, 1:s2] = img[:, end-s2:end]
        A[s1+1:end-s1, end-s2+1:end] = img[:, 1:s2]
        A[1:s1, s2+1:end-s2] = img[end-s1+1:end, :]
        A[end-s1+1:end, s2+1:end-s2] = img[1:s1, :]
        A[1:s1, 1:s2] = img[end-s1+1:end, end-s2+1:end]
        A[end-s1+1:end, 1:s2] = img[1:s1, end-s2+1:end]
        A[1:s1, end-s2+1:end] = img[end-s1+1:end, 1:s2]
        A[end-s1+1:end, end-s2+1:end] = img[1:s1, 1:s2]
    elseif border == "mirror"
        A[s1+1:end-s1, s2+1:end-s2] = img
        A[s1+1:end-s1, 1:s2] = fliplr(img[:, 1:s2])
        A[s1+1:end-s1, end-s2+1:end] = fliplr(img[:, end-s2:end])
        A[1:s1, s2+1:end-s2] = flipud(img[1:s1, :])
        A[end-s1+1:end, s2+1:end-s2] = flipud(img[end-s1+1:end, :])
        A[1:s1, 1:s2] = fliplr(fliplr(img[1:s1, 1:s2])')
        A[end-s1+1:end, 1:s2] = img[end-s1+1:end, 1:s2]'
        A[1:s1, end-s2+1:end] = img[1:s1, end-s2+1:end]'
        A[end-s1+1:end, end-s2+1:end] = flipud(fliplr(img[end-s1+1:end, end-s2+1:end]))'
    elseif border == "value"
        A += value
        A[s1+1:end-s1, s2+1:end-s2] = img
    else
        error("wrong border treatment")
    end
    # check if separable
    U, S, V = svd(filter)
    separable = true;
    for i = 2:length(S)
        # assumption that <10^-7 \approx 0
        separable = separable && (abs(S[i]) < 10^-7)
    end
    if separable
        # conv2 isn't suitable for this (kernel center should be the actual center of the kernel)
        #C = conv2(squeeze(U[:,1]*sqrt(S[1])), squeeze(V[1,:]*sqrt(S[1])), A)
        x = squeeze(U[:,1]*sqrt(S[1]))
        y = squeeze(V[1,:]*sqrt(S[1]))
        sa = size(A)
        m = length(y)+sa[1]
        n = length(x)+sa[2]
        B = zeros(T, m, n)
        B[int((length(x))/2)+1:sa[1]+int((length(x))/2),int((length(y))/2)+1:sa[2]+int((length(y))/2)] = A
        y = fft([zeros(T,int((m-length(y)-1)/2)); y; zeros(T,int((m-length(y)-1)/2))])
        x = fft([zeros(T,int((m-length(x)-1)/2)); x; zeros(T,int((n-length(x)-1)/2))])
        C = fftshift(ifft2(fft2(B) .* (y * x.')))
        if T <: Real
            C = real(C)
        end
    else
        #C = conv2(A, filter)
        sa, sb = size(A), size(filter)
        At = zeros(T, sa[1]+sb[1], sa[2]+sb[2])
        Bt = zeros(T, sa[1]+sb[1], sa[2]+sb[2])
        At[int(end/2-sa[1]/2)+1:int(end/2+sa[1]/2), int(end/2-sa[2]/2)+1:int(end/2+sa[2]/2)] = A
        Bt[int(end/2-sb[1]/2)+1:int(end/2+sb[1]/2), int(end/2-sb[2]/2)+1:int(end/2+sb[2]/2)] = filter
        C = fftshift(ifft2(fft2(At).*fft2(Bt)))
        if T <: Real
            C = real(C)
        end
    end
    sc = size(C)
    out = C[int(sc[1]/2-si[1]/2):int(sc[1]/2+si[1]/2)-1, int(sc[2]/2-si[2]/2):int(sc[2]/2+si[2]/2)-1]
end

# imfilter for multi channel images
function imfilter{T}(img::Array{T,3}, filter::Matrix{T}, border::String, value)
    x, y, c = size(img)
    out = zeros(T, x, y, c)
    for i = 1:c
        out[:,:,i] = imfilter(squeeze(img[:,:,i]), filter, border, value)
    end
    out
end

imfilter(img, filter) = imfilter(img, filter, "replicate", 0)
imfilter(img, filter, border) = imfilter(img, filter, border, 0)

function imlineardiffusion{T}(img::Array{T,2}, dt::Float, iterations::Integer)
    u = img
    f = imlaplacian()
    for i = dt:dt:dt*iterations
        u = u + dt*imfilter(u, f, "replicate")
    end
    u
end

function imthresh{T}(img::Array{T,2}, threshold::Float)
    if !(0.0 <= threshold <= 1.0)
        error("threshold must be between 0 and 1")
    end
    img_max, img_min = max(img), min(img)
    tmp = zeros(T, size(img))
    # matter of taste?
    #tmp[img >= threshold*(img_max-img_min)+img_min] = 1
    tmp[img >= threshold] = 1
    return tmp
end

function imgaussiannoise{T}(img::Array{T}, variance::Number, mean::Number)
    return img + sqrt(variance)*randn(size(img)) + mean
end

imgaussiannoise{T}(img::Array{T}, variance::Number) = imgaussiannoise(img, variance, 0)
imgaussiannoise{T}(img::Array{T}) = imgaussiannoise(img, 0.01, 0)

# 'illustrates' fourier transform
ftshow{T}(A::Array{T,2}) = imshow(log(1+abs(fftshift(A))),[])

function rgb2ntsc{T}(img::Array{T})
    trans = [0.299 0.587 0.114; 0.596 -0.274 -0.322; 0.211 -0.523 0.312]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * squeeze(img[i,j,:])
    end
    return out
end

function ntsc2rgb{T}(img::Array{T})
    trans = [1 0.956 0.621; 1 -0.272 -0.647; 1 -1.106 1.703]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * squeeze(img[i,j,:])
    end
    return out
end

function rgb2ycbcr{T}(img::Array{T})
    trans = [65.481 128.533 24.966; -37.797 -74.203 112; 112 -93.786 -18.214]
    offset = [16.0; 128.0; 128.0]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = offset + trans * squeeze(img[i,j,:])
    end
    return out
end

function ycbcr2rgb{T}(img::Array{T})
    trans = inv([65.481 128.533 24.966; -37.797 -74.203 112; 112 -93.786 -18.214])
    offset = [16.0; 128.0; 128.0]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * (squeeze(img[i,j,:]) - offset)
    end
    return out
end

function imcomplement{T}(img::Array{T})
    return 1 - img
end

function rgb2hsi{T}(img::Array{T})
    R = img[:,:,1]
    G = img[:,:,2]
    B = img[:,:,3]
    H = acos((1/2*(2*R - G - B)) ./ (((R - G).^2 + (R - B).*(G - B)).^(1/2)+eps(T))) 
    H[B .> G] = 2*pi - H[B .> G]
    H /= 2*pi
    rgb_sum = R + G + B
    rgb_sum[rgb_sum .== 0] = eps(T)
    S = 1 - 3./(rgb_sum).*min(R, G, B)
    H[S .== 0] = 0
    I = 1/3*(R + G + B)
    return cat(3, H, S, I)
end

function hsi2rgb{T}(img::Array{T})
    H = img[:,:,1]*(2pi)
    S = img[:,:,2]
    I = img[:,:,3]
    R = zeros(T, size(img,1), size(img,2))
    G = zeros(T, size(img,1), size(img,2))
    B = zeros(T, size(img,1), size(img,2))
    RG = 0 .<= H .< 2*pi/3
    GB = 2*pi/3 .<= H .< 4*pi/3
    BR = 4*pi/3 .<= H .< 2*pi
    # RG sector
    B[RG] = I[RG].*(1 - S[RG])
    R[RG] = I[RG].*(1 + (S[RG].*cos(H[RG]))./cos(pi/3 - H[RG]))
    G[RG] = 3*I[RG] - R[RG] - B[RG]
    # GB sector
    R[GB] = I[GB].*(1 - S[GB])
    G[GB] = I[GB].*(1 + (S[GB].*cos(H[GB] - pi/3))./cos(H[GB]))
    B[GB] = 3*I[GB] - R[GB] - G[GB]
    # BR sector
    G[BR] = I[BR].*(1 - S[BR])
    B[BR] = I[BR].*(1 + (S[BR].*cos(H[BR] - 2*pi/3))./cos(-pi/3 - H[BR]))
    R[BR] = 3*I[BR] - G[BR] - B[BR]
    return cat(3, R, G, B)
end

function imstretch{T}(img::Array{T,2}, m::Number, slope::Number)
    return 1./(1 + (m./(img + eps(T))).^slope)
end

function imedge{T}(img::Array{T}, method::String, border::String)
    # needs more methods
    if method == "sobel"
        s1, s2 = sobel()
        img1 = imfilter(img, s1, border)
        img2 = imfilter(img, s2, border)
        return img1, img2, sqrt(img1.^2 + img2.^2), atan2(img2, img1)
    elseif method == "prewitt"
        s1, s2 = prewitt()
        img1 = imfilter(img, s1, border)
        img2 = imfilter(img, s2, border)
        return img1, img2, sqrt(img1.^2 + img2.^2), atan2(img2, img1)
    end
end

imedge{T}(img::Array{T}, method::String) = imedge(img, method, "replicate")
imedge{T}(img::Array{T}) = imedge(img, "sobel", "replicate")

# forward and backward differences 
# can be very helpful for discretized continuous models 
forwarddiffy{T}(u::Array{T,2}) = [u[2:end,:]; u[end,:]] - u
forwarddiffx{T}(u::Array{T,2}) = [u[:,2:end] u[:,end]] - u
backdiffy{T}(u::Array{T,2}) = u - [u[1,:]; u[1:end-1,:]]
backdiffx{T}(u::Array{T,2}) = u - [u[:,1] u[:,1:end-1]]

function imROF{T}(img::Array{T,2}, lambda::Number, iterations::Integer)
    # Total Variation regularized image denoising using the primal dual algorithm
    # Also called Rudin Osher Fatemi (ROF) model
    # lambda: regularization parameter
    s1, s2 = size(img)
    p = zeros(T, s1, s2, 2)
    u = zeros(T, s1, s2)
    grad_u = zeros(T, s1, s2, 2)
    div_p = zeros(T, s1, s2)
    dt = lambda/4
    for i = 1:iterations
        div_p = backdiffx(squeeze(p[:,:,1])) + backdiffy(squeeze(p[:,:,2]))
        u = img + div_p/lambda
        grad_u = cat(3, forwarddiffx(u), forwarddiffy(u))
        grad_u_mag = sqrt(grad_u[:,:,1].^2 + grad_u[:,:,2].^2)
        tmp = 1 + grad_u_mag*dt
        p = (dt*grad_u + p)./cat(3, tmp, tmp)
    end
    return u
end

# ROF Model for color images
function imROF{T}(img::Array{T,3}, lambda::Number, iterations::Integer)
    out = zeros(T, size(img))
    for i = 1:size(img, 3)
        out[:,:,i] = imROF(squeeze(img[:,:,i]), lambda, iterations)
    end
    return out
end
