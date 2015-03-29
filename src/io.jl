##########   I/O   ###########

import .LibMagick
import .LibOSXNative

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

function _loadformat(index::Int)
    filename = joinpath("ioformats", filesrc[index])
    if !isfile(filename)
        filename = joinpath(dirname(@__FILE__), "ioformats", filesrc[index])
    end
    include(filename)
    filesrcloaded[index] = true
end
function loadformat{FileType<:ImageFileType}(::Type{FileType})
    indx = find(filetype .== FileType)
    if length(indx) == 1
        _loadformat(indx[1])
    elseif isempty(indx)
        error("File format ", FileType, " not found")
    else
        error("File format ", FileType, " is entered multiple times")
    end
    nothing
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
add_image_file_format{ImageType<:ImageFileType}(ext::ByteString, ::Type{ImageType}, filecode::ASCIIString) = add_image_file_format(ext, b"", ImageType, filecode)

# Define our fallback file formats now, because we need them in generic imread.
# This has no extension (and is not added to the database), because it is always used as a stream.
type ImageMagick <: ImageFileType end
type OSXNative <: ImageFileType end

function imread(filename::String;extraprop="",extrapropertynames=false)

    _, ext = splitext(filename)
    ext = lowercase(ext)

    img = open(filename, "r") do stream
        # Tries to read the image using a set of potential type candidates.
        # Returns the image if successful, `nothing` else.
        function tryread(candidates)
            if (index = image_decode_magic(stream, candidates)) > 0
                # Position to end of this type's magic bytes
                seek(stream, length(filemagic[index]))
                if !filesrcloaded[index]
                    _loadformat(index)
                end
                imread(stream, filetype[index])
            end
        end

        # Use the extension as a hint to determine file type
        if haskey(fileext, ext) && (img = tryread(fileext[ext])) != nothing
            return img
        end

        # Extension wasn't helpful, look at all known magic bytes
        if (img = tryread(1:length(filemagic))) != nothing
            return img
        end
    end

    if img != nothing
        return img
    end

    @osx_only begin
        img = imread(filename, OSXNative)
        if img != nothing
            return img
        end
    end

    # There are no registered readers for this type. Try using ImageMagick if available.
    if have_imagemagick
        return imread(filename, ImageMagick,extraprop=extraprop,extrapropertynames=extrapropertynames)
    else
        error("Do not know how to read file ", filename, ". Is ImageMagick installed properly? See README.")
    end
end

imread{C<:ColorValue}(filename::String, ::Type{C}) = imread(filename, ImageMagick, C)

# Identify via magic bytes
function image_decode_magic{S<:IO}(stream::S, candidates::AbstractVector{Int})
    maxlen = 0
    for i in candidates
        len = length(filemagic[i])
        maxlen = (len > maxlen) ? len : maxlen
    end
    # If there are no magic bytes, simply use the file extension
    if maxlen == 0 && length(candidates) == 1
        return candidates[1]
    end

    magicbuf = zeros(Uint8, maxlen)
    for i=1:maxlen
        if eof(stream) break end
        magicbuf[i] = read(stream, Uint8)
    end
    for i in candidates
        if length(filemagic[i]) == 0
            continue
        end
        ret = ccall(:memcmp, Int32, (Ptr{Uint8}, Ptr{Uint8}, Int), magicbuf, filemagic[i], min(length(filemagic[i]), length(magicbuf)))
        if ret == 0
            return i
        end
    end
    return -1
end

function imwrite(img, filename::String; kwargs...)
    _, ext = splitext(filename)
    ext = lowercase(ext)
    if haskey(fileext, ext)
        # Write using specific format
        candidates = fileext[ext]
        index = candidates[1]  # TODO?: use options, don't default to first
        if !filesrcloaded[index]
            _loadformat(index)
        end
        imwrite(img, filename, filetype[index]; kwargs...)
    elseif have_imagemagick
        # Fall back on ImageMagick
        imwrite(img, filename, ImageMagick; kwargs...)
    else
        error("Do not know how to write file ", filename)
    end
end

function imwrite{T<:ImageFileType}(img, filename::String, ::Type{T}; kwargs...)
    open(filename, "w") do s
        imwrite(img, s, T; kwargs...)
    end
end

# only mime writeable to PNG if 2D (used by IJulia for example)
mimewritable(::MIME"image/png", img::AbstractImage) = sdims(img) == 2 && timedim(img) == 0
# We have to disable Color's display via SVG, because both will get sent with unfortunate results.
# See IJulia issue #229
mimewritable{T<:ColorValue}(::MIME"image/svg+xml", ::AbstractMatrix{T}) = false

function writemime(stream::IO, ::MIME"image/png", img::AbstractImage; mapi=mapinfo_writemime(img), minpixels=10^4, maxpixels=10^6)
    assert2d(img)
    if isa(img, AbstractImageIndexed)
        # For now, convert to direct
        img = convert(Image, img)
    end
    A = data(img)
    nc = ncolorelem(img)
    npix = length(A)/nc
    while npix > maxpixels
        A = restrict(A, coords_spatial(img))
        npix = length(A)/nc
    end
    if npix < minpixels
        fac = ceil(Int, sqrt(minpixels/npix))
        r = ones(Int, ndims(img))
        r[coords_spatial(img)] = fac
        A = repeat(A, inner=r)
    end
    if eltype(A) != eltype(img)
        mapi = similar(mapi, eltype(mapi), eltype(A))
    end
    wand = image2wand(shareproperties(img, A), mapi, nothing)
    blob = LibMagick.getblob(wand, "png")
    write(stream, blob)
end

mapinfo_writemime{T}(img::AbstractImage{Gray{T}}) = mapinfo(Gray{Ufixed8},img)
mapinfo_writemime{C<:ColorValue}(img::AbstractImage{C}) = mapinfo(RGB{Ufixed8},img)
mapinfo_writemime{AC<:GrayAlpha}(img::AbstractImage{AC}) = mapinfo(GrayAlpha{Ufixed8},img)
mapinfo_writemime{AC<:AbstractAlphaColorValue}(img::AbstractImage{AC}) = mapinfo(RGBA{Ufixed8},img)
mapinfo_writemime(img::AbstractImage) = mapinfo(Ufixed8,img)


#### Implementation of specific formats ####

#### OSX Native readers from CoreGraphics

imread(filename::String, ::Type{OSXNative}) = LibOSXNative.imread(filename)

#### ImageMagick library

# fixed type for depths > 8
const ufixedtype = @compat Dict(10=>Ufixed10, 12=>Ufixed12, 14=>Ufixed14, 16=>Ufixed16)

function imread(filename::String, ::Type{ImageMagick};extraprop="",extrapropertynames=false)
    wand = LibMagick.MagickWand()
    LibMagick.readimage(wand, filename)
    LibMagick.resetiterator(wand)

    if extrapropertynames
        return(LibMagick.getimageproperties(wand,"*"))
    end
    
    imtype = LibMagick.getimagetype(wand)
    # Determine what we need to know about the image format
    sz = size(wand)
    n = LibMagick.getnumberimages(wand)
    if n > 1
        sz = tuple(sz..., n)
    end
    havealpha = LibMagick.getimagealphachannel(wand)
    prop = @compat Dict("spatialorder" => ["x", "y"], "pixelspacing" => [1,1])
    cs = LibMagick.getimagecolorspace(wand)
    if imtype == "GrayscaleType" || imtype == "GrayscaleMatteType"
        cs = "Gray"
    end
    prop["IMcs"] = cs

    if extraprop != ""
        for extra in [extraprop;]
            prop[extra] = LibMagick.getimageproperty(wand,extra)
        end
    end
        
    depth = LibMagick.getimagechanneldepth(wand, LibMagick.DefaultChannels)
    if depth <= 8
        T = Ufixed8     # always use 8-bit for 8-bit and less
    else
        T = ufixedtype[2*((depth+1)>>1)]  # always use an even # of bits (see issue 242#issuecomment-68845157)
    end

    channelorder = cs
    if havealpha
        if channelorder == "sRGB" || channelorder == "RGB"
            if is_little_endian
                T, channelorder = BGRA{T}, "BGRA"
            else
                T, channelorder = ARGB{T}, "ARGB"
            end
        elseif channelorder == "Gray"
            T, channelorder = GrayAlpha{T}, "IA"
        else
            error("Cannot parse colorspace $channelorder")
        end
    else
        if channelorder == "sRGB" || channelorder == "RGB"
            T, channelorder = RGB{T}, "RGB"
        elseif channelorder == "Gray"
            T, channelorder = Gray{T}, "I"
        else
            error("Cannot parse colorspace $channelorder")
        end
    end
    # Allocate the buffer and get the pixel data
    buf = Array(T, sz...)
    LibMagick.exportimagepixels!(buf, wand, cs, channelorder)
    if n > 1
        prop["timedim"] = ndims(buf)
    end
    Image(buf, prop)
end

imread{C<:ColorType}(filename::String, ::Type{ImageMagick}, ::Type{C}) = convert(Image{C}, imread(filename, ImageMagick))

function imwrite(img, filename::String, ::Type{ImageMagick}; mapi = mapinfo(ImageMagick, img), quality = nothing)
    wand = image2wand(img, mapi, quality)
    LibMagick.writeimage(wand, filename)
end

function image2wand(img, mapi, quality)
    if isa(img, AbstractImageIndexed)
        # For now, convert to direct
        img = convert(Image, img)
    end
    imgw = map(mapi, img)
    imgw = permutedims_horizontal(imgw)
    have_color = colordim(imgw)!=0
    if ndims(imgw) > 3+have_color
        error("At most 3 dimensions are supported")
    end
    wand = LibMagick.MagickWand()
    if haskey(img, "IMcs")
        cs = img["IMcs"]
    else
        cs = colorspace(imgw)
        if in(cs, ("RGB", "RGBA", "ARGB", "BGRA"))
            cs = LibMagick.libversion > v"6.7.5" ? "sRGB" : "RGB"
        end
    end
    channelorder = colorspace(imgw)
    if channelorder == "Gray"
        channelorder = "I"
    elseif channelorder == "GrayAlpha"
        channelorder = "IA"
    end
    tmp = to_explicit(to_contiguous(data(imgw)))
    LibMagick.constituteimage(tmp, wand, cs, channelorder)
    if quality != nothing
        LibMagick.setimagecompressionquality(wand, quality)
    end
    LibMagick.resetiterator(wand)
    wand
end

# ImageMagick mapinfo client. Converts to RGB and uses Ufixed.
mapinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{T}) = MapNone{T}()
mapinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{T}) = ClampMinMax(Ufixed8, zero(T), one(T))
for ACV in (ColorValue, AbstractRGB,AbstractGray)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && !(CV.abstract)) || continue
        CVnew = CV<:AbstractGray ? Gray : RGB
        @eval mapinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{$CV{T}}) = MapNone{$CVnew{T}}()
        @eval mapinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{$CV{T}}) =
            Clamp{$CVnew{Ufixed8}}()
        CVnew = CV<:AbstractGray ? Gray : BGR
        for AC in subtypes(AbstractAlphaColorValue)
            (length(AC.parameters) == 2 && !(AC.abstract)) || continue
            @eval mapinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{$AC{$CV{T},T}}) = MapNone{$AC{$CVnew{T},T}}()
            @eval mapinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{$AC{$CV{T},T}}) = Clamp{$AC{$CVnew{Ufixed8}, Ufixed8}}()
        end
    end
end
mapinfo(::Type{ImageMagick}, img::AbstractArray{RGB24}) = MapNone{RGB{Ufixed8}}()
mapinfo(::Type{ImageMagick}, img::AbstractArray{ARGB32}) = MapNone{BGRA{Ufixed8}}()


# Make the data contiguous in memory, this is necessary for imagemagick since it doesn't handle stride.
to_contiguous(A::AbstractArray) = A
to_contiguous(A::SubArray) = copy(A)

to_explicit(A::AbstractArray) = A
to_explicit{T<:Ufixed}(A::AbstractArray{T}) = reinterpret(FixedPointNumbers.rawtype(T), A)
to_explicit{T<:Ufixed}(A::AbstractArray{RGB{T}}) = reinterpret(FixedPointNumbers.rawtype(T), A, tuple(3, size(A)...))
to_explicit{T<:FloatingPoint}(A::AbstractArray{RGB{T}}) = to_explicit(map(ClipMinMax(RGB{Ufixed8}, zero(RGB{T}), one(RGB{T})), A))
to_explicit{T<:Ufixed}(A::AbstractArray{Gray{T}}) = reinterpret(FixedPointNumbers.rawtype(T), A, size(A))
to_explicit{T<:FloatingPoint}(A::AbstractArray{Gray{T}}) = to_explicit(map(ClipMinMax(Gray{Ufixed8}, zero(Gray{T}), one(Gray{T})), A))
to_explicit{T<:Ufixed}(A::AbstractArray{GrayAlpha{T}}) = reinterpret(FixedPointNumbers.rawtype(T), A, tuple(2, size(A)...))
to_explicit{T<:FloatingPoint}(A::AbstractArray{GrayAlpha{T}}) = to_explicit(map(ClipMinMax(GrayAlpha{Ufixed8}, zero(GrayAlpha{T}), one(GrayAlpha{T})), A))
to_explicit{T<:Ufixed}(A::AbstractArray{BGRA{T}}) = reinterpret(FixedPointNumbers.rawtype(T), A, tuple(4, size(A)...))
to_explicit{T<:FloatingPoint}(A::AbstractArray{BGRA{T}}) = to_explicit(map(ClipMinMax(BGRA{Ufixed8}, zero(BGRA{T}), one(BGRA{T})), A))
to_explicit{T<:Ufixed}(A::AbstractArray{RGBA{T}}) = reinterpret(FixedPointNumbers.rawtype(T), A, tuple(4, size(A)...))
to_explicit{T<:FloatingPoint}(A::AbstractArray{RGBA{T}}) = to_explicit(map(ClipMinMax(RGBA{Ufixed8}, zero(RGBA{T}), one(RGBA{T})), A))

# Write values in permuted order
let method_cache = Dict()
global writepermuted
function writepermuted(stream, img, mapi::MapInfo, perm; gray2color::Bool = false)
    cd = colordim(img)
    key = (perm, cd, gray2color)
    if !haskey(method_cache, key)
        mapfunc = cd > 0 ? (:map1) : (:map)
        loopsyms = [symbol(string("i_",d)) for d = 1:ndims(img)]
        body = gray2color ? quote
                g = $mapfunc(mapi, img[$(loopsyms...)])
                write(stream, g)
                write(stream, g)
                write(stream, g)
            end : quote
                write(stream, $mapfunc(mapi, img[$(loopsyms...)]))
            end
        loopargs = [:($(loopsyms[d]) = 1:size(img, $d)) for d = 1:ndims(img)]
        loopexpr = Expr(:for, Expr(:block, loopargs[perm[end:-1:1]]...), body)
        f = @eval begin
            local _writefunc_
            function _writefunc_(stream, img, mapi)
                $loopexpr
            end
        end
    else
        f = method_cache[key]
    end
    f(stream, img, mapi)
    nothing
end
end

function write{T<:Ufixed}(io::IO, c::AbstractRGB{T})
    write(io, reinterpret(c.r))
    write(io, reinterpret(c.g))
    write(io, reinterpret(c.b))
end

function write(io::IO, c::RGB24)
    write(io, red(c))
    write(io, green(c))
    write(io, blue(c))
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
    skipchars(stream, isspace, linecomment='#')
    maxvalline = strip(readline(stream))
    parse(Int, maxvalline)
end

function imread{S<:IO}(stream::S, ::Type{PPMBinary})
    w, h = parse_netpbm_size(stream)
    maxval = parse_netpbm_maxval(stream)
    local dat
    if maxval <= 255
        datraw = read(stream, Ufixed8, 3, w, h)
        dat = reinterpret(RGB{Ufixed8}, datraw, (w, h))
    elseif maxval <= typemax(Uint16)
        # read first as Uint16 so the loop is type-stable, then convert to Ufixed
        datraw = Array(Uint16, 3, w, h)
        # there is no endian standard, but netpbm is big-endian
        if !is_little_endian
            for indx = 1:3*w*h
                datraw[indx] = read(stream, Uint16)
            end
        else
            for indx = 1:3*w*h
                datraw[indx] = bswap(read(stream, Uint16))
            end
        end
        # Determine the appropriate Ufixed type
        T = ufixedtype[ceil(Int, log2(maxval)/2)<<1]
        dat = reinterpret(RGB{T}, datraw, (w, h))
    else
        error("Image file may be corrupt. Are there really more than 16 bits in this image?")
    end
    T = eltype(dat)
    Image(dat, @compat Dict("spatialorder" => ["x", "y"], "pixelspacing" => [1,1]))
end

function imread{S<:IO}(stream::S, ::Type{PGMBinary})
    w, h = parse_netpbm_size(stream)
    maxval = parse_netpbm_maxval(stream)
    local dat
    if maxval <= 255
        dat = read(stream, Ufixed8, w, h)
    elseif maxval <= typemax(Uint16)
        datraw = Array(Uint16, w, h)
        if !is_little_endian
            for indx = 1:w*h
                datraw[indx] = read(stream, Uint16)
            end
        else
            for indx = 1:w*h
                datraw[indx] = bswap(read(stream, Uint16))
            end
        end
        # Determine the appropriate Ufixed type
        T = ufixedtype[ceil(Int, log2(maxval)/2)<<1]
        dat = reinterpret(RGB{T}, datraw, (w, h))
    else
        error("Image file may be corrupt. Are there really more than 16 bits in this image?")
    end
    T = eltype(dat)
    Image(dat, @compat Dict("colorspace" => "Gray", "spatialorder" => ["x", "y"], "pixelspacing" => [1,1]))
end

function imread{S<:IO}(stream::S, ::Type{PBMBinary})
    w, h = parse_netpbm_size(stream)
    dat = BitArray(w, h)
    nbytes_per_row = ceil(Int, w/8)
    for irow = 1:h, j = 1:nbytes_per_row
        tmp = read(stream, Uint8)
        offset = (j-1)*8
        for k = 1:min(8, w-offset)
            dat[offset+k, irow] = (tmp>>>(8-k))&0x01
        end
    end
    Image(dat, @compat Dict("spatialorder" => ["x", "y"], "pixelspacing" => [1,1]))
end

function imwrite(img, filename::String, ::Type{PPMBinary})
    open(filename, "w") do stream
        write(stream, "P6\n")
        write(stream, "# ppm file written by Julia\n")
        imwrite(img, stream, PPMBinary)
    end
end

pnmmax{T<:FloatingPoint}(::Type{T}) = 255
pnmmax{T<:Ufixed}(::Type{T}) = reinterpret(FixedPointNumbers.rawtype(T), one(T))
pnmmax{T<:Unsigned}(::Type{T}) = typemax(T)

function imwrite{T<:ColorValue}(img::AbstractArray{T}, s::IO, ::Type{PPMBinary}, mapi = mapinfo(ImageMagick, img))
    w, h = widthheight(img)
    TE = eltype(T)
    mx = pnmmax(TE)
    write(s, "$w $h\n$mx\n")
    p = permutation_horizontal(img)
    writepermuted(s, img, mapi, p; gray2color = T <: AbstractGray)
end

function imwrite{T}(img::AbstractArray{T}, s::IO, ::Type{PPMBinary}, mapi = mapinfo(ImageMagick, img))
    w, h = widthheight(img)
    cs = colorspace(img)
    in(cs, ("RGB", "Gray")) || error("colorspace $cs not supported")
    mx = pnmmax(T)
    write(s, "$w $h\n$mx\n")
    p = permutation_horizontal(img)
    writepermuted(s, img, mapi, p; gray2color = cs == "Gray")
end


function parseints(line, n)
    ret = Array(Int, n)
    pos = 1
    for i = 1:n
        pos2 = search(line, ' ', pos)
        if pos2 == 0
            pos2 = length(line)+1
        end
        ret[i] = parse(Int, line[pos:pos2-1])
        pos = pos2+1
        if pos > length(line) && i < n
            error("Line terminated without finding all ", n, " integers")
        end
    end
    tuple(ret...)

end

# Permute to a color, horizontal, vertical, ... storage order (with time always last)
function permutation_horizontal(img)
    cd = colordim(img)
    td = timedim(img)
    p = spatialpermutation(["x", "y"], img)
    if cd != 0
        p[p .>= cd] += 1
        insert!(p, 1, cd)
    end
    if td != 0
        push!(p, td)
    end
    p
end

permutedims_horizontal(img) = permutedims(img, permutation_horizontal(img))


### Register formats for later loading here
type Dummy <: ImageFileType; end
add_image_file_format(".dummy", b"Dummy Image", Dummy, "dummy.jl")

# NRRD image format
type NRRDFile <: ImageFileType end
add_image_file_format(".nrrd", b"NRRD", NRRDFile, "nrrd.jl")
# NRRD header only
type NRRDHeader <: ImageFileType end
add_image_file_format(".nhdr", b"NRRD", NRRDHeader, "nrrd.jl")

# Andor Technologies SIF file format
type AndorSIF <: Images.ImageFileType end
add_image_file_format(".sif", b"Andor Technology Multi-Channel File", AndorSIF, "SIF.jl")

# Imagine file format (http://holylab.wustl.edu, "Software" tab)
type ImagineFile <: ImageFileType end
add_image_file_format(".imagine", b"IMAGINE", ImagineFile, "Imagine.jl")

# PCO b16 image format
type B16File <: ImageFileType end
add_image_file_format(".b16", b"PCO-", B16File, "b16.jl")
