module Imagine

using Images
using SIUnits, SIUnits.ShortUnits, Compat

import Images.imread
import Base.convert

export imagine2nrrd, Micron

Micron = SIUnits.NonSIUnit{typeof(Meter),:µm}()
convert(::Type{SIUnits.SIQuantity},::typeof(Micron)) = Micro*Meter

function imread{S<:IO}(s::S, ::Type{Images.ImagineFile})
    h = parse_header(s, Images.ImagineFile)
    filename = s.name[7:end-1]
    basename, ext = splitext(filename)
    camfilename = basename*".cam"
    T = h["pixel data type"]
    sz = [h["image width"], h["image height"], h["frames per stack"], h["nStacks"]]
    if sz[4] == 1
        sz = sz[1:3]
        if sz[3] == 1
            sz = sz[1:2]
        end
    end
    havez = h["frames per stack"] > 1
    havet = h["nStacks"] > 1
    # Check that the file size is consistent with the expected size
    if !isfile(camfilename)
        warn("Cannot open ", camfilename)
        data = Array(T, sz[1], sz[2], sz[3], 0)
    else
        fsz = filesize(camfilename)
        n_stacks = sz[end]
        if fsz != sizeof(T)*prod(map(Int64,sz))  # guard against overflow on 32bit
            warn("Size of image file is different from expected value")
            n_stacks = ifloor(fsz / sizeof(T) / prod(sz[1:end-1]))
        end
        if sizeof(T)*prod(map(Int64,sz[1:end-1]))*n_stacks > typemax(Uint)
            warn("File size is too big to mmap on 32bit")
            n_stacks = ifloor(fsz / sizeof(T) / typemax(Uint))
        end
        if n_stacks < sz[end]
            println("Truncating to ", n_stacks, length(sz) == 4 ? " stacks" : " frames")
            sz[end] = n_stacks
        end
        sc = open(camfilename, "r")
        data = mmap_array(T, ntuple(i->sz[i], length(sz)), sc)
    end
    um_per_pixel = h["um per pixel"]*µm
    pstart = h["piezo"]["stop position"]
    pstop = h["piezo"]["start position"]
    if length(sz)>2
        dz = abs(pstart - pstop)/sz[3]
    else dz = 0.0 end

    props = @compat Dict(
        "spatialorder" => havez ? ["x", "l", "z"] : ["x", "l"],
        "colorspace" => "Gray",
        "pixelspacing" => havez ? [um_per_pixel, um_per_pixel, dz] : [um_per_pixel, um_per_pixel],
        "limits" => (@compat(UInt16(0)), @compat(UInt16(2^h["original image depth"]-1))),
        "imagineheader" => h,
        "suppress" => Set(Any["imagineheader"])
    )
    if havet
        props["timedim"] = havez ? 4 : 3
    end
    Image(data, props)
end

abstract Endian
type LittleEndian <: Endian; end
type BigEndian <: Endian; end
const endian_dict = @compat Dict("l"=>LittleEndian, "b"=>BigEndian)
const nrrd_endian_dict = @compat Dict(LittleEndian=>"little",BigEndian=>"big")
parse_endian(s::ASCIIString) = endian_dict[lowercase(s)]

function parse_vector_int(s::String)
    ss = @compat split(s, r"[ ,;]", keep=false)
    v = Array(Int, length(ss))
    for i = 1:length(ss)
        v[i] = parse(Int,ss[i])
    end
    return v
end

const bitname_dict = @compat Dict(
  "int8"      => Int8,
  "uint8"     => Uint8,
  "int16"     => Int16,
  "uint16"    => Uint16,
  "int32"     => Int32,
  "uint32"    => Uint32,
  "int64"     => Int64,
  "uint64"    => Uint64,
  "float16"   => Float16,
  "float32"   => Float32,
  "single"    => Float32,
  "float64"   => Float64,
  "double"    => Float64)

parse_bittypename(s::ASCIIString) = bitname_dict[lowercase(s)]

function float64_or_empty(s::ASCIIString)
    if isempty(s)
        return NaN
    else
        return parse(Float64,s)
    end
end

function parse_quantity_or_empty(s::ASCIIString)
    if isempty(s)
        return NaN
    else
        return parse_quantity(s)
    end
end

_unit_string_dict = @compat Dict("um" => Micro*Meter, "s" => Second, "us" => Micro*Second, "MHz" => Mega*Hertz)
function parse_quantity(s::String, strict::Bool = true)
    # Find the last character of the numeric component
    m = match(r"[0-9\.\+-](?![0-9\.\+-])", s)
    if m == nothing
        error("String does not have a 'value unit' structure")
    end
    val = parse(Float64, s[1:m.offset])
    ustr = strip(s[m.offset+1:end])
    if isempty(ustr)
        if strict
            error("String does not have a 'value unit' structure")
        else
            return val
        end
    end
    val * _unit_string_dict[ustr]
end

# Read and parse a *.imagine file (an Imagine header file)
const compound_fields = Any["piezo", "binning"]
const field_key_dict = @compat Dict{String,Function}(
    "header version"               => x->parse(Float64,x),
    "app version"                  => identity,
    "date and time"                => identity,
    "rig"                          => identity,
    "byte order"                   => parse_endian,
    "stimulus file content"        => identity,  # stimulus info parsed separately
    "comment"                      => identity,
    "ai data file"                 => identity,
    "image data file"              => identity,
    "start position"               => parse_quantity,
    "stop position"                => parse_quantity,
    "bidirection"                  => x->parse(Int,x) != 0,
    "output scan rate"             => x->parse_quantity(x, false),
    "nscans"                       => x->parse(Int,x),
    "channel list"                 => parse_vector_int,
    "label list"                   => identity,
    "scan rate"                    => x->parse_quantity(x, false),
    "min sample"                   => x->parse(Int,x),
    "max sample"                   => x->parse(Int,x),
    "min input"                    => x->parse(Float64,x),
    "max input"                    => x->parse(Float64,x),
    "original image depth"         => x->parse(Int,x),
    "saved image depth"            => x->parse(Int,x),
    "image width"                  => x->parse(Int,x),
    "image height"                 => x->parse(Int,x),
    "number of frames requested"   => x->parse(Int,x),
    "nStacks"                      => x->parse(Int,x),
    "idle time between stacks"     => parse_quantity,
    "pre amp gain"                 => float64_or_empty,
    "EM gain"                      => float64_or_empty,
    "gain"                         => float64_or_empty,
    "exposure time"                => parse_quantity,
    "vertical shift speed"         => parse_quantity_or_empty,
    "vertical clock vol amp"       => x->parse(Float64,x),
    "readout rate"                 => parse_quantity_or_empty,
    "pixel order"                  => identity,
    "frame index offset"           => x->parse(Int,x),
    "frames per stack"             => x->parse(Int,x),
    "pixel data type"              => parse_bittypename,
    "camera"                       => identity,
    "um per pixel"                 => x->parse(Float64,x),
    "hbin"                         => x->parse(Int,x),
    "vbin"                         => x->parse(Int,x),
    "hstart"                       => x->parse(Int,x),
    "hend"                         => x->parse(Int,x),
    "vstart"                       => x->parse(Int,x),
    "vend"                         => x->parse(Int,x),
    "angle from horizontal (deg)"  => float64_or_empty)

function parse_header(s::IOStream, ::Type{Images.ImagineFile})
    headerdict = Dict{ASCIIString, Any}()
    for this_line = eachline(s)
        this_line = strip(this_line)
        if !isempty(this_line) && !ismatch(r"\[.*\]", this_line)
            # Split on =
            m = match(r"=", this_line)
            if m.offset < 2
                error("Line does not contain =")
            end
            k = this_line[1:m.offset-1]
            v = this_line[m.offset+1:end]
            if in(k, compound_fields)
                thisdict = Dict{ASCIIString, Any}()
                # Split on semicolon
                strs = split(v, r";")
                for i = 1:length(strs)
                    substrs = split(strs[i], r":")
                    @assert length(substrs) == 2
                    k2 = strip(substrs[1])
                    func = field_key_dict[k2]
                    v2 = strip(substrs[2])
                    try
                        thisdict[k2] = func(v2)
                    catch err
                        println("Error processing key ", k2, " with value ", v2)
                        rethrow(err)
                    end
                end
                headerdict[k] = thisdict
            else
                func = field_key_dict[k]
                try
                    headerdict[k] = func(v)
                catch err
                    println("Error processing key ", k, " with value ", v)
                    rethrow(err)
                end
            end
        end
    end
    return headerdict
end

function imagine2nrrd(sheader::IO, h::Dict{ASCIIString, Any}, datafilename = nothing)
    println(sheader, "NRRD0001")
    T = h["pixel data type"]
    if T<:FloatingPoint
        println(sheader, "type: ", (T == Float32) ? "float" : "double")
    else
        println(sheader, "type: ", lowercase(string(T)))
    end
    sz = [h["image width"], h["image height"], h["frames per stack"], h["nStacks"]]
    kinds = ["space", "space", "space", "time"]
    if sz[end] == 1
        sz = sz[1:3]
        kinds = kinds[[1,2,4]]
    end
    println(sheader, "dimension: ", length(sz))
    print(sheader, "sizes:")
    for z in sz
        print(sheader, " ", z)
    end
    print(sheader, "\nkinds:")
    for k in kinds
        print(sheader, " ", k)
    end
    print(sheader, "\n")
    println(sheader, "encoding: raw")
    println(sheader, "endian: ", nrrd_endian_dict[h["byte order"]])
    if isa(datafilename, String)
        println(sheader, "data file: ", datafilename)
    end
    sheader
end

function imagine2nrrd(nrrdname::String, h::Dict{ASCIIString, Any}, datafilename = nothing)
    sheader = open(nrrdname, "w")
    imagine2nrrd(sheader, h, datafilename)
    close(sheader)
end

end
