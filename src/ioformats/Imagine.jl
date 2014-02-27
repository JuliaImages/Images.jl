module Imagine

using Images
using SIUnits, SIUnits.ShortUnits

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
    end
    havez = length(sz) == 4
    # Check that the file size is consistent with the expected size
    if !isfile(camfilename)
        warn("Cannot open ", camfilename)
        data = Array(T, sz[1], sz[2], sz[3], 0)
    else
        fsz = filesize(camfilename)
        n_stacks = sz[end]
        if fsz != sizeof(T)*prod(int64(sz))  # guard against overflow on 32bit
            warn("Size of image file is different from expected value")
            n_stacks = ifloor(fsz / sizeof(T) / prod(sz[1:end-1]))
        end
        if sizeof(T)*prod(int64(sz[1:end-1]))*n_stacks > typemax(Uint)
            warn("File size is too big to mmap on 32bit")
            n_stacks = ifloor(fsz / sizeof(T) / typemax(Uint))
        end
        if n_stacks < sz[end]
            println("Truncating to ", n_stacks, length(sz) == 4 ? " stacks" : " frames")
            sz[end] = n_stacks
        end
        sc = open(camfilename, "r")
        data = mmap_array(T, ntuple(length(sz), i->sz[i]), sc)
    end
    um_per_pixel = h["um per pixel"]*µm
    pstart = h["piezo"]["stop position"]
    pstop = h["piezo"]["start position"]
    dz = abs(pstart - pstop)/sz[3]

    Image(data, ["spatialorder" => havez ? ["x", "l", "z"] : ["x", "l"],
                 "timedim" => havez ? 4 : 3,
                 "colorspace" => "Gray",
                 "pixelspacing" => havez ? [um_per_pixel, um_per_pixel, dz] : [um_per_pixel, um_per_pixel],
                 "limits" => (uint16(0), uint16(2^h["original image depth"]-1)),
                 "imagineheader" => h,
                 "suppress" => Set({"imagineheader"})])
end    

abstract Endian
type LittleEndian <: Endian; end
type BigEndian <: Endian; end
const endian_dict = Dict(("l", "b"), (LittleEndian, BigEndian))
const nrrd_endian_dict = Dict((LittleEndian, BigEndian), ("little","big"))
parse_endian(s::ASCIIString) = endian_dict[lowercase(s)]

function parse_vector_int(s::String)
    ss = split(s, r"[ ,;]", false)
    v = Array(Int, length(ss))
    for i = 1:length(ss)
        v[i] = int(ss[i])
    end
    return v
end

const bitname_table = {
  ("int8",      Int8)
  ("uint8",     Uint8)
  ("int16",     Int16)
  ("uint16",    Uint16)
  ("int32",     Int32)
  ("uint32",    Uint32)
  ("int64",     Int64)
  ("uint64",    Uint64)
  ("float32",   Float32)
  ("single",    Float32)
  ("float64",   Float64)
  ("double",    Float64)
}
bitname_dict = Dict{ASCIIString, Any}()
for l in bitname_table
    bitname_dict[l[1]] = l[2]
end
parse_bittypename(s::ASCIIString) = bitname_dict[lowercase(s)]

function float64_or_empty(s::ASCIIString)
    if isempty(s)
        return NaN
    else
        return float64(s)
    end
end

function parse_quantity_or_empty(s::ASCIIString)
    if isempty(s)
        return NaN
    else
        return parse_quantity(s)
    end
end

_unit_string_dict = ["um" => Micro*Meter, "s" => Second, "us" => Micro*Second, "MHz" => Mega*Hertz]
function parse_quantity(s::String, strict::Bool = true)
    # Find the last character of the numeric component
    m = match(r"[0-9\.\+-](?![0-9\.\+-])", s)
    if m == nothing
        error("String does not have a 'value unit' structure")
    end
    val = float64(s[1:m.offset])
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
const compound_fields = {"piezo", "binning"}
const field_parser_list = {
    "header version"               float64;
    "app version"                  identity;
    "date and time"                identity;
    "rig"                          identity;
    "byte order"                   parse_endian;
    "stimulus file content"        identity;  # stimulus info parsed separately
    "comment"                      identity;
    "ai data file"                 identity;
    "image data file"              identity;
    "start position"               parse_quantity;
    "stop position"                parse_quantity;
    "bidirection"                  x->bool(int(x));
    "output scan rate"             x->parse_quantity(x, false);
    "nscans"                       int;
    "channel list"                 parse_vector_int;
    "label list"                   identity;
    "scan rate"                    x->parse_quantity(x, false);
    "min sample"                   int;
    "max sample"                   int;
    "min input"                    float64;
    "max input"                    float64;
    "original image depth"         int;
    "saved image depth"            int;
    "image width"                  int;
    "image height"                 int;
    "number of frames requested"   int;
    "nStacks"                      int;
    "idle time between stacks"     parse_quantity;
    "pre amp gain"                 float64_or_empty;
    "EM gain"                      float64_or_empty;
    "gain"                         float64_or_empty;
    "exposure time"                parse_quantity;
    "vertical shift speed"         parse_quantity_or_empty;
    "vertical clock vol amp"       float64;
    "readout rate"                 parse_quantity_or_empty;
    "pixel order"                  identity;
    "frame index offset"           int;
    "frames per stack"             int;
    "pixel data type"              parse_bittypename;
    "camera"                       identity;
    "um per pixel"                 float64;
    "hbin"                         int;
    "vbin"                         int;
    "hstart"                       int;
    "hend"                         int;
    "vstart"                       int;
    "vend"                         int;
    "angle from horizontal (deg)"  float64_or_empty;
}
const field_key_dict = (String=>Function)[field_parser_list[i,1] => field_parser_list[i,2] for i = 1:size(field_parser_list,1)]

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
