module NRRD

using Images, SIUnits, SIUnits.ShortUnits
import Images: imread, imwrite
import Zlib

typedict = [
    "signed char" => Int8,
    "int8" => Int8,
    "int8_t" => Int8,
    "uchar" => Uint8,
    "unsigned char" => Uint8,
    "uint8" => Uint8,
    "uint8_t" => Uint8,
    "short" => Int16,
    "short int" => Int16,
    "signed short" => Int16,
    "signed short int" => Int16,
    "int16" => Int16,
    "int16_t" => Int16,
    "ushort" => Uint16,
    "unsigned short" => Uint16,
    "unsigned short int" => Uint16,
    "uint16" => Uint16,
    "uint16_t" => Uint16,
    "int" => Int32,
    "signed int" => Int32,
    "int32" => Int32,
    "int32_t" => Int32,
    "uint" => Uint32,
    "unsigned int" => Uint32,
    "uint32" => Uint32,
    "uint32_t" => Uint32,
    "longlong" => Int64,
    "long long" => Int64,
    "long long int" => Int64,
    "signed long long" => Int64,
    "signed long long int" => Int64,
    "int64" => Int64,
    "int64_t" => Int64,
    "ulonglong" => Uint64,
    "unsigned long long" => Uint64,
    "unsigned long long int" => Uint64,
    "uint64" => Uint64,
    "uint64_t" => Uint64,
    "float16" => Float16,
    "float" => Float32,
    "double" => Float64]

spacedimdict = [
    "right-anterior-superior" => 3,
    "ras" => 3,
    "left-anterior-superior" => 3,
    "las" => 3,
    "left-posterior-superior" => 3,
    "lps" => 3,
    "right-anterior-superior-time" => 4,
    "rast" => 4,
    "left-anterior-superior-time" => 4,
    "last" => 4,
    "left-posterior-superior-time" => 4,
    "lpst" => 4,
    "scanner-xyz" => 3,
    "scanner-xyz-time" => 4,
    "3d-right-handed" => 3,
    "3d-left-handed" => 3,
    "3d-right-handed-time" => 4,
    "3d-left-handed-time" => 4]

function myendian()
    if ENDIAN_BOM == 0x04030201
        return "little"
    elseif ENDIAN_BOM == 0x01020304
        return "big"
    end
end

function imread{S<:IO}(stream::S, ::Type{Images.NRRDFile})
    version = ascii(read(stream, Uint8, 4))
    skipchars(stream,isspace)
    header = Dict{ASCIIString, UTF8String}()
    props = Dict{ASCIIString, Any}()
    comments = Array(ASCIIString, 0)
    # Read until we encounter a blank line, which is the separator between the header and data
    line = strip(readline(stream))
    while !isempty(line)
        if line[1] != '#'
            key, value = split(line, ":")
            if !isempty(value) && value[1] == '='
                # This is a NRRD key/value pair, insert straight into props
                props[key] = value[2:end]
            else
                header[lowercase(key)] = strip(value)
            end
        else
            cmt = strip(lstrip(line, collect("#")))
            if !isempty(cmt)
                push!(comments, cmt)
            end
        end
        line = strip(readline(stream))
    end
    # Fields that go straight into the properties dict
    if haskey(header, "content")
        props["content"] = header["content"]
    end
    # Check to see whether the data are in an external file
    sdata = stream
    if haskey(header, "data file")
        path = dirname(stream2name(sdata))
        sdata = open(joinpath(path, header["data file"]))
    elseif haskey(header, "datafile")
        path = dirname(stream2name(sdata))
        sdata = open(joinpath(path, header["datafile"]))
    end
    if in(header["encoding"], ("gzip", "gz"))
        sdata = Zlib.Reader(sdata)
    end
    # Parse properties and read the data
    nd = int(header["dimension"])
    sz = parse_vector_int(header["sizes"])
    length(sz) == nd || error("parsing of sizes: $(header["sizes"]) is inconsistent with $nd dimensions")
    T = typedict[header["type"]]
    local A
    need_bswap = haskey(header, "endian") && header["endian"] != myendian() && sizeof(T) > 1
    if header["encoding"] == "raw" && prod(sz) > 10^8 && !need_bswap
        # Use memory-mapping for large files
        fn = stream2name(sdata)
        datalen = div(filesize(fn) - position(sdata), sizeof(T))
        strds = [1,cumprod(sz)]
        k = length(sz)
        sz[k] = div(datalen, strds[k])
        while sz[k] == 0 && k > 1
            pop!(sz)
            k -= 1
            sz[k] = div(datalen, strds[k])
        end
        A = mmap_array(T, tuple(sz...), sdata, position(sdata))
    elseif header["encoding"] == "raw" || in(header["encoding"], ("gzip", "gz"))
        A = read(sdata, T, sz...)
        if need_bswap
            A = reshape([bswap(a) for a in A], size(A))
        end
    else
      error("\"", header["encoding"], "\" encoding not supported.")
    end
    isspatial = trues(nd)
    # Optional fields
    if haskey(header, "kinds")
        kinds = split(header["kinds"], " ")
        length(kinds) == nd || error("parsing of kinds: $(header["kinds"]) is inconsistent with $nd dimensions")
        for i = 1:nd
            k = kinds[i]
            if k == "time"
                props["timedim"] = i
                isspatial[i] = false
            elseif in(k, ("3-color","4-color","list","point","vector","covariant-vector","normal","2-vector","3-vector","4-vector","3-gradient","3-normal","scalar","complex","quaternion"))
                props["colordim"] = i
                props["colorspace"] = k
                isspatial[i] = false
            elseif k == "RGB-color"
                props["colordim"] = i
                props["colorspace"] = "RGB"
                isspatial[i] = false
            elseif k == "HSV-color"
                props["colordim"] = i
                props["colorspace"] = "HSV"
                isspatial[i] = false
            elseif k == "RGBA-color"
                props["colordim"] = i
                props["colorspace"] = "RGBA"
                isspatial[i] = false
            elseif contains(k, "matrix")
                error("matrix types are not yet supported")
            end
        end
    end
    if !haskey(props, "colordim")
        props["colorspace"] = "Gray"
    end
    if haskey(header, "min") || haskey(header, "max")
        mn = typemin(T)
        mx = typemax(T)
        if T <: Integer
            if haskey(header, "min")
                mn = convert(T, parseint(header["min"]))
            end
            if haskey(header, "max")
                mx = convert(T, parseint(header["max"]))
            end
        else
            if haskey(header, "min")
                mn = convert(T, parsefloat(header["min"]))
            end
            if haskey(header, "max")
                mx = convert(T, parsefloat(header["max"]))
            end
        end
        props["limits"] = (mn, mx)
    end
    if haskey(header, "labels")
        lbls = parse_vector_strings(header["labels"])
        props["spatialorder"] = lbls[isspatial]
    else
        props["spatialorder"] = [string(char(97+(i+22)%26)) for i = 1:sum(isspatial)]
    end
    spacedim = nd
    if haskey(header, "space")
        props["space"] = header["space"]
        spacedim = spacedimdict[lowercase(header["space"])]
        spacedim == sum(isspatial) || error("spacedim $spacedim disagrees with isspatial=$isspatial")
    elseif haskey(header, "space dimension")
        spacedim = int(header["space dimension"])
        spacedim == sum(isspatial) || error("spacedim $spacedim disagrees with isspatial=$isspatial")
    end
    units = Array(Union(SIUnits.SIUnit,SIUnits.SIQuantity), 0)
    if haskey(header, "space units")
        ustrs = parse_vector_strings(header["space units"])
        length(ustrs) == spacedim || error("parsing of space units: $(header["space units"]) is inconsistent with $spacedim space dimensions")
        for i = 1:spacedim
            try
                push!(units, eval(symbol(ustrs[i])))
            catch
                evalworked = false
                warn("Could not evaluate unit string $(ustrs[i])")
            end
        end
        if length(units) < spacedim
            units = Array(SIUnits.SIQuantity, 0)  # Don't use any units
            props["spaceunits"] = ustrs          # ...but store the string representation
        end
    end
    if haskey(header, "space directions")
        # space directions are per-axis, but can be "none"
        sd = split(header["space directions"], " ")
        length(sd) == nd || error("parsing of space directions: $(header["space directions"]) is inconsistent with $nd dimensions")
        sdf = Array(Any, 0)
        for i = 1:nd
            if sd[i] == "none"
                isspatial[i] && error("Dimension $i is spatial, but has space directions \"none\".")
            else
                v = parse_vector_float(sd[i][2:end-1])
                if !isempty(units)
                    vu = [v[i]*units[i] for i = 1:spacedim]
                    push!(sdf, vu)
                else
                    push!(sdf, v)
                end
            end
        end
        props["spacedirections"] = sdf
        # If spacedirections is diagonal, put that into pixelspacing
        have_pixelspacing = true
        for i = 1:length(sdf)
            v = sdf[i]
            for j = 1:spacedim
                if i != j
                    have_pixelspacing &= 0*v[j] == v[j]  # this is units-safe
                end
            end
        end
        if have_pixelspacing
            ps = [sdf[i][i] for i = 1:spacedim]
            props["pixelspacing"] = ps
        end
    elseif haskey(header, "spacings")
        ps = parse_vector_float(header["spacings"])
        length(ps) == nd || error("parsing of spacings: $(header["spacings"]) is inconsistent with $nd dimensions")
        pss = ps[isspatial]
        if !isempty(units)
            unitss = units[isspatial]
            vu = [pss[i]*units[i] for i = 1:length(pss)]
            props["pixelspacing"] = vu
        else
            props["pixelspacing"] = pss
        end
    end
    if !isempty(comments)
        props["comments"] = comments
    end
    Image(A, props)
end

function imwrite(img, sheader::IO, ::Type{Images.NRRDFile}; props::Dict = Dict{ASCIIString,Any}())
    println(sheader, "NRRD0001")
    # Write the datatype
    T = get(props, "type", eltype(data(img)))
    if T<:FloatingPoint
        println(sheader, "type: ", (T == Float32) ? "float" :
                                   (T == Float64) ? "double" :
                                   (T == Float16) ? "float16" :
                                   error("Can't write type $T"))
    else
        println(sheader, "type: ", lowercase(string(T)))
    end
    # Extract size and kinds
    sz = get(props, "sizes", size(img))
    kinds = ["space" for i = 1:length(sz)]
    td = timedim(img)
    if td != 0
        kinds[td] = "time"
    end
    cd = colordim(img)
    if cd != 0
        if colorspace(img) == "RGB"
            kinds[cd] = "RGB-color"
        elseif size(img, cd) == 3
            kinds[cd] = "3-color"
        else
            kinds[cd] = "list"
        end
    end
    kinds = get(props, "kinds", kinds)
    # Write size and kinds
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
    println(sheader, "encoding: ", get(props, "encoding", "raw"))
    println(sheader, "endian: ", get(props, "endian", ENDIAN_BOM == 0x04030201 ? "little" : "big"))
    ps = get(props, "pixelspacing", pixelspacing(img))
    ps = convert(Vector{Any}, ps)
    index = sort([cd, td])
    index[1] > 0 && insert!(ps, index[1], "nan")
    index[2] > 0 && insert!(ps, index[2], "nan")
    print(sheader, "spacings:")
    printunits = false
    for x in ps
        if isa(x, SIUnits.SIQuantity)
            printunits = true
            print(sheader, " ", x.val)
        else
            print(sheader, " ", x)
        end
    end
    print(sheader,"\n")
    if printunits
        print(sheader, "space units:")
        for x in ps
            print(sheader," \"", strip(string(SIUnits.unit(x))), "\"")
        end
        print(sheader, "\n")
    end
    datafilename = get(props, "datafile", "")
    if isempty(datafilename)
        datafilename = get(props, "data file", "")
    end
    if isempty(datafilename)
        write(sheader, "\n")
        write(sheader, data(img))
    else
        println(sheader, "data file: ", datafilename)
    end
    sheader
end

function parse_vector_int(s::String)
    ss = split_nokeep(s, r"[ ,;]")
    v = Array(Int, length(ss))
    for i = 1:length(ss)
        v[i] = int(ss[i])
    end
    return v
end

function parse_vector_float(s::String)
    ss = split_nokeep(s, r"[ ,;]")
    v = Array(Float64, length(ss))
    for i = 1:length(ss)
        v[i] = float(ss[i])
    end
    return v
end

function parse_vector_strings(s::String)
    (first(s) == '"' && last(s) == '"') || error("Strings must be delimited with quotes")
    split(s[2:end-1], "\" \"")
end

if VERSION < v"0.4-dev"
    split_nokeep(a, b) = split(a, b, false)
else
    split_nokeep(a, b) = split(a, b, keep=false)
end

function stream2name(s::IO)
    name = s.name
    if !beginswith(name, "<file ")
        error("stream name ", name, " doesn't fit expected pattern")
    end
    name[7:end-1]
end

_unit_string_dict = ["um" => Micro*Meter, "mm" => Milli*Meter, "s" => Second]
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

end
