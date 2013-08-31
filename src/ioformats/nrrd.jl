module NRRD

using Images
import Images: imread, imwrite

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
    "float" => Float32,
    "double" => Float64]  # yet more can be added...

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
    header = Dict{ASCIIString, ASCIIString}()
    comments = Array(ASCIIString, 0)
    line = strip(readline(stream))
    while !isempty(line)
        if line[1] != '#'
            key, value = split(line, ":")
            header[key] = strip(value)
        else
            cmt = strip(lstrip(line, collect("#")))
            if !isempty(cmt)
                push!(comments, cmt)
            end
        end
        line = strip(readline(stream))
    end
    sdata = stream
    if haskey(header, "data file")
        sdata = open(header["data file"])
    elseif haskey(header, "datafile")
        sdata = open(header["datafile"])
    end
    # Parse properties and read the data
    sz = parse_vector_int(header["sizes"])
    T = typedict[header["type"]]
    props = Dict{ASCIIString, Any}()
    local A
    if header["encoding"] == "raw"
        # Use memory-mapping for large files
        if prod(sz) > 10^8
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
            if haskey(header, "endian")
                if header["endian"] != myendian()
                    props["bswap"] = true
                end
            end
        else
            A = read(sdata, T, sz...)
            if haskey(header, "endian")
                if header["endian"] != myendian()
                    A = bswap(A)
                end
            end
        end
    end
    if haskey(header, "kinds")
        kinds = split(header["kinds"], " ")
        for i = 1:length(kinds)
            k = kinds[i]
            if k == "time"
                props["timedim"] = i
            elseif contains(("list","3-color","4-color"), k)
                props["colordim"] = i
            elseif k == "RGB-color"
                props["colordim"] = i
                props["colorspace"] = "RGB"
            elseif k == "HSV-color"
                props["colordim"] = i
                props["colorspace"] = "HSV"
            elseif k == "RGBA-color"
                props["colordim"] = i
                props["colorspace"] = "RGBA"
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
    if haskey(header, "spacings")
        ps = parse_vector_float(header["spacings"])
        keep = trues(length(ps))
        cd = get(props, "colordim", 0)
        if 1 <= cd <= length(keep)
            keep[cd] = false
        end
        td = get(props, "timedim", 0)
        if 1 <= td <= length(keep)
            keep[td] = false
        end
        props["pixelspacing"] = ps[keep]
    end
    if !isempty(comments)
        props["comments"] = comments
    end
    img = Image(A, props)
    spatialorder = ["x", "y", "z"]
    img.properties["spatialorder"] = spatialorder[1:sdims(img)]
    img
end

function parse_vector_int(s::String)
    ss = split(s, r"[ ,;]", false)
    v = Array(Int, length(ss))
    for i = 1:length(ss)
        v[i] = int(ss[i])
    end
    return v
end

function parse_vector_float(s::String)
    ss = split(s, r"[ ,;]", false)
    v = Array(Float64, length(ss))
    for i = 1:length(ss)
        v[i] = float(ss[i])
    end
    return v
end

function stream2name(s::IO)
    name = s.name
    if !beginswith(name, "<file ")
        error("stream name ", name, " doesn't fit expected pattern")
    end
    name[7:end-1]
end

end
