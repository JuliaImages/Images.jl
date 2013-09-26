module NRRD

using Images, Units
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
    header = Dict{ASCIIString, UTF8String}()
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
            elseif in(k, ("list","3-color","4-color"))
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
        if haskey(header, "space units")
            ustrs = split(header["space units"], " ")
            if length(ustrs) != length(ps)
                warn("space units parsing did not agree with spacings")
                @show u
                @show ps
                props["pixelspacing"] = ps[keep]
            else
                u = Array(Quantity, 0)
                for i = 1:length(ustrs)
                    if keep[i]
                        try
                            utmp = ps[i]*parse_quantity("1 "*ustrs[i][2:end-1])
                            push!(u, utmp)
                        catch
                            push!(u, Quantity(SINone, Unknown, ps[i]))
                        end
                    end
                end
                props["pixelspacing"] = u
            end
        else
            props["pixelspacing"] = ps[keep]
        end
    end
    if !isempty(comments)
        props["comments"] = comments
    end
    img = Image(A, props)
    spatialorder = ["x", "y", "z"]
    img.properties["spatialorder"] = spatialorder[1:sdims(img)]
    img
end

function imwrite(img, sheader::IO, ::Type{Images.NRRDFile}, props::Dict = Dict{ASCIIString,Any}())
    println(sheader, "NRRD0001")
    # Write the datatype
    T = get(props, "type", eltype(data(img)))
    if T<:FloatingPoint
        println(sheader, "type: ", (T == Float32) ? "float" : "double")
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
    print(sheader, "spacings:")
    printunits = false
    for x in ps
        if isa(x, Quantity)
            printunits = true
            print(sheader, " ", x.value)
        else
            print(sheader, " ", x)
        end
    end
    print(sheader,"\n")
    if printunits
        print(sheader, "space units:")
        for x in ps
            print(sheader," \"")
            pshow(sheader,prefix(x))
            pshow(sheader,base(x))
            print(sheader,"\"")
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
