#### Converting images to uint32 color ####
# This is the crucial operation in image display

# First, the no-op conversions
if is_little_endian
    uint32color{CV<:Union(BGRA{Ufixed8},ARGB32)}(img::AbstractArray{CV}) = reinterpret(Uint32, img, size(img))
else
    uint32color{CV<:Union(ARGB{Ufixed8},ARGB32)}(img::AbstractArray{CV}) = reinterpret(Uint32, img, size(img))
end

function uint32color(img, args...)
    sz = size(img)
    ssz = sz[coords_spatial(img)]
    buf = Array(Uint32, ssz)
    uint32color!(buf, img, args...)
    buf
end

function uint32color!(buf::Array{Uint32}, img, args...)
    cdim = colordim(img)
    if cdim != 0
        sz = size(img)
        if size(buf) != tuple(sz[1:cdim-1]..., sz[cdim+1:end]...)
            error("Size mismatch")
        end
        _uint32color!(buf, img, colorspace(img), cdim, args...)
    else
        if size(buf) != size(img)
            error("Size mismatch")
        end
        if colorspace(img) == "RGB24"
            _uint32color_rgb24!(buf, img, args...)
        else
            _uint32color_gray!(buf, img, args...)
        end
    end
    buf
end

# Indexed images (colormaps)
function uint32color!(buf::Array{Uint32}, img::AbstractImageIndexed)
    dat = data(img)
    cmap = img.cmap
    for i = 1:length(buf)
        buf[i] = convert(Uint32, cmap[dat[i]])
    end
    buf
end

# ColorValue arrays
for N = 1:4
    @eval begin
        function uint32color!{C<:ColorType}(buf::Array{Uint32}, img::AbstractArray{C,$N}, scalei = scaleinfo(Uint32, img))
            if size(buf) != size(img)
                error("Size mismatch")
            end
            dat = data(img)
            k = 0
            T32 = cvtypes(C)
            @inbounds @nloops $N i dat begin
                val = @nref $N dat i
                buf[k+=1] = convert(T32, scale(scalei, val))#clamp(convert(TP, val)))
            end
            buf
        end
    end
end
cvtypes{C<:ColorValue}(::Type{C}) = RGB24
cvtypes{C<:AbstractAlphaColorValue}(::Type{C}) = ARGB32

# Grayscale arrays
for N = 1:4
    @eval begin
        function _uint32color_gray!{T}(buf::Array{Uint32}, A::AbstractArray{T,$N}, scalei::ScaleInfo = scaleinfo(Ufixed8, A))
            scalei_t = take(scalei, A)
            Adat = data(A)
            k = 0
            @inbounds @nloops $N i A begin
                val = @nref $N Adat i
                gr = scale(scalei_t, val)
                buf[k+=1] = rgb24(gr, gr, gr)
            end
            buf
        end
    end
end

# RGB24 arrays (just a copy operation)
for N = 1:4
    @eval begin
        function _uint32color_rgb24!{T}(buf::Array{Uint32}, A::AbstractArray{T,$N})
            k = 0
            Adat = data(A)
            @inbounds @nloops $N i Adat begin
                buf[k+=1] = @nref $N Adat i
            end
            buf
        end
    end
end

# Arrays where one dimension encodes color or transparency
for N = 1:5
    @eval begin
        function _uint32color!{T}(buf::Array{Uint32}, A::AbstractArray{T,$N}, cs::String, cdim::Int, scalei::ScaleInfo = scaleinfo(Ufixed8, A))
            scalei_t = take(scalei, A)
            k = 0
            Adat = data(A)
            # Because cdim is not known at compile-time, we have to prepare size and step variables
            cstep = zeros(Int, $N)
            cstep[cdim] = 1
            szA = [size(Adat, d) for d = 1:$N]
            szA[cdim] = 1  # don't loop over color dimension
            @nextract $N szA szA
            @nextract $N cstep cstep
            if cs == "GrayAlpha"
                @inbounds @nloops $N i d->(1:szA_d) begin
                    gr = @nref $N Adat i
                    alpha = @nref $N Adat d->(i_d+cstep_d)
                    buf[k+=1] = argb32(scalei_t, alpha, gr, gr, gr)
                end
            elseif cs == "RGB"
                @inbounds @nloops $N i d->(1:szA_d) begin
                    r = @nref $N Adat i
                    g = @nref $N Adat d->(i_d+cstep_d)
                    b = @nref $N Adat d->(i_d+2cstep_d)
                    buf[k+=1] = rgb24(scalei_t, r, g, b)
                end
            elseif cs == "ARGB"
                @inbounds @nloops $N i d->(1:szA_d) begin
                    a = @nref $N Adat i
                    r = @nref $N Adat d->(i_d+cstep_d)
                    g = @nref $N Adat d->(i_d+2cstep_d)
                    b = @nref $N Adat d->(i_d+3cstep_d)
                    buf[k+=1] = argb32(scalei_t, a, r, g, b)
                end
            elseif cs == "RGBA"
                @inbounds @nloops $N i d->(1:szA_d) begin
                    r = @nref $N Adat i
                    g = @nref $N Adat d->(i_d+cstep_d)
                    b = @nref $N Adat d->(i_d+2cstep_d)
                    a = @nref $N Adat d->(i_d+3cstep_d)
                    buf[k+=1] = argb32(scalei_t, a, r, g, b)
                end
            else
                error("colorspace ", cs, " not yet supported")
            end
        end
    end
end

# Overlays
_wrap(A::Array, B::AbstractArray) = B
_wrap(S::SubArray, B::AbstractArray) = sub(B, S.indexes...)
function uint32color!{O<:Overlay,N,IT<:(RangeIndex...,)}(buf::Array{Uint32}, img::Union(Overlay, Image{RGB,N,O}, SubArray{RGB,N,O,IT}, Image{RGB,N,SubArray{RGB,N,O,IT}}))
    if size(buf) != size(img)
        error("Size mismatch")
    end
    rgb = fill(RGB(0,0,0), size(buf))
    dat = data(img)
    p = parent(dat)
    for i = 1:nchannels(p)
        if p.visible[i]
            accumulate!(rgb, _wrap(dat, data(p.channels[i])), p.colors[i], p.scalei[i])
        end
    end
    clip!(rgb)
    for i = 1:length(buf)
        buf[i] = convert(RGB24, rgb[i])
    end
    buf
end

# Signed grayscale arrays colorized magenta (positive) and green (negative)
for N = 1:4
    @eval begin
        function _uint32color_gray!{T}(buf::Array{Uint32}, A::AbstractArray{T,$N}, scalei::ScaleSigned)
            k = 0
            Adat = data(A)
            @inbounds @nloops $N i Adat begin
                val = @nref $N Adat i
                gr = scale(scalei, val)
                if isfinite(gr)
                    if gr >= 0
                        gr8 = convert(Ufixed8, gr)
                        buf[k+=1] = rgb24(gr8, zero(Ufixed8), gr8)
                    else
                        gr8 = convert(Ufixed8, -gr)
                        buf[k+=1] = rgb24(zero(Ufixed8), gr8, zero(Ufixed8))
                    end
                else
                    buf[k+=1] = zero(Uint32)
                end
            end
            buf
        end
    end
end

function rgb24(r::Uint8, g::Uint8, b::Uint8)
    ret = convert(Uint32,r)<<16 | convert(Uint32,g)<<8 | convert(Uint32,b)
end
rgb24(r::Ufixed8, g::Ufixed8, b::Ufixed8) = rgb24(reinterpret(r), reinterpret(g), reinterpret(b))


argb32(a::Ufixed8, r::Ufixed8, g::Ufixed8, b::Ufixed8) = argb32(reinterpret(a), reinterpret(r), reinterpret(g), reinterpret(b))
function argb32(a::Uint8, r::Uint8, g::Uint8, b::Uint8)
    ret = convert(Uint32,a)<<24 | convert(Uint32,r)<<16 | convert(Uint32,g)<<8 | convert(Uint32,b)
end

rgb24(scalei::ScaleInfo, r, g, b) = rgb24(scale(scalei,r), scale(scalei,g), scale(scalei,b))

argb32(scalei::ScaleInfo, a, r, g, b) = argb24(scale(scalei,a), scale(scalei,r), scale(scalei,g), scale(scalei,b))

#### Color palettes ####

function lut(pal::Vector, a)
    out = similar(a, eltype(pal))
    n = length(pal)
    for i=1:length(a)
        out[i] = pal[clamp(a[i], 1, n)]
    end
    out
end

function indexedcolor(data, pal)
    mn = minimum(data); mx = maximum(data)
    indexedcolor(data, pal, mx-mn, (mx+mn)/2)
end

function indexedcolor(data, pal, w, l)
    n = length(pal)-1
    if n == 0
        return fill(pal[1], size(data))
    end
    w_min = l - w/2
    scale = w==0 ? 1 : w/n
    lut(pal, iround((data - w_min)./scale) + 1)
end

const palette_gray32  = [0xff000000,0xff080808,0xff101010,0xff181818,0xff202020,0xff292929,0xff313131,0xff393939,
                         0xff414141,0xff4a4a4a,0xff525252,0xff5a5a5a,0xff626262,0xff6a6a6a,0xff737373,0xff7b7b7b,
                         0xff838383,0xff8b8b8b,0xff949494,0xff9c9c9c,0xffa4a4a4,0xffacacac,0xffb4b4b4,0xffbdbdbd,
                         0xffc5c5c5,0xffcdcdcd,0xffd5d5d5,0xffdedede,0xffe6e6e6,0xffeeeeee,0xfff6f6f6,0xffffffff]

const palette_gray64  = [0xff000000,0xff040404,0xff080808,0xff0c0c0c,0xff101010,0xff141414,0xff181818,0xff1c1c1c,
                         0xff202020,0xff242424,0xff282828,0xff2c2c2c,0xff303030,0xff343434,0xff383838,0xff3c3c3c,
                         0xff404040,0xff444444,0xff484848,0xff4c4c4c,0xff505050,0xff555555,0xff595959,0xff5d5d5d,
                         0xff616161,0xff656565,0xff696969,0xff6d6d6d,0xff717171,0xff757575,0xff797979,0xff7d7d7d,
                         0xff818181,0xff858585,0xff898989,0xff8d8d8d,0xff919191,0xff959595,0xff999999,0xff9d9d9d,
                         0xffa1a1a1,0xffa5a5a5,0xffaaaaaa,0xffaeaeae,0xffb2b2b2,0xffb6b6b6,0xffbababa,0xffbebebe,
                         0xffc2c2c2,0xffc6c6c6,0xffcacaca,0xffcecece,0xffd2d2d2,0xffd6d6d6,0xffdadada,0xffdedede,
                         0xffe2e2e2,0xffe6e6e6,0xffeaeaea,0xffeeeeee,0xfff2f2f2,0xfff6f6f6,0xfffafafa,0xffffffff]

const palette_fire    = [0xff5a5a5a,0xff636058,0xff6c6757,0xff756e56,0xff7e7455,0xff877b54,0xff908253,0xff998851,
                         0xffa28f50,0xffab964f,0xffb49c4e,0xffbda34d,0xffc6aa4c,0xffcfb04a,0xffd8b749,0xffe1be48,
                         0xffeac447,0xfff3cb46,0xfffdd245,0xfffccc42,0xfffcc640,0xfffcc03d,0xfffbba3b,0xfffbb438,
                         0xfffbae36,0xfffaa833,0xfffaa231,0xfffa9c2e,0xfffa962c,0xfff99029,0xfff98a27,0xfff98424,
                         0xfff87e22,0xfff8781f,0xfff8721d,0xfff76c1a,0xfff76618,0xfff76015,0xfff75a13,0xfff65410,
                         0xfff64e0e,0xfff6480b,0xfff54209,0xfff53c06,0xfff53604,0xfff53102,0xffe72e01,0xffd92b01,
                         0xffcc2801,0xffbe2601,0xffb02301,0xffa32001,0xff951d01,0xff881b01,0xff7a1801,0xff6c1500,
                         0xff5f1300,0xff511000,0xff440d00,0xff360a00,0xff280800,0xff1b0500,0xff0d0200,0xff000000]

const palette_rainbow = [0xff0e46e9,0xff0d58ea,0xff0c6bec,0xff0c7eee,0xff0b91f0,0xff0ba4f1,0xff0ab7f3,0xff0acaf5,
                         0xff09ddf7,0xff09f0f9,0xff06efbd,0xff04ef81,0xff02ee45,0xff00ee0a,0xff1cee08,0xff38ee07,
                         0xff54ee06,0xff70ee05,0xff8dee04,0xffa9ee03,0xffc5ee02,0xffe1ee01,0xfffeee00,0xfffcd401,
                         0xfffbba02,0xfffaa104,0xfff98705,0xfff86e07,0xfff75408,0xfff63b0a,0xfff5210b,0xfff4080d]

redval(p)   = (p>>>16)&0xff
greenval(p) = (p>>>8)&0xff
blueval(p)  = p&0xff
alphaval(p)   = (p>>>24)&0xff
