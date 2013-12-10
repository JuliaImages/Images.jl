#### Math with images ####

(+)(img::AbstractImageDirect, n::Number) = limadj(copy(img, data(img)+n), limplus(limits(img), n))
(+)(n::Number, img::AbstractImageDirect) = limadj(copy(img, data(img)+n), limplus(limits(img), n))
(+)(img::AbstractImageDirect, A::BitArray) = limadj(copy(img, data(img)+A), limplus(limits(img), Bool))
(+)(img::AbstractImageDirect, A::AbstractArray) = limadj(copy(img, data(img)+data(A)), limplus(limits(img), limits(A)))
(.+)(img::AbstractImageDirect, A::BitArray) = limadj(copy(img, data(img).+A), limplus(limits(img), Bool))
(.+)(img::AbstractImageDirect, A::AbstractArray) = limadj(copy(img, data(img).+data(A)), limplus(limits(img), limits(A)))
(-)(img::AbstractImageDirect, n::Number) = limadj(copy(img, data(img)-n), limminus(limits(img), n))
(-)(n::Number, img::AbstractImageDirect) = limadj(copy(img, n-data(img)), limminus(n, limits(img)))
(-)(img::AbstractImageDirect, A::BitArray) = limadj(copy(img, data(img)-A), limminus(limits(img), Bool))
(-)(img::AbstractImageDirect, A::AbstractArray) = limadj(copy(img, data(img)-data(A)), limminus(limits(img), limits(A)))
# (-)(A::AbstractArray, img::AbstractImageDirect) = limadj(copy(img, data(A) - data(img)), limminus(limits(A), limits(img)))
(.-)(img::AbstractImageDirect, A::BitArray) = limadj(copy(img, data(img).-A), limminus(limits(img), Bool))
(.-)(img::AbstractImageDirect, A::AbstractArray) = limadj(copy(img, data(img).-data(A)), limminus(limits(img), limits(A)))
(*)(img::AbstractImageDirect, n::Number) = limadj(copy(img, data(img)*n), limtimes(limits(img), n))
(*)(n::Number, img::AbstractImageDirect) = limadj(copy(img, data(img)*n), limtimes(limits(img), n))
(/)(img::AbstractImageDirect, n::Number) = limadj(copy(img, data(img)/n), limdivide(limits(img), n))
(.*)(img1::AbstractImageDirect, img2::AbstractImageDirect) = limadj(copy(img1, data(img1).*data(img2)), limtimes(limits(img1), limits(img2)))
(.*)(img::AbstractImageDirect, A::BitArray) = limadj(copy(img, data(img).*A), limtimes(limits(img), Bool))
(.*)(A::BitArray, img::AbstractImageDirect) = limadj(copy(img, data(img).*A), limtimes(limits(img), Bool))
(.*)(img::AbstractImageDirect, A::AbstractArray) = limadj(copy(img, data(img).*A), limtimes(limits(img), limits(A)))
(.*)(A::AbstractArray, img::AbstractImageDirect) = limadj(copy(img, data(img).*A), limtimes(limits(img), limits(A)))
(./)(img::AbstractImageDirect, A::BitArray) = limadj(copy(img, data(img)./A), limdivide(limits(img), Bool))  # needed to avoid ambiguity warning
(./)(img1::AbstractImageDirect, img2::AbstractImageDirect) = limadj(copy(img1, data(img1)./data(img2)), limdivide(limits(img1), limits(img2)))
(./)(img::AbstractImageDirect, A::AbstractArray) = limadj(copy(img, data(img)./A), limdivide(limits(img), limits(A)))
# (./)(A::AbstractArray, img::AbstractImageDirect) = limadj(copy(img, A./data(img))

function limadj(img::AbstractImageDirect, newlim)
    img["limits"] = newlim
    img
end

limplus(a::Tuple, n::Number) = a[1]+n, a[2]+n
limplus(a::Tuple, ::Type{Bool}) = a[1], a[2]+1
limplus(a::Tuple, b::Tuple) = a[1]+b[1], a[2]+b[2]
limminus(a::Tuple, n::Number) = a[1]-n, a[2]-n
limminus(n::Number, a::Tuple) = n-a[2], n-a[1]
limminus(a::Tuple, ::Type{Bool}) = a[1]-1, a[2]
limminus(a::Tuple, b::Tuple) = a[1]-b[2], a[2]-b[1]
limtimes(a::Tuple, n::Number) = n > 0 ? (a[1]*n, a[2]*n) : (a[2]*n, a[1]*n)
limtimes(a::Tuple, ::Type{Bool}) = min(a[1], false*a[1]), max(a[2], false*a[2])
limtimes(a::Tuple, b::Tuple) = min(a[1]*b[1],a[1]*b[2],a[2]*b[1],a[2]*b[2]), max(a[1]*b[1],a[1]*b[2],a[2]*b[1],a[2]*b[2])
limdivide(a::Tuple, n::Number) = n > 0 ? (a[1]/n, a[2]/n) : (a[2]/n, a[1]/n)
limdivide(a::Tuple, ::Type{Bool}) = min(a[1],a[1]/0), max(a[2],a[2]/0)
limdivide(a::Tuple, b::Tuple) = min(a[1]/b[1],a[1]/b[2],a[2]/b[1],a[2]/b[2]), max(a[1]/b[1],a[1]/b[2],a[2]/b[1],a[2]/b[2])

function sum(img::AbstractImageDirect, region)
    f = prod(size(img)[region...])
    l = limits(img)
    limadj(copy(img, sum(data(img), region)), (f*l[1], f*l[2]))
end

# Logical operations
(.<)(img::AbstractImageDirect, n::Number) = data(img) .< n
(.>)(img::AbstractImageDirect, n::Number) = data(img) .> n
(.<)(img::AbstractImageDirect, A::AbstractArray) = data(img) .< A
(.>)(img::AbstractImageDirect, A::AbstractArray) = data(img) .> A
(.==)(img::AbstractImageDirect, n::Number) = data(img) .== n
(.==)(img::AbstractImageDirect, A::AbstractArray) = data(img) .== A

#### Overlay AbstractArray implementation ####
length(o::Overlay) = isempty(o.channels) ? 0 : length(o.channels[1])
size(o::Overlay) = isempty(o.channels) ? (0,) : size(o.channels[1])
size(o::Overlay, d::Integer) = isempty(o.channels) ? 0 : size(o.channels[1],d)
eltype(o::Overlay) = RGB
nchannels(o::Overlay) = length(o.channels)

similar(o::Overlay) = Array(RGB, size(o))
similar(o::Overlay, ::NTuple{0}) = Array(RGB, size(o))
similar{T}(o::Overlay, ::Type{T}) = Array(T, size(o))
similar{T}(o::Overlay, ::Type{T}, sz::Int64) = Array(T, sz)
similar{T}(o::Overlay, ::Type{T}, sz::Int64...) = Array(T, sz)
similar{T}(o::Overlay, ::Type{T}, sz) = Array(T, sz)

function getindex(o::Overlay, indexes::Integer...)
    rgb = RGB(0,0,0)
    for i = 1:nchannels(o)
        if o.visible[i]
            rgb += scale(o.scalei[i], o.channels[i][indexes...])*o.colors[i]
        end
    end
    clip(rgb)
end

# Fix ambiguity warning
getindex(o::Overlay, i::Real) = getindex_overlay(o, i)
getindex(o::Overlay, indexes::Union(Real,AbstractVector)...) = getindex_overlay(o, indexes...)

function getindex_overlay(o::Overlay, indexes::Union(Real,AbstractVector)...)
    len = [length(i) for i in indexes]
    n = length(len)
    while len[n] == 1
        pop!(len)
        n -= 1
    end
    rgb = fill(RGB(0,0,0), len...)
    for i = 1:nchannels(o)
        if o.visible[i]
            tmp = scale(o.scalei[i], o.channels[i][indexes...])
            accumulate!(rgb, tmp, o.colors[i])
        end
    end
    clip!(rgb)
end

getindex(o::Overlay, indexes::(AbstractVector...)) = getindex(o, indexes...)

setindex!(o::Overlay, x, index::Real) = error("Overlays are read-only")
setindex!(o::Overlay, x, indexes...) = error("Overlays are read-only")

# Identical to getindex except it saves a call to each array's getindex
function convert(::Type{Array{RGB}}, o::Overlay)
    rgb = fill(RGB(0,0,0), size(o))
    for i = 1:length(o.channels)
        if o.visible[i]
            tmp = scale(o.scalei[i], o.channels[i])
            accumulate!(rgb, tmp, o.colors[i])
        end
    end
    clip!(rgb)
end

function accumulate!(rgb::Array{RGB}, A::Array{Float64}, color::RGB)
    for j = 1:length(rgb)
        rgb[j] += A[j]*color
    end
end

for N = 1:4
    @eval begin
        function accumulate!{T}(rgb::Array{RGB,$N}, A::AbstractArray{T,$N}, color::RGB, scalei::ScaleInfo)
            k = 0
            @inbounds @nloops $N i A begin
                rgb[k+=1] += scale(scalei, (@nref $N A i))*color
            end
        end
    end
end

(*)(f::FloatingPoint, c::RGB) = RGB(f*c.r, f*c.g, f*c.b)
(*)(f::Uint8, c::RGB) = (f/255)*c
(/)(c::RGB, f::Real) = (1.0/f)*c
(.*)(f::AbstractArray, c::RGB) = [x*c for x in f]
(+)(a::RGB, b::RGB) = RGB(a.r+b.r, a.g+b.g, a.b+b.b)
convert(::Type{Uint32}, c::ColorValue) = convert(RGB24, c).color

#### Converting images to uint32 color ####
# This is the crucial operation in image display

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
        function uint32color!{C<:ColorValue}(buf::Array{Uint32}, img::AbstractArray{C,$N})
            if size(buf) != size(img)
                error("Size mismatch")
            end
            dat = data(img)
            k = 0
            @inbounds @nloops $N i dat begin
                val = @nref $N dat i
                buf[k+=1] = convert(RGB24, val)
            end
            buf
        end
    end
end

# Grayscale arrays
for N = 1:4
    @eval begin
        function _uint32color_gray!{T}(buf::Array{Uint32}, A::AbstractArray{T,$N}, scalei::ScaleInfo = scaleinfo(Uint8, A))
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
        function _uint32color!{T}(buf::Array{Uint32}, A::AbstractArray{T,$N}, cs::String, cdim::Int, scalei::ScaleInfo = scaleinfo(Uint8, A))
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

# Signed grayscale arrays colorized magenta (positive) and green (negative
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
                        gr8 = iround(Uint8, 255.0*gr)
                        buf[k+=1] = rgb24(gr8, 0x00, gr8)
                    else
                        gr8 = iround(Uint8, -255.0*gr)
                        buf[k+=1] = rgb24(0x00, gr8, 0x00)
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
    ret::Uint32
    ret = convert(Uint32,r)<<16 | convert(Uint32,g)<<8 | convert(Uint32,b)
end

function argb32(a::Uint8, r::Uint8, g::Uint8, b::Uint8)
    ret::Uint32
    ret = convert(Uint32,a)<<24 | convert(Uint32,r)<<16 | convert(Uint32,g)<<8 | convert(Uint32,b)
end

function rgb24{T}(scalei::ScaleInfo{Uint8}, r::T, g::T, b::T)
    ret::Uint32
    ret = convert(Uint32,scale(scalei,r))<<16 | convert(Uint32,scale(scalei,g))<<8 | convert(Uint32,scale(scalei,b))
end

function argb32{T}(scalei::ScaleInfo{Uint8}, a::T, r::T, g::T, b::T)
    ret::Uint32
    ret = convert(Uint32,scale(scalei,a))<<24 | convert(Uint32,scale(scalei,r))<<16 | convert(Uint32,scale(scalei,g))<<8 | convert(Uint32,scale(scalei,b))
end

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

function imadjustintensity{T}(img::AbstractArray{T}, range)
    assert_scalar_color(img)
    if length(range) == 0
        range = [minimum(img) maximum(img)]
    elseif length(range) == 1
        error("incorrect range")
    end
    tmp = (img - range[1])/(range[2] - range[1])
    tmp[tmp .> 1] = 1
    tmp[tmp .< 0] = 0
    out = tmp
end

function red{T<:ColorValue}(img::AbstractArray{T})
    out = Array(Float64, size(img))
    for i = 1:length(img)
        out[i] = convert(RGB, img[i]).r
    end
    out
end

function red(img::AbstractArray)
    if colorspace(img) == "RGB" || colorspace(img) == "RGBA"
        ret = sliceim(img, "color", 1)
    else
        error("Not yet implemented")
    end
end

function green{T<:ColorValue}(img::AbstractArray{T})
    out = Array(Float64, size(img))
    for i = 1:length(img)
        out[i] = convert(RGB, img[i]).g
    end
    out
end

function green(img::AbstractArray)
    if colorspace(img) == "RGB" || colorspace(img) == "RGBA"
        ret = sliceim(img, "color", 2)
    else
        error("Not yet implemented")
    end
end

function blue{T<:ColorValue}(img::AbstractArray{T})
    out = Array(Float64, size(img))
    for i = 1:length(img)
        out[i] = convert(RGB, img[i]).b
    end
    out
end

function blue(img::AbstractArray)
    if colorspace(img) == "RGB" || colorspace(img) == "RGBA"
        ret = sliceim(img, "color", 3)
    else
        error("Not yet implemented")
    end
end

for N = 1:4
    N1 = N-1
    @eval begin
        function rgb2gray{T}(img::AbstractArray{T,$N})
            cs = colorspace(img)
            if cs == "Gray"
                return img
            end
            if cs != "RGB"
                error("Color space of image is $cs, not RGB")
            end
            cd = colordim(img)
            cstrd = stride(img, cd)
            sz = [size(img,d) for d=1:$N]
            sz[cd] = 1
            szs = sz[setdiff(1:$N,cd)]
            out = Array(T, szs...)::Array{T,$N1}
            wr, wg, wb = 0.30, 0.59, 0.11
            dat = data(img)
            indx = 0
            @nloops $N i d->1:sz[d] begin
                _, k = @nlinear $N dat i
                out[indx+=1] = truncround(T,wr*dat[k] + wg*dat[k+cstrd] + wb*dat[k+2cstrd])
            end
            p = properties(img)
            p["colorspace"] = "Gray"
            p["colordim"] = 0
            Image(out, p)
        end
    end
end

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
ssd{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sum((data(A)-data(B)).^2)

# normalized by Array size
ssdn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = ssd(A, B)/length(A)

# sum of absolute differences
sad{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sum(abs(data(A)-data(B)))

# normalized by Array size
sadn{T}(A::AbstractArray{T}, B::AbstractArray{T}) = sad(A, B)/length(A)

# normalized cross correlation
function ncc{T}(A::AbstractArray{T}, B::AbstractArray{T})
    Am = (data(A)-mean(data(A)))[:]
    Bm = (data(B)-mean(data(B)))[:]
    return dot(Am,Bm)/(norm(Am)*norm(Bm))
end

function padarray{T,n}(img::AbstractArray{T,n}, prepad::Vector{Int}, postpad::Vector{Int}, border::String, value::T)
    I = Array(Vector{Int}, n)
    for d = 1:n
        M = size(img, d)
        I[d] = [(1 - prepad[d]):(M + postpad[d])]
        if border == "value"
            I[d] = int((I[d] .>= 1) & (I[d] .<= M))
        elseif border == "replicate"
            I[d] = min(max(I[d], 1), M)
        elseif border == "circular"
            I[d] = 1 + mod(I[d] - 1, M)
        elseif border == "symmetric"
            I[d] = [1:M, M:-1:1][1 + mod(I[d] - 1, 2 * M)]
        elseif border == "reflect"
            I[d] = [1:M, M-1:-1:2][1 + mod(I[d] - 1, 2 * M - 2)]
        else
            error("unknown border condition")
        end
    end

    if border == "value"
        A = Array(T, map(length, I)...)
        fill!(A, value)
        A[map(x->map(bool, x), I)...] = img
    else
        A = img[I...]
    end

    return A
end

padarray{T,n}(img::AbstractArray{T,n}, padding) = padarray(img, padding, padding, "replicate", zero(T))
padarray{T,n}(img::AbstractArray{T,n}, padding::Vector{Int}, border::String) = padarray(img, padding, padding, border, zero(T))
padarray{T,n}(img::AbstractArray{T,n}, padding::Vector{Int}, value::T) = padarray(img, padding, padding, "value", value)

function padarray{T,n}(img::AbstractArray{T,n}, padding::Vector{Int}, border::String, direction::String)
    if direction == "both"
        return padarray(img, padding, padding, border, zero(T))
    elseif direction == "pre"
        return padarray(img, padding, 0 * padding, border, zero(T))
    elseif direction == "post"
        return padarray(img, 0 * padding, padding, border, zero(T))
    end
end

function padarray{T,n}(img::AbstractArray{T,n}, padding::Vector{Int}, value::T, direction::String)
    if direction == "both"
        return padarray(img, padding, padding, "value", value)
    elseif direction == "pre"
        return padarray(img, padding, 0 * padding, "value", value)
    elseif direction == "post"
        return padarray(img, 0 * padding, padding, "value", value)
    end
end

function _imfilter{T}(img::StridedMatrix{T}, filter::Matrix{T}, border::String, value)
    si, sf = size(img), size(filter)
    fw = iceil(([sf...] - 1) / 2)
    A = padarray(img, fw, fw, border, convert(T, value))
    # correlation instead of convolution
    filter = rot180(filter)
    # check if separable
    SVD = svdfact(filter)
    U, S, Vt = SVD[:U], SVD[:S], SVD[:Vt]
    separable = true
    for i = 2:length(S)
        separable &= (abs(S[i]) < sqrt(eps(T)))
    end
    if separable
        # conv2 isn't suitable for this (kernel center should be the actual center of the kernel)
        y = U[:,1]*sqrt(S[1])
        x = vec(Vt[1,:])*sqrt(S[1])
        sa = size(A)
        m = length(y)+sa[1]
        n = length(x)+sa[2]
        B = zeros(T, m, n)
        B[int(length(y)/2)+1:sa[1]+int(length(y)/2),int(length(x)/2)+1:sa[2]+int(length(x)/2)] = A
        yp = zeros(T, m)
        halfy = int((m-length(y)-1)/2)
        yp[halfy+1:halfy+length(y)] = y
        y = fft(yp)
        xp = zeros(T, n)
        halfx = int((n-length(x)-1)/2)
        xp[halfx+1:halfx+length(x)] = x
        x = fft(xp)
        C = fftshift(ifft(fft(B) .* (y * x.')))
        if T <: Real
            C = real(C)
        end
    else
        #C = conv2(A, filter)
        sa, sb = size(A), size(filter)
        At = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
        Bt = zeros(T, sa[1]+sb[1]-1, sa[2]+sb[2]-1)
        halfa1 = ifloor((size(At,1)-sa[1])/2)
        halfa2 = ifloor((size(At,2)-sa[2])/2)
        halfb1 = ifloor((size(Bt,1)-sb[1])/2)
        halfb2 = ifloor((size(Bt,2)-sb[2])/2)
        At[halfa1+1:halfa1+sa[1], halfa2+1:halfa2+sa[2]] = A
        Bt[halfb1+1:halfb1+sb[1], halfb2+1:halfb2+sb[2]] = filter
        C = fftshift(ifft(fft(At).*fft(Bt)))
        if T <: Real
            C = real(C)
        end
    end
    sc = size(C)
    out = C[int(sc[1]/2-si[1]/2):int(sc[1]/2+si[1]/2)-1, int(sc[2]/2-si[2]/2):int(sc[2]/2+si[2]/2)-1]
end

# imfilter for multi channel images
function imfilter{T}(img::AbstractArray{T}, filter::Matrix{T}, border::String, value)
    assert2d(img)
    cd = colordim(img)
    local A
    if cd == 0
        A = _imfilter(data(img), filter, border, value)
    else
        A = similar(data(img))
        coords = RangeIndex[1:size(img,i) for i = 1:ndims(img)]
        for i = 1:size(img, cd)
            coords[cd] = i
            simg = slice(img, coords...)
            tmp = _imfilter(simg, filter, border, value)
            A[coords...] = tmp[:]
        end
    end
    share(img, A)
end

imfilter(img, filter) = imfilter(img, filter, "replicate", 0)
imfilter(img, filter, border) = imfilter(img, filter, border, 0)

# imfilter{S,T<:FloatingPoint}(img::AbstractArray{S}, filter::Matrix{T}, args...) = imfilter(copy!(similar(img, T), img), filter, args...)

# IIR filtering with Gaussians
# See
#  Young, van Vliet, and van Ginkel, "Recursive Gabor Filtering",
#    IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 50: 2798-2805.
# and
#  Triggs and Sdika, "Boundary Conditions for Young - van Vliet
#    Recursive Filtering, IEEE TRANSACTIONS ON SIGNAL PROCESSING,
# Here we're using NA boundary conditions, so we set i- and i+
# (in Triggs & Sdika notation) to zero.
# Note these two papers use different sign conventions for the coefficients.

function imfilter_gaussian{T<:FloatingPoint}(img::AbstractArray{T}, sigma::Vector; emit_warning = true)
    A = copy(data(img))
    nanflag = isnan(A)
    hasnans = any(nanflag)
    if hasnans
        A[nanflag] = zero(T)
        validpixels = convert(Array{T}, !nanflag)
        imfilter_gaussian!(A, validpixels, sigma; emit_warning=emit_warning)
        A[nanflag] = nan(T)
    else
        imfilter_gaussian_no_nans!(A, sigma; emit_warning=emit_warning)
    end
    share(img, A)
end

function imfilter_gaussian{T<:Integer}(img::AbstractArray{T}, sigma::Vector; emit_warning = true)
    A = float64(data(img))
    imfilter_gaussian_no_nans!(A, sigma; emit_warning=emit_warning)
    share(img, truncround(T, A))
end

# This version is in-place, and destructive
# Any NaNs have to already be removed from data (and marked in validpixels)
function imfilter_gaussian!{T<:FloatingPoint}(data::Array{T}, validpixels::Array{T}, sigma::Vector; emit_warning = true)
    nd = ndims(data)
    if length(sigma) != nd
        error("Dimensionality mismatch")
    end
    _imfilter_gaussian!(data, sigma, emit_warning=emit_warning)
    _imfilter_gaussian!(validpixels, sigma, emit_warning=false)
    for i = 1:length(data)
        data[i] /= validpixels[i]
    end
    return data
end

# When there are no NaNs, the normalization is separable and hence can be computed far more efficiently
# This speeds the algorithm by approximately twofold
function imfilter_gaussian_no_nans!{T<:FloatingPoint}(data::Array{T}, sigma::Vector; emit_warning = true)
    nd = ndims(data)
    if length(sigma) != nd
        error("Dimensionality mismatch")
    end
    _imfilter_gaussian!(data, sigma, emit_warning=emit_warning)
    denom = Array(Vector{T}, nd)
    for i = 1:nd
        denom[i] = ones(T, size(data, i))
        if sigma[i] > 0
            _imfilter_gaussian!(denom[i], sigma[i:i], emit_warning=false)
        end
    end
    imfgnormalize!(data, denom)
    return data
end

for N = 1:5
    @eval begin
        function imfgnormalize!{T}(data::Array{T,$N}, denom)
            @nextract $N denom denom
            @nloops $N i data begin
                den = one(T)
                @nexprs $N d->(den *= denom_d[i_d])
                (@nref $N data i) /= den
            end
        end
    end
end

function iir_gaussian_coefficients(T::Type, sigma::Number; emit_warning = true)
    if sigma < 1 && emit_warning
        warn("sigma is too small for accuracy")
    end
    m0 = convert(T,1.16680)
    m1 = convert(T,1.10783)
    m2 = convert(T,1.40586)
    q = convert(T,1.31564*(sqrt(1+0.490811*sigma*sigma) - 1))
    scale = (m0+q)*(m1*m1 + m2*m2  + 2m1*q + q*q)
    B = m0*(m1*m1 + m2*m2)/scale
    B *= B
    # This is what Young et al call b, but in filt() notation would be called a
    a1 = q*(2*m0*m1 + m1*m1 + m2*m2 + (2*m0+4*m1)*q + 3*q*q)/scale
    a2 = -q*q*(m0 + 2m1 + 3q)/scale
    a3 = q*q*q/scale
    a = [-a1,-a2,-a3]
    Mdenom = (1+a1-a2+a3)*(1-a1-a2-a3)*(1+a2+(a1-a3)*a3)
    M = [-a3*a1+1-a3^2-a2      (a3+a1)*(a2+a3*a1)  a3*(a1+a3*a2);
          a1+a3*a2            -(a2-1)*(a2+a3*a1)  -(a3*a1+a3^2+a2-1)*a3;
          a3*a1+a2+a1^2-a2^2   a1*a2+a3*a2^2-a1*a3^2-a3^3-a3*a2+a3  a3*(a1+a3*a2)]/Mdenom;
    return a, B, M
end

function _imfilter_gaussian!{T<:FloatingPoint}(A::Array{T}, sigma::Vector; emit_warning = true)
    nd = ndims(A)
    szA = [size(A,i) for i = 1:nd]
    strdsA = [stride(A,i) for i = 1:nd]
    for d = 1:nd
        if sigma[d] == 0
            continue
        end
        if size(A, d) < 3
            error("All filtered dimensions must be of size 3 or larger")
        end
        a, B, M = iir_gaussian_coefficients(T, sigma[d], emit_warning=emit_warning)
        a1 = a[1]
        a2 = a[2]
        a3 = a[3]
        n1 = size(A,1)
        keepdims = [false,trues(nd-1)]
        if d == 1
            x = zeros(T, 3)
            vstart = zeros(T, 3)
            szhat = szA[keepdims]
            strdshat = strdsA[keepdims]
            if isempty(szhat)
                szhat = [1]
                strdshat = [1]
            end
            @forcartesian c szhat begin
                coloffset = offset(c, strdshat)
                A[2+coloffset] -= a1*A[1+coloffset]
                A[3+coloffset] -= a1*A[2+coloffset] + a2*A[1+coloffset]
                for i = 4:n1
                    A[i+coloffset] -= a1*A[i-1+coloffset] + a2*A[i-2+coloffset] + a3*A[i-3+coloffset]
                end
                copytail!(x, A, coloffset, 1, n1)
                A_mul_B(vstart, M, x)
                A[n1+coloffset] = vstart[1]
                A[n1-1+coloffset] -= a1*vstart[1]   + a2*vstart[2] + a3*vstart[3]
                A[n1-2+coloffset] -= a1*A[n1-1+coloffset] + a2*vstart[1] + a3*vstart[2]
                for i = n1-3:-1:1
                    A[i+coloffset] -= a1*A[i+1+coloffset] + a2*A[i+2+coloffset] + a3*A[i+3+coloffset]
                end
            end
        else
            x = Array(T, 3, n1)
            vstart = similar(x)
            keepdims[d] = false
            szhat = szA[keepdims]
            szd = szA[d]
            strdshat = strdsA[keepdims]
            strdd = strdsA[d]
            if isempty(szhat)
                szhat = [1]
                strdshat = [1]
            end
            @forcartesian c szhat begin
                coloffset = offset(c, strdshat)  # offset for the remaining dimensions
                for i = 1:n1 A[i+strdd+coloffset] -= a1*A[i+coloffset] end
                for i = 1:n1 A[i+2strdd+coloffset] -= a1*A[i+strdd+coloffset] + a2*A[i+coloffset] end
                for j = 3:szd-1
                    for i = 1:n1 A[i+j*strdd+coloffset] -= a1*A[i+(j-1)*strdd+coloffset] + a2*A[i+(j-2)*strdd+coloffset] + a3*A[i+(j-3)*strdd+coloffset] end
                end
                copytail!(x, A, coloffset, strdd, szd)
                A_mul_B(vstart, M, x)
                for i = 1:n1 A[i+(szd-1)*strdd+coloffset] = vstart[1,i] end
                for i = 1:n1 A[i+(szd-2)*strdd+coloffset] -= a1*vstart[1,i]   + a2*vstart[2,i] + a3*vstart[3,i] end
                for i = 1:n1 A[i+(szd-3)*strdd+coloffset] -= a1*A[i+(szd-2)*strdd+coloffset] + a2*vstart[1,i] + a3*vstart[2,i] end
                for j = szd-4:-1:0
                    for i = 1:n1 A[i+j*strdd+coloffset] -= a1*A[i+(j+1)*strdd+coloffset] + a2*A[i+(j+2)*strdd+coloffset] + a3*A[i+(j+3)*strdd+coloffset] end
                end
            end
        end
        for i = 1:length(A)
            A[i] *= B
        end
    end
    A
end

function offset(c::Vector{Int}, strds::Vector{Int})
    o = 0
    for i = 1:length(c)
        o += (c[i]-1)*strds[i]
    end
    o
end

function copytail!(dest, A, coloffset, strd, len)
    for j = 1:3
        for i = 1:size(dest, 2)
            tmp = A[i + coloffset + (len-j)*strd]
            dest[j,i] = tmp
        end
    end
    dest
end




function imlineardiffusion{T}(img::Array{T,2}, dt::FloatingPoint, iterations::Integer)
    u = img
    f = imlaplacian()
    for i = dt:dt:dt*iterations
        u = u + dt*imfilter(u, f, "replicate")
    end
    u
end

function imgaussiannoise{T}(img::AbstractArray{T}, variance::Number, mean::Number)
    return img + sqrt(variance)*randn(size(img)) + mean
end

imgaussiannoise{T}(img::AbstractArray{T}, variance::Number) = imgaussiannoise(img, variance, 0)
imgaussiannoise{T}(img::AbstractArray{T}) = imgaussiannoise(img, 0.01, 0)

function rgb2ntsc{T}(img::Array{T})
    trans = [0.299 0.587 0.114; 0.596 -0.274 -0.322; 0.211 -0.523 0.312]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * vec(img[i,j,:])
    end
    return out
end

function ntsc2rgb{T}(img::Array{T})
    trans = [1 0.956 0.621; 1 -0.272 -0.647; 1 -1.106 1.703]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * vec(img[i,j,:])
    end
    return out
end

function rgb2ycbcr{T}(img::Array{T})
    trans = [65.481 128.533 24.966; -37.797 -74.203 112; 112 -93.786 -18.214]
    offset = [16.0; 128.0; 128.0]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = offset + trans * vec(img[i,j,:])
    end
    return out
end

function ycbcr2rgb{T}(img::Array{T})
    trans = inv([65.481 128.533 24.966; -37.797 -74.203 112; 112 -93.786 -18.214])
    offset = [16.0; 128.0; 128.0]
    out = zeros(T, size(img))
    for i = 1:size(img,1), j = 1:size(img,2)
        out[i,j,:] = trans * (vec(img[i,j,:]) - offset)
    end
    return out
end

function imcomplement{T}(img::AbstractArray{T})
    l = limits(img)
    if l[2] != 1
        error("imcomplement not defined unless upper limit is 1")
    end
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

function imstretch{T}(img::AbstractArray{T}, m::Number, slope::Number)
    assert_scalar_color(img)
    if limits(img) != (0,1)
        warn("Image limits ", limits(img), " are not (0,1)")
    end
    share(img, 1./(1 + (m./(data(img) + eps(T))).^slope))
end

function imedge{T}(img::AbstractArray{T}, method::String, border::String)
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

imedge{T}(img::AbstractArray{T}, method::String) = imedge(img, method, "replicate")
imedge{T}(img::AbstractArray{T}) = imedge(img, "sobel", "replicate")

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
        div_p = backdiffx(p[:,:,1]) + backdiffy(p[:,:,2])
        u = img + div_p/lambda
        grad_u = cat(3, forwarddiffx(u), forwarddiffy(u))
        grad_u_mag = sqrt(grad_u[:,:,1].^2 + grad_u[:,:,2].^2)
        tmp = 1 + grad_u_mag*dt
        p = (dt*grad_u + p)./cat(3, tmp, tmp)
    end
    return u
end

# ROF Model for color images
function imROF(img::AbstractArray, lambda::Number, iterations::Integer)
    cd = colordim(img)
    local out
    if cd != 0
        out = similar(img)
        for i = size(img, cd)
            imsl = img["color", i]
            outsl = slice(out, "color", i)
            copy!(outsl, imROF(imsl, lambda, iterations))
        end
    else
        out = share(img, imROF(data(img), lambda, iterations))
    end
    out
end


### Morphological operations

# Erode and dilate support 3x3 regions only (and higher-dimensional generalizations).
dilate(img::AbstractArray, region=coords_spatial(img)) = dilate!(copy(img), region)
erode(img::AbstractArray, region=coords_spatial(img)) = erode!(copy(img), region)

dilate!(maxfilt, region=coords_spatial(maxfilt)) = extremefilt!(data(maxfilt), Base.Order.Forward, region)
erode!(minfilt, region=coords_spatial(minfilt)) = extremefilt!(data(minfilt), Base.Order.Reverse, region)
function extremefilt!(extrfilt::Array, order::Ordering, region=coords_spatial(extrfilt))
    for d = 1:ndims(extrfilt)
        if size(extrfilt, d) == 1 || !in(d, region)
            continue
        end
        sz = [size(extrfilt,i) for i = 1:ndims(extrfilt)]
        s = stride(extrfilt, d)
        sz[d] = 1
        @forcartesian i sz begin
            k = cartesian_linear(extrfilt, i)
            a2 = extrfilt[k]
            a3 = extrfilt[k+s]
            extrfilt[k] = extr(order, a2, a3)
            for l = 2:size(extrfilt,d)-1
                k += s
                a1 = a2
                a2 = a3
                a3 = extrfilt[k+s]
                extrfilt[k] = extr(order, a1, a2, a3)
            end
            extrfilt[k+s] = extr(order, a2, a3)
        end
    end
    extrfilt
end

opening(img::AbstractArray, region=coords_spatial(img)) = opening!(copy(img), region)
opening!(img::AbstractArray, region=coords_spatial(img)) = dilate!(erode!(img, region),region)
closing(img::AbstractArray, region=coords_spatial(img)) = closing!(copy(img), region)
closing!(img::AbstractArray, region=coords_spatial(img)) = erode!(dilate!(img, region),region)

extr(order::ForwardOrdering, x::Real, y::Real) = max(x,y)
extr(order::ForwardOrdering, x::Real, y::Real, z::Real) = max(x,y,z)
extr(order::ReverseOrdering, x::Real, y::Real) = min(x,y)
extr(order::ReverseOrdering, x::Real, y::Real, z::Real) = min(x,y,z)

extr(order::Ordering, x::RGB, y::RGB) = RGB(extr(order, x.r, y.r), extr(order, x.g, y.g), extr(order, x.b, y.b))
extr(order::Ordering, x::RGB, y::RGB, z::RGB) = RGB(extr(order, x.r, y.r, z.r), extr(order, x.g, y.g, z.g), extr(order, x.b, y.b, z.b))

extr(order::Ordering, x::ColorValue, y::ColorValue) = extr(order, convert(RGB, x), convert(RGB, y))
extr(order::Ordering, x::ColorValue, y::ColorValue, z::ColorValue) = extr(order, convert(RGB, x), convert(RGB, y), convert(RGB, z))

# On input, A should be 0 or 1. On output it will be the labeled array, where regions are labeled by the specified connectivity

# Connectivity for an arbitrary neighborhood
label_components(A::Union(BitArray, Array{Bool}), connectivity=1:ndims(A)) = label_components!(convert(Array{Int}, A), connectivity)

for N = 1:4
    @eval begin
        function label_components!(A::Array{Int,$N}, connectivity::Union(BitArray{$N}, Array{Bool,$N}))
            for i = 1:$N
                if size(connectivity, i) != 1 && size(connectivity, i) != 3
                    error("connectivity must be of size 1 or 3 in each dimension")
                end
            end
            @nextract $N halfwidth d->(halfwidth(size(connectivity,d)))
            @nextract $N offset d->(1+halfwidth(size(connectivity,d)))
            # Step 0: map each non-zero pixel to itself
            for i = 1:length(A)
                if A[i] > 0
                    A[i] = i
                end
            end
            # Step 1: associate each pixel with its connected-neighbor of lowest index
            @nloops $N i A begin
                minindex = @nref $N A i
                if minindex > 0               # not a background pixel
                    @nloops $N j d->(-halfwidth_d:halfwidth_d) begin   # j is the displacement to the neighbor
                        @nexprs $N d->(k_d=i_d+j_d)
                        if (@nrefshift $N connectivity j offset)   # if in connectivity pattern
                            if (@nall $N d->(1<=k_d<=size(A,d)))       # check bounds
                                nbrindex = @nref $N A k
                                if nbrindex > 0
                                    if nbrindex < minindex
                                        minindex = nbrindex
                                    else
                                        A[nbrindex] = minindex
                                    end
                                end
                            end
                        end
                    end
                    (@nref $N A i) = minindex
                end
            end
            # Step 2: iterate the graph until it stops changing
            flowmap!(A)
            # Step 3: assign unique labels
            labelmap!(A)
        end
    end
end

# A faster version for 4-connectivity (2d) and 6-connectivity (3d)
# region = [1,3] if you want connectivity along axes 1 and 3 but not 2
for N = 1:4
    @eval begin
        function label_components!(A::Array{Int,$N}, region::Union(Tuple, AbstractVector{Int}))
            usedim = falses($N)
            usedim[region] = true
            @nextract $N usedim usedim
            # Step 0: map each non-zero pixel to itself
            for i = 1:length(A)
                if A[i] > 0
                    A[i] = i
                end
            end
            # Step 1: associate each pixel with its connected-neighbor of lowest index
            @nloops $N i A begin
                minindex = @nref $N A i
                if minindex > 0             # not a background pixel
                    (@nexprs $N d->begin
                        if usedim_d            # if in pattern
                            if i_d>1           # if within bounds
#                                 _, nbrindex = (@nlinear $N A (d2->(d2==d) ? i_d-1 : i_d2))
                                nbrindex = @nref $N A (d2->(d2==d) ? i_d-1 : i_d2)
                                if nbrindex > 0
                                    if nbrindex < minindex
                                        minindex = nbrindex
                                    else
                                        A[nbrindex] = minindex
                                    end
                                end
                            end
                            if i_d<size(A,d)   # if within bounds
                                nbrindex = (@nref $N A (d2->(d2==d) ? i_d+1 : i_d2))
                                if nbrindex > 0
                                    if nbrindex < minindex
                                        minindex = nbrindex
                                    else
                                        A[nbrindex] = minindex
                                    end
                                end
                            end
                        end
                    end)
                    (@nref $N A i) = minindex
                end
            end
            # Step 2: iterate the graph until it stops changing
            flowmap!(A)
            # Step 3: assign unique labels
            labelmap!(A)
        end
    end
end

# Converges in at most logN steps
function flowmap!(map::Array{Int})
    changed = true
    while changed
        changed = false
        for i = 1:length(map)
            peak = map[i]
            if peak > 0
                next = map[peak]
                changed |= peak != next
                map[i] = next
            end
        end
    end
    map
end

function labelmap!(map::Array{Int})
    lbl = zeros(Int, size(map))
    j = 1
    for i = 1:length(lbl)
        if map[i] == i
            lbl[i] = j
            j += 1
        end
    end
    for i = 1:length(lbl)
        m = map[i]
        if m > 0
            lbl[i] = lbl[m]
        end
    end
    lbl
end

halfwidth(i::Integer) = div((i-1),2)

# phantom images

function shepp_logan(M,N; highContrast=true)
  # Initially proposed in Shepp, Larry; B. F. Logan (1974). 
  # "The Fourier Reconstruction of a Head Section". IEEE Transactions on Nuclear Science. NS-21.
  
  P = zeros(M,N)
 
  x = linspace(-1,1,M)'
  y = linspace(1,-1,N)
 
  centerX = [0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06]
  centerY = [0, -0.0184, 0, 0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605]
  majorAxis = [0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 0.046, 0.046, 0.023, 0.023]
  minorAxis = [0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 0.046, 0.023, 0.023, 0.046]
  theta = [0, 0, -18.0, 18.0, 0, 0, 0, 0, 0, 0]
  
  # original (CT) version of the phantom
  grayLevel = [2, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
  
  if(highContrast)
    # high contrast (MRI) version of the phantom
    grayLevel = [1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  end

  for l=1:length(theta)
    P += grayLevel[l] * (
           ( (cos(theta[l] / 360*2*pi) * (x - centerX[l]) .+
              sin(theta[l] / 360*2*pi) * (y - centerY[l])) / majorAxis[l] ).^2 .+
           ( (sin(theta[l] / 360*2*pi) * (x - centerX[l]) .-
              cos(theta[l] / 360*2*pi) * (y - centerY[l])) / minorAxis[l] ).^2 .< 1 )
  end

  return P
end
 
shepp_logan(N;highContrast=true) = shepp_logan(N,N;highContrast=highContrast)
