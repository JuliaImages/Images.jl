#### Scaling/clipping/type conversion ####

minclamp(minval, val) = val < minval ? minval : convert(typeof(minval), val)
maxclamp(maxval, val) = val > maxval ? maxval : convert(typeof(maxval), val)

clamp{T}(::Type{T}, val::T) = T
clamp{T}(::Type{T}, val) = val > typemax(T) ? typemax(T) : (val < typemin(T) ? typemin(T) : convert(T, val))
clamp{S<:Integer, T<:FloatingPoint}(::Type{T}, val::S) = convert(T, S)

# "Safe" rounding to an integer. Clips the value to the target range before converting the type.
truncround{T<:Integer}(::Type{T}, val::T) = val
truncround{T<:Integer}(::Type{T}, val::Integer) = val > typemax(T) ? typemax(T) : (val < typemin(T) ? typemin(T) : val)
truncround{T<:Integer,F<:FloatingPoint}(::Type{T}, val::F) = val > convert(F,typemax(T)) ? typemax(T) : (val < convert(F,typemin(T)) ? typemin(T) : iround(T, val))
function truncround{T<:Integer}(::Type{T}, A::AbstractArray)
    X = similar(A, T)
    for i = 1:length(A)
        X[i] = truncround(T, A[i])
    end
    X
end

## scale() for array types

function scale{T}(scalei::ScaleInfo{T}, img::AbstractArray)
    out = similar(img, T)
    scale!(out, scalei, img)
end

function scale!{T}(out, scalei::ScaleInfo{T}, img::AbstractArray)
    dimg = data(img)
    dout = data(out)
    for i = 1:length(dimg)
        dout[i] = scale(scalei, dimg[i])  # FIXME? SubArray performance
    end
    out
end

function scale!{T}(out::AbstractImage, scalei::ScaleInfo{T}, img::AbstractArray)
    dimg = data(img)
    dout = data(out)
    for i = 1:length(dimg)
        dout[i] = scale(scalei, dimg[i])  # FIXME? SubArray performance
    end
    l = limits(img)
    out["limits"] = (scale(scalei, l[1]), scale(scalei, l[2]))
    out
end

## ScaleNone

type ScaleNone{T} <: ScaleInfo{T}; end

scale{T<:Real}(scalei::ScaleNone{T}, val::T) = val
scale{T}(scalei::ScaleNone{T}, img::AbstractArray{T}) = img
scale{T,S<:Real}(scalei::ScaleNone{T}, val::S) = convert(T, val)
scale{T<:Integer,S<:FloatingPoint}(scalei::ScaleNone{T}, val::S) = iround(T, val)

scale(scalei::ScaleNone{Uint32}, val::RGB) = convert(Uint32, convert(RGB24, val))
scale(scalei::ScaleNone{RGB8}, val::RGB) = RGB8(iround(Uint8, 255*val.r), iround(Uint8, 255*val.g), iround(Uint8, 255*val.b))

## BitShift

type BitShift{T<:Integer,N} <: ScaleInfo{T} end

scale{T,N}(scalei::BitShift{T,N}, val::Integer) = convert(T, val>>>N)

## Clip types

# The Clip types just enforce bounds, but do not scale or
# subtract the minimum
abstract Clip{T} <: ScaleInfo{T}
type ClipMin{T,From} <: Clip{T}
    min::From
end
ClipMin{T,From}(::Type{T}, min::From) = ClipMin{T,From}(min)
ClipMin(::Type{RGB}) = ClipMinMax(Float64,0.0)
type ClipMax{T,From} <: Clip{T}
    max::From
end
ClipMax{T,From}(::Type{T}, max::From) = ClipMax{T,From}(max)
ClipMax(::Type{RGB}) = ClipMinMax(Float64,1.0)
type ClipMinMax{T,From} <: Clip{T}
    min::From
    max::From
end
ClipMinMax{T,From}(::Type{T}, min::From, max::From) = ClipMinMax{T,From}(min,max)
ClipMinMax(::Type{RGB}) = ClipMinMax(Float64,0.0,1.0)
ClipMinMax(::Type{RGB8}, ::Type{RGB}) = ClipMinMax{RGB8,RGB}(RGB(0,0,0),RGB(1,1,1))

scale{T<:Integer,F<:FloatingPoint}(scalei::ClipMin{T,F}, val::F) = iround(T, max(val, scalei.min))
scale{T<:Real,F<:Real}(scalei::ClipMin{T,F}, val::F) = convert(T, max(val, scalei.min))
scale{T<:Integer,F<:FloatingPoint}(scalei::ClipMax{T,F}, val::F) = iround(T, min(val, scalei.max))
scale{T<:Real,F<:Real}(scalei::ClipMax{T,F}, val::F) = convert(T, min(val, scalei.max))
scale{T<:Real}(scalei::ClipMinMax{T,T}, val::T) = min(max(val, scalei.min), scalei.max)
scale{T<:Real,F<:Real}(scalei::ClipMinMax{T,F}, val::F) = convert(T,min(max(val, scalei.min), scalei.max))
scale{T<:Integer,F<:FloatingPoint}(scalei::ClipMinMax{T,F}, val::F) = iround(T,min(max(val, scalei.min), scalei.max))

scale(scalei::Clip, v::RGB) = RGB(scale(scalei, v.r), scale(scalei, v.g), scale(scalei, v.b))
scale(scalei::ClipMinMax{RGB8,RGB}, val::RGB) = RGB8(truncround(Uint8, 255*val.r), truncround(Uint8, 255*val.g), truncround(Uint8, 255*val.b))

clip(v::RGB) = RGB(min(1.0,v.r),min(1.0,v.g),min(1.0,v.b))
function clip!(A::Array{RGB})
    for i = 1:length(A)
        A[i] = clip(A[i])
    end
    A
end

## ScaleMinMax

# This scales and subtracts the min value
type ScaleMinMax{To,From} <: ScaleInfo{To}
    min::From
    max::From
    s::Float64
end
ScaleMinMax{To,From}(::Type{To}, min::From, max::From, s::Real) = ScaleMinMax{To,From}(min, max, float64(s))

function scale{To<:Integer,From<:Real}(scalei::ScaleMinMax{To,From}, val::From)
    # Clip to range min:max and subtract min
    t::From = (val > scalei.min) ? ((val < scalei.max) ? val-scalei.min : scalei.max-scalei.min) : zero(From)
    convert(To, iround(scalei.s*t))
end

function scale{To<:Real,From<:Real}(scalei::ScaleMinMax{To,From}, val::From)
    t::From = (val > scalei.min) ? ((val < scalei.max) ? val-scalei.min : scalei.max-scalei.min) : zero(From)
    convert(To, scalei.s*t)
end

# ScaleMinMax constructors that take AbstractArray input
scaleminmax{To<:Integer,From}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax{To,From}(convert(From,mn), convert(From,mx), float64(typemax(To)/(mx-mn)))
scaleminmax{To<:FloatingPoint,From}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax{To,From}(convert(From,mn), convert(From,mx), 1.0/(mx-mn))
scaleminmax{To}(::Type{To}, img::AbstractArray) = scaleminmax(To, img, minfinite(img), maxfinite(img))
scaleminmax(img::AbstractArray) = scaleminmax(Uint8, img)
scaleminmax(img::AbstractArray, mn::Real, mx::Real) = scaleminmax(Uint8, img, mn, mx)
scaleminmax{To<:Real,From<:Real}(::Type{To}, mn::From, mx::From) = ScaleMinMax{To,From}(mn, mx, 255.0/(mx-mn))
function scaleminmax{To}(::Type{To}, img::AbstractImage, tindex::Integer)
    1 <= tindex <= nimages(img) || error("The image does not have a time slice of ", tindex)
    imsl = sliceim(img, "t", 1)
    scaleminmax(To, imsl)
end
scaleminmax(img::AbstractImage, tindex::Integer) = scaleminmax(Uint8, img, tindex)

## ScaleSigned

# Multiplies by a scaling factor and then clips to the range [-1,1].
# Intended for positive/negative coloring
type ScaleSigned <: ScaleInfo{Float64}
    s::Float64
end

scalesigned(img::AbstractArray) = ScaleSigned(1.0/maxabsfinite(img))
function scalesigned(img::AbstractImage, tindex::Integer)
    1 <= tindex <= nimages(img) || error("The image does not have a time slice of ", tindex)
    imsl = sliceim(img, "t", 1)
    ScaleSigned(1.0/maxabsfinite(imsl))
end

function scale(scalei::ScaleSigned, val::Real)
    sval::Float64 = scalei.s*val
    return sval>1.0 ? 1.0 : (sval<-1.0 ? -1.0 : sval)
end


## ScaleAutoMinMax

# Works only on whole arrays, not values
type ScaleAutoMinMax{T} <: ScaleInfo{T} end
ScaleAutoMinMax() = ScaleAutoMinMax{Uint8}()

scale!{T}(out::AbstractImage, scalei::ScaleAutoMinMax{T}, img::Union(StridedArray,AbstractImageDirect)) = scale!(out, take(scalei, img), img)
scale!{T}(out, scalei::ScaleAutoMinMax{T}, img::Union(StridedArray,AbstractImageDirect)) = scale!(out, take(scalei, img), img)

take(scalei::ScaleInfo, img::AbstractArray) = scalei
take{T}(scalei::ScaleAutoMinMax{T}, img::AbstractArray) = scaleminmax(T, img)
take{To,From}(scalei::ScaleMinMax{To}, img::AbstractArray{From}) = ScaleMinMax(To, convert(From, scalei.min), convert(From, scalei.max), scalei.s)


## ScaleInfo defaults

scaleinfo{T}(::Type{Any}, img::AbstractArray{T}) = ScaleNone{T}()

function scaleinfo{T}(img::AbstractArray{T})
    cs = colorspace(img)
    if cs == "RGB24" || cs == "ARGB32"
        return ScaleNone{T}()
    end
    scaleinfo(Uint8, img)
end
scaleinfo{T<:Unsigned}(::Type{T}, img::AbstractArray{T}) = ScaleNone{T}()
scaleinfo{T<:Signed}(::Type{T}, img::AbstractArray{T}) = ScaleNone{T}()
scaleinfo{To<:FloatingPoint,From}(::Type{To}, img::AbstractArray{From}) = ScaleNone{To}()
scaleinfo{To<:Unsigned,From<:Unsigned}(::Type{To}, img::AbstractArray{From}) = scaleinfo_uint(To, img)
scaleinfo{To<:Unsigned,From<:Union(Integer,FloatingPoint)}(::Type{To}, img::AbstractArray{From}) = scaleinfo_uint(To, img)
function scaleinfo(::Type{RGB}, img::AbstractArray)
    l = climdefault(img)
    ScaleMinMax(Float64, l[1], l[2], 1.0/(l[2]-l[1]))
end

function scaleinfo_uint{To<:Unsigned,From<:Unsigned}(::Type{To}, img::AbstractArray{From})
    l = limits(img)
    n = max(0, iceil(log2(l[2]/(float64(typemax(To))+1))))
    BitShift{To, n}()
end
function scaleinfo_uint{To<:Unsigned,From<:Integer}(::Type{To}, img::AbstractArray{From})
    l = climdefault(img)
    ScaleMinMax(To, zero(From), l[2], typemax(To)/(l[2]-l[1]))
end
function scaleinfo_uint{To<:Unsigned,From<:FloatingPoint}(::Type{To}, img::AbstractArray{From})
    l = climdefault(img)
    ScaleMinMax(To, l[1], l[2], typemax(To)/(l[2]-l[1]))
end

scaleinfo_uint{From<:Unsigned}(img::AbstractImageDirect{From}) = @get img "scalei" scaleinfo_uint(data(img))
scaleinfo_uint{From<:Integer}(img::AbstractImageDirect{From}) = @get img "scalei" scaleinfo_uint(data(img))
scaleinfo_uint{From<:FloatingPoint}(img::AbstractImageDirect{From}) = @get img "scalei" scaleinfo_uint(data(img))
scaleinfo_uint(img::AbstractImageDirect{RGB}) = @get img "scalei" scaleinfo_uint(data(img))

scaleinfo_uint{From<:Unsigned}(img::AbstractArray{From}) = ScaleNone{From}()
scaleinfo_uint{From<:Integer}(img::AbstractArray{From}) = scaleinfo_uint(unsigned(From), img)
scaleinfo_uint{From<:FloatingPoint}(img::AbstractArray{From}) = scaleinfo_uint(Uint8, img)
scaleinfo_uint(img::AbstractArray{RGB}) = ClipMinMax(RGB8, RGB)

climdefault{T<:Integer}(img::AbstractArray{T}) = limits(img)
function climdefault{T<:FloatingPoint}(img::AbstractArray{T})
    l = limits(img)
    if isinf(l[1]) || isinf(l[2])
        l = (zero(T),one(T))
    end
    l
end

unsigned{T<:Unsigned}(::Type{T}) = T
unsigned(::Type{Bool}) = Uint8
unsigned(::Type{Int8}) = Uint8
unsigned(::Type{Int16}) = Uint16
unsigned(::Type{Int32}) = Uint32
unsigned(::Type{Int64}) = Uint64
unsigned(::Type{Int128}) = Uint128

minfinite(A::AbstractArray) = minimum(A)
function minfinite{T<:FloatingPoint}(A::AbstractArray{T})
    ret = nan(T)
    for a in A
        ret = isfinite(a) ? (ret < a ? ret : a) : ret
    end
    ret
end

maxfinite(A::AbstractArray) = maximum(A)
function maxfinite{T<:FloatingPoint}(A::AbstractArray{T})
    ret = nan(T)
    for a in A
        ret = isfinite(a) ? (ret > a ? ret : a) : ret
    end
    ret
end

function maxabsfinite(A::AbstractArray)
    ret = abs(A[1])
    for i = 2:length(A)
        a = abs(A[i])
        ret = a > ret ? a : ret
    end
    ret
end
function maxabsfinite{T<:FloatingPoint}(A::AbstractArray{T})
    ret = nan(T)
    for sa in A
        a = abs(sa)
        ret = isfinite(a) ? (ret > a ? ret : a) : ret
    end
    ret
end

sc(img::AbstractArray) = scale(scaleminmax(img), img)
sc(img::AbstractArray, mn::Real, mx::Real) = scale(scaleminmax(img, mn, mx), img)

convert{T}(::Type{AbstractImageDirect{T,2}},M::Tridiagonal) = error("Not defined") # prevent ambiguity warning

convert{T,S}(::Type{Image{T}}, img::AbstractImageDirect{S}) = scale(scaleinfo(T, img), img)

float32(img::AbstractImageDirect) = scale(scaleinfo(Float32, img), img)
float64(img::AbstractImageDirect) = scale(scaleinfo(Float64, img), img)
function float32sc(img::AbstractImageDirect)
    l = climdefault(img)
    scalei = ScaleMinMax(Float32, l[1], l[2], 1.0f0/(l[2]-l[1]))
    scale(scalei, img)
end
function float64sc(img::AbstractImageDirect)
    l = climdefault(img)
    scalei = ScaleMinMax(Float64, l[1], l[2], 1.0/(l[2]-l[1]))
    scale(scalei, img)
end

uint8sc(img::AbstractImageDirect) = scale(scaleinfo(Uint8, img), img)
uint16sc(img::AbstractImageDirect) = scale(scaleinfo(Uint16, img), img)
uint32sc(img::AbstractImageDirect) = scale(scaleinfo(Uint32, img), img)

convert{C<:ColorValue}(::Type{Image{C}}, img::Image{C}) = img
convert{Cdest<:ColorValue,Csrc<:ColorValue}(::Type{Image{Cdest}}, img::Union(AbstractArray{Csrc},AbstractImageDirect{Csrc})) = share(img, convert(Array{Cdest}, data(img)))

function convert{C<:ColorValue,T<:Union(Integer,FloatingPoint)}(::Type{Image{C}}, img::Union(AbstractArray{T},AbstractImageDirect{T}))
    cs = colorspace(img)
    if !(cs == "RGB" || cs == "RGBA")
        error("Unsupported colorspace $cs of input image. Only RGB and RGBA are currently supported.")
    end
    scalei = scaleinfo(RGB, img)
    cd = colordim(img)
    d = data(img)
    sz = size(img)
    szout = sz[setdiff(1:ndims(img), cd)]
    dout = Array(C, szout)
    if cd == 1
        s = stride(d,2)
        for i in 0:length(dout)-1
            tmp = RGB(scale(scalei,d[i*s+1]), scale(scalei,d[i*s+2]), scale(scalei,d[i*s+3]))
            dout[i+1] = convert(C, tmp)
        end
    elseif cd == ndims(img)
        s = stride(d,cd)
        for i in 1:length(dout)
            tmp = RGB(scale(scalei,d[i]), scale(scalei,d[i+s]), scale(scalei,d[i+2s]))
            dout[i] = convert(C, tmp)
        end
    else
        error("Not yet implemented")
    end
    p = copy(properties(img))
    delete!(p, "colordim")
    delete!(p, "limits")
    delete!(p, "colorspace")
    Image(dout, p)
end

