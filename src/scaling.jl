#### Scaling/clamping/type conversion ####

# Structure of ScaleInfo definitions:
#   - type definition
#   - constructors for scalars
#   - constructors for AbstractArrays
#   - implementation of scale() for scalars
#   - implementation of scale() for AbstractArrays


## Fall-back definitions of scale() for array types

function scale{T}(scalei::ScaleInfo{T}, img::AbstractArray)
    out = similar(img, T)
    scale!(out, scalei, img)
end

@ngenerate N typeof(out) function scale!{T,T1,N}(out, scalei::ScaleInfo{T}, img::AbstractArray{T1,N})
    dimg = data(img)
    dout = data(out)
    @nloops N i dout begin
        @inbounds @nref(N, dout, i) = scale(scalei, @nref(N, dimg, i))
    end
    out
end

## ScaleNone
# At most, this does conversions

immutable ScaleNone{T} <: ScaleInfo{T}; end

# Constructors
ScaleNone{T}(val::T) = ScaleNone{T}()
ScaleNone{T}(A::AbstractArray{T}) = ScaleNone{T}()

# Implementation
scale{T<:Real}(scalei::ScaleNone{T}, val::T) = val
scale{T,S<:Real}(scalei::ScaleNone{T}, val::S) = convert(T, val)
scale{T<:ColorType}(scalei::ScaleNone{T}, val::T) = val
scale(scalei::ScaleNone{Uint32}, val::ColorValue) = convert(Uint32, convert(RGB24, val))
scale(scalei::ScaleNone{Uint32}, val::AlphaColorValue) = convert(Uint32, convert(RGBA32, val))

scale{T}(scalei::ScaleNone{T}, img::AbstractArray{T}) = img

## BitShift

immutable BitShift{T<:Union(Integer,Ufixed),N} <: ScaleInfo{T} end

# Constructors
BitShift{T}(::Type{T}, N::Integer) = BitShift{T,N}()

# Implementation
scale{T,N}(scalei::BitShift{T,N}, val::Integer) = convert(T, val>>>N)
scale{T<:Ufixed,N}(scalei::BitShift{T,N}, val::Ufixed) = reinterpret(T, convert(FixedPointNumbers.rawtype(T), reinterpret(val)>>>N))

## Clamp types

# The Clamp types just enforce bounds, but do not scale or offset
# Types and constructors
abstract Clamp{T} <: ScaleInfo{T}
immutable ClampMin{T,From} <: Clamp{T}
    min::From
end
ClampMin{T,From}(::Type{T}, min::From) = ClampMin{T,From}(min)
ClampMin{T}(::Type{RGB{T}}) = ClampMin(T,zero(T))
immutable ClampMax{T,From} <: Clamp{T}
    max::From
end
ClampMax{T,From}(::Type{T}, max::From) = ClampMax{T,From}(max)
ClampMax{T}(::Type{RGB{T}}) = ClampMax(T,one(T))
immutable ClampMinMax{T,From} <: Clamp{T}
    min::From
    max::From
end
ClampMinMax{T,From}(::Type{T}, min::From, max::From) = ClampMinMax{T,From}(min,max)
ClampMinMax{T}(::Type{RGB{T}}) = ClampMinMax(RGB{T},zero(RGB{T}),one(RGB{T}))


# Implementation
scale{T<:Real,F<:Real}(scalei::ClampMin{T,F}, val::F) = convert(T, max(val, scalei.min))
scale{T<:Real,F<:Real}(scalei::ClampMax{T,F}, val::F) = convert(T, min(val, scalei.max))
scale{T<:Real,F<:Real}(scalei::ClampMinMax{T,F}, val::F) = convert(T,min(max(val, scalei.min), scalei.max))
scale{CV<:ColorType}(scalei::ClampMinMax{CV,CV}, val::CV) = mincv(maxcv(val, scalei.min), scalei.max)
scale{T<:ColorType, F<:ColorType}(scalei::ClampMinMax{T,F}, val::F) = convert(T, mincv(maxcv(val, scalei.min), scalei.max))

mincv{T}(c1::RGB{T}, c2::RGB{T}) = RGB{T}(min(getfield(c1,1),getfield(c2,1)), min(getfield(c1,2),getfield(c2,2)), min(getfield(c1,3),getfield(c2,3)))
maxcv{T}(c1::RGB{T}, c2::RGB{T}) = RGB{T}(max(getfield(c1,1),getfield(c2,1)), max(getfield(c1,2),getfield(c2,2)), max(getfield(c1,3),getfield(c2,3)))

# For certain Ufixed types and Clamp values, scale() should be a no-op. These are purely for efficiency.
scale{T<:Ufixed}(scalei::ClampMin{T,T}, img::AbstractArray{T}) =
    scalei.min == zero(T) ? img : invoke(scale, (ScaleInfo{T}, AbstractArray), scalei, img)
scale{T<:Ufixed}(scalei::ClampMin{RGB{T},RGB{T}}, img::AbstractArray{RGB{T}}) =
    scalei.min == zero(RGB{T}) ? img : invoke(scale, (ScaleInfo{T}, AbstractArray), scalei, img)
scale{T<:Union(Ufixed8,Ufixed16)}(scalei::ClampMax{T,T}, img::AbstractArray{T}) =
    scalei.max == one(T) ? img : invoke(scale, (ScaleInfo{T}, AbstractArray), scalei, img)
scale{T<:Union(Ufixed8,Ufixed16)}(scalei::ClampMax{RGB{T},RGB{T}}, img::AbstractArray{RGB{T}}) =
    scalei.max == one(RGB{T}) ? img : invoke(scale, (ScaleInfo{T}, AbstractArray), scalei, img)
scale{T<:Union(Ufixed8,Ufixed16)}(scalei::ClampMinMax{T,T}, img::AbstractArray{T}) =
    scalei.min == zero(T) && scalei.max == one(T) ? img : invoke(scale, (ScaleInfo{T}, AbstractArray), scalei, img)
scale{T<:Union(Ufixed8,Ufixed16)}(scalei::ClampMinMax{RGB{T},RGB{T}}, img::AbstractArray{RGB{T}}) =
    scalei.min == zero(RGB{T}) && scalei.max == one(RGB{T}) ? img : invoke(scale, (ScaleInfo{T}, AbstractArray), scalei, img)

clamp{T}(v::RGB{T}) = scale(ClampMinMax(RGB{T}), v)


## ScaleMinMax

# This subtracts the min value and scales
immutable ScaleMinMax{To,From,S<:FloatingPoint} <: ScaleInfo{To}
    min::From
    max::From
    s::S
end
ScaleMinMax{To,From}(::Type{To}, min::From, max::From, s::FloatingPoint) = ScaleMinMax{To,From,typeof(s)}(min, max, s)
ScaleMinMax{To<:Real,From<:Real}(::Type{To}, mn::From, mx::From) = ScaleMinMax(To, mn, mx, 1.0f0/(mx-mn))

# ScaleMinMax constructors that take AbstractArray input
ScaleMinMax{To<:FloatingPoint,From}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(mx-mn))
ScaleMinMax{To<:Ufixed,From}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(mx-mn))
ScaleMinMax{To}(::Type{To}, img::AbstractArray) = ScaleMinMax(To, img, minfinite(img), maxfinite(img))
ScaleMinMax(img::AbstractArray) = ScaleMinMax(Ufixed8, img)
ScaleMinMax(img::AbstractArray, mn::Real, mx::Real) = ScaleMinMax(Ufixed8, img, mn, mx)

function scale{To<:Real,From<:Real}(scalei::ScaleMinMax{To,From}, val::From)
    t::From = (val > scalei.min) ? ((val < scalei.max) ? val-scalei.min : scalei.max-scalei.min) : zero(From)
    convert(To, scalei.s*t)
end

scale{To,From}(scalei::ScaleMinMax{To,From}, img::AbstractArray{From}) =
    scalei.min == typemin(From) && scalei.max == typemax(From) && scalei.s == 1.0 ?
        scale(ScaleNone{To}, img) :
        invoke(scale, (ScaleInfo{To}, AbstractArray), scalei, img)

## ScaleSigned

# Multiplies by a scaling factor and then clamps to the range [-1,1].
# Intended for positive/negative coloring
immutable ScaleSigned{S<:FloatingPoint} <: ScaleInfo{S}
    s::S
end
ScaleSigned{S<:FloatingPoint}(s::S) = ScaleSigned{S}(s)

scalesigned(img::AbstractArray) = ScaleSigned(1.0f0/maxabsfinite(img))
function scalesigned(img::AbstractImage, tindex::Integer)
    1 <= tindex <= nimages(img) || error("The image does not have a time slice of ", tindex)
    imsl = sliceim(img, "t", 1)
    ScaleSigned(1.0f0/maxabsfinite(imsl))
end

scale(scalei::ScaleSigned, val::Real) = clamppm(scalei.s*val)

clamppm{T}(sval::T) = ifelse(sval>one(T), one(T), ifelse(sval<-one(T), -one(T), sval))

## ScaleAutoMinMax

# Works only on whole arrays, not values
immutable ScaleAutoMinMax{T} <: ScaleInfo{T} end
ScaleAutoMinMax() = ScaleAutoMinMax{Ufixed8}()

scale!{T}(out::AbstractImage, scalei::ScaleAutoMinMax{T}, img::Union(StridedArray,AbstractImageDirect)) = scale!(out, take(scalei, img), img)
scale!{T}(out, scalei::ScaleAutoMinMax{T}, img::Union(StridedArray,AbstractImageDirect)) = scale!(out, take(scalei, img), img)

take(scalei::ScaleInfo, img::AbstractArray) = scalei
take{T}(scalei::ScaleAutoMinMax{T}, img::AbstractArray) = ScaleMinMax(T, img)
take{To,From}(scalei::ScaleMinMax{To}, img::AbstractArray{From}) = ScaleMinMax(To, convert(From, scalei.min), convert(From, scalei.max), scalei.s)




#### ScaleInfo defaults

# scaleinfo{T}(::Type{Any}, img::AbstractArray{T}) = ScaleNone{T}()

scaleinfo{T<:Ufixed}(img::AbstractArray{T}) = ScaleNone(img)  # do we need to restrict to Ufixed8 & Ufixed16?
scaleinfo{CV<:ColorType}(img::AbstractArray{CV}) = _scaleinfocv(CV, img)
_scaleinfocv{CV}(::Type{CV}, img) = _scaleinfocv(eltype(CV), img)
_scaleinfocv{T<:Ufixed}(::Type{T}, img) = ScaleNone(img)
_scaleinfocv{T<:FloatingPoint}(::Type{T}, img) = scaleinfo(RGB{Ufixed8}, img)
function scaleinfo{T}(img::AbstractArray{T})
    cs = colorspace(img)
    if cs == "RGB24" || cs == "ARGB32"
        return ScaleNone{T}()
    end
    scaleinfo(Ufixed8, img)
end
# scaleinfo{T<:FloatingPoint}(img::AbstractArray{RGB{T}}) = scaleinfo(RGB{Ufixed8}, img)

scaleinfo{T<:Ufixed}(::Type{T}, img::AbstractArray{T}) = ScaleNone(img)
for (T,n) in ((Ufixed10, 2), (Ufixed12, 4), (Ufixed14, 6), (Ufixed16, 8))
    @eval scaleinfo(::Type{Ufixed8}, img::AbstractArray{$T}) = BitShift(Ufixed8,$n)
end
scaleinfo{To<:Ufixed,From<:FloatingPoint}(::Type{To}, img::AbstractArray{From}) =
    ClampMinMax(To, zero(From), one(From))
scaleinfo{T}(::Type{T}, img::AbstractArray{T}) = ScaleNone(img)
scaleinfo{To<:Ufixed}(::Type{RGB{To}}, img::AbstractArray{RGB{To}}) = ScaleNone(img)
scaleinfo{To<:Ufixed,From<:FloatingPoint}(::Type{RGB{To}}, img::AbstractArray{RGB{From}}) =
    ClampMinMax(RGB{To}, zero(RGB{From}), one(RGB{From}))

#=
function scaleinfo_uint{To<:Ufixed8,From<:Ufixed8}(::Type{To}, img::AbstractArray{From})
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
scaleinfo_uint{T<:Ufixed}(img::AbstractArray{RGB{T}}) = ScaleNone(RGB{T})
scaleinfo_uint{T<:FloatingPoint}(img::AbstractArray{RGB{T}}) = ClampMinMax(RGB{T})

climdefault{T<:Integer}(img::AbstractArray{T}) = limits(img)
function climdefault{T<:FloatingPoint}(img::AbstractArray{T})
    l = limits(img)
    if isinf(l[1]) || isinf(l[2])
        l = (zero(T),one(T))
    end
    l
end
=#

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

sc(img::AbstractArray) = scale(ScaleMinMax(img), img)
sc(img::AbstractArray, mn::Real, mx::Real) = scale(ScaleMinMax(img, mn, mx), img)

convert{T}(::Type{AbstractImageDirect{T,2}},M::Tridiagonal) = error("Not defined") # prevent ambiguity warning

convert{T<:Real,S<:Real}(::Type{Image{T}}, img::AbstractImageDirect{S}) = scale(scaleinfo(T, img), img)

for (fn,T) in ((:float32, Float32), (:float64, Float64), (:ufixed8, Ufixed8),
               (:ufixed10, Ufixed10), (:ufixed12, Ufixed12), (:ufixed14, Ufixed14),
               (:ufixed16, Ufixed16))
    @eval begin
        function $fn{C<:ColorType}(A::AbstractArray{C})
            newC = eval(C.name.name){$T}
            convert(Array{newC}, A)
        end
        $fn{C<:ColorType}(img::AbstractImage{C}) = share(img, $fn(data(img)))
    end
end


ufixedsc{T<:Ufixed}(::Type{T}, img::AbstractImageDirect) = scale(scaleinfo(T, img), img)
ufixed8sc(img::AbstractImageDirect) = ufixedsc(Ufixed8, img)
