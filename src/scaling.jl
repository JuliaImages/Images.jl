#### Scaling/clamping/type conversion ####

# This file exists primarily to handle conversions for display and
# saving to disk. Both of these operations require Ufixed-valued
# elements, but with display we always want to convert to 8-bit
# whereas saving can handle 16-bit.
# We also can't trust that user images are clamped properly.
# Finally, this supports adjustable contrast limits.

# Structure of ScaleInfo definitions:
#   - type definition
#   - constructors for scalars
#   - constructors for AbstractArrays
#   - implementation of scale() for scalars
#   - implementation of scale() for AbstractArrays
# scale(scalei::ScaleInfo{T}, x) should return an object of type T (for x not an array)
# scale1(scalei::ScaleInfo{T}, x) is designed to allow T<:ColorValue to work on
#    scalars x::Fractional

## ScaleNone
# At most, this does conversions

immutable ScaleNone{T} <: ScaleInfo{T}; end

# Constructors
ScaleNone{T}(val::T) = ScaleNone{T}()
ScaleNone{T}(A::AbstractArray{T}) = ScaleNone{T}()

# Implementation
scale{T<:Real}(scalei::ScaleNone{T}, val::Real) = convert(T, val)
scale1(scalei::Union(ScaleNone{RGB24},ScaleNone{ARGB32}), val::Fractional) = convert(Ufixed8, val)
scale1{CT<:ColorType}(scalei::ScaleNone{CT}, val::Fractional) = convert(eltype(CT), val)

scale{T}(scalei::ScaleNone{T}, img::AbstractArray{T}) = img


## BitShift
# This is really a "saturating bitshift", for example
#    scale(BitShift{Uint8,7}(), 0xf0ff) == 0xff rather than 0xe1 even though 0xf0ff>>>7 == 0x01e1

immutable BitShift{T,N} <: ScaleInfo{T} end

# Must directly use BitShift{T,N}() to construct, because passing an argument N would not be type stable

# Implementation
immutable BS{N} end
_scale{T<:Unsigned,N}(::Type{T}, ::Type{BS{N}}, val::Unsigned) = (v = val>>>N; tm = oftype(val, typemax(T)); convert(T, ifelse(v > tm, tm, v)))
_scale{T<:Ufixed,N}(::Type{T}, ::Type{BS{N}}, val::Ufixed) = reinterpret(T, _scale(FixedPointNumbers.rawtype(T), BS{N}, reinterpret(val)))
scale{T<:Real,N}(scalei::BitShift{T,N}, val::Real) = _scale(T, BS{N}, val)
scale1{N}(scalei::Union(BitShift{RGB24,N},BitShift{ARGB32,N}), val::Unsigned) = _scale(Uint8, BS{N}, val)
scale1{N}(scalei::Union(BitShift{RGB24,N},BitShift{ARGB32,N}), val::Ufixed) = _scale(Ufixed8, BS{N}, val)
scale1{CT<:ColorType,N}(scalei::BitShift{CT,N}, val::Ufixed) = _scale(eltype(CT), BS{N}, val)


## Clamp types
# The Clamp types just enforce bounds, but do not scale or offset

# Types and constructors
abstract Clamp{T} <: ScaleInfo{T}
immutable ClampMin{T,From} <: Clamp{T}
    min::From
end
ClampMin{T,From}(::Type{T}, min::From) = ClampMin{T,From}(min)
immutable ClampMax{T,From} <: Clamp{T}
    max::From
end
ClampMax{T,From}(::Type{T}, max::From) = ClampMax{T,From}(max)
immutable ClampMinMax{T,From} <: Clamp{T}
    min::From
    max::From
end
ClampMinMax{T,From}(::Type{T}, min::From, max::From) = ClampMinMax{T,From}(min,max)
immutable Clamp01{T} <: Clamp{T} end  # specialized for clamping between 0 and 1

# Implementation
scale{T<:Real,F<:Real}(scalei::ClampMin{T,F}, val::F) = convert(T, max(val, scalei.min))
scale{T<:Real,F<:Real}(scalei::ClampMax{T,F}, val::F) = convert(T, min(val, scalei.max))
scale{T<:Real,F<:Real}(scalei::ClampMinMax{T,F}, val::F) = convert(T,min(max(val, scalei.min), scalei.max))
scale1{T<:Union(RGB24,ARGB32),F<:Fractional}(scalei::ClampMin{T,F}, val::F) = convert(Ufixed8, max(val, scalei.min))
scale1{T<:Union(RGB24,ARGB32),F<:Fractional}(scalei::ClampMax{T,F}, val::F) = convert(Ufixed8, min(val, scalei.max))
scale1{T<:Union(RGB24,ARGB32),F<:Fractional}(scalei::ClampMinMax{T,F}, val::F) = convert(Ufixed8,min(max(val, scalei.min), scalei.max))
scale1{CT<:ColorType,F<:Fractional}(scalei::ClampMin{CT,F}, val::F) = convert(eltype(CT), max(val, scalei.min))
scale1{CT<:ColorType,F<:Fractional}(scalei::ClampMax{CT,F}, val::F) = convert(eltype(CT), min(val, scalei.max))
scale1{CT<:ColorType,F<:Fractional}(scalei::ClampMinMax{CT,F}, val::F) = convert(eltype(CT), min(max(val, scalei.min), scalei.max))

scale{To<:Real}(::Clamp01{To}, val::Real) = clamp01(To, val)
scale1{CT<:ColorType}(::Clamp01{CT}, val::Real) = clamp01(eltype(CT), val)

# Also available as a stand-alone function
clamp01{T}(::Type{T}, x::Real) = min(max(convert(T, x), zero(T)), one(T))
clamp01(x::Real) = clamp01(typeof(x), x)
clamp01{To}(::Type{RGB{To}}, x::AbstractRGB) = RGB{To}(clamp01(To, x.r), clamp01(To, x.g), clamp01(To, x.b))
clamp01{T}(x::AbstractRGB{T}) = RGB{T}(clamp01(T, x.r), clamp01(T, x.g), clamp01(T, x.b))

## ScaleMinMax
# This clamps, subtracts the min value, then scales

immutable ScaleMinMax{To,From,S<:FloatingPoint} <: ScaleInfo{To}
    min::From
    max::From
    s::S
end

ScaleMinMax{To,From}(::Type{To}, min::From, max::From, s::FloatingPoint) = ScaleMinMax{To,From,typeof(s)}(min, max, s)
ScaleMinMax{To<:Fractional,From<:Real}(::Type{To}, mn::From, mx::From) = ScaleMinMax(To, mn, mx, 1.0f0/(mx-mn))

# ScaleMinMax constructors that take AbstractArray input
ScaleMinMax{To,From}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(mx-mn))
ScaleMinMax{To}(::Type{To}, img::AbstractArray) = ScaleMinMax(To, img, minfinite(img), maxfinite(img))

function scale{To<:Real,From<:Real}(scalei::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < scalei.min, zero(From), ifelse(val > scalei.max, scalei.max-scalei.min, val-scalei.min))
    convert(To, scalei.s*t)
end
function scale1{To<:Union(RGB24,ARGB32),From<:Fractional}(scalei::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < scalei.min, zero(From), ifelse(val > scalei.max, scalei.max-scalei.min, val-scalei.min))
    convert(Ufixed8, scalei.s*t)
end
function scale1{To<:ColorType,From<:Fractional}(scalei::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < scalei.min, zero(From), ifelse(val > scalei.max, scalei.max-scalei.min, val-scalei.min))
    convert(eltype(To), scalei.s*t)
end


## ScaleSigned
# Multiplies by a scaling factor and then clamps to the range [-1,1].
# Intended for positive/negative coloring

immutable ScaleSigned{S<:FloatingPoint} <: ScaleInfo{S}
    s::S
end
ScaleSigned{S<:FloatingPoint}(s::S) = ScaleSigned{S}(s)

ScaleSigned(img::AbstractArray) = ScaleSigned(1.0f0/maxabsfinite(img))

scale(scalei::ScaleSigned, val::Real) = clamppm(scalei.s*val)

clamppm{T}(sval::T) = ifelse(sval>one(T), one(T), ifelse(sval<-one(T), -one(T), sval))

## ScaleAutoMinMax
# Works only on whole arrays, not values

immutable ScaleAutoMinMax{T} <: ScaleInfo{T} end
ScaleAutoMinMax() = ScaleAutoMinMax{Ufixed8}()




# Conversions to RGB{T}, RGBA{T}, RGB24, ARGB32,
# for grayscale, AbstractRGB, and abstract ARGB inputs
for SI in (ScaleInfo, Clamp)
    for ST in subtypes(SI)
        ST.abstract && continue
        ST == ScaleSigned && continue
        @eval begin
            # Grayscale and GrayAlpha inputs
            scale(scalei::$ST{RGB24}, g::Gray) = scale(scalei, g.val)
            scale(scalei::$ST{RGB24}, g::Real) = (x = scale1(scalei, g); convert(RGB24, RGB{Ufixed8}(x,x,x)))
            scale(scalei::$ST{ARGB32}, g::Gray) = scale(scalei, g.val)
            function scale(scalei::$ST{ARGB32}, g::Real)
                x = scale1(scalei, g)
                convert(ARGB32, ARGB{Ufixed8}(x,x,x))
            end
            function scale(scalei::$ST{ARGB32}, g::GrayAlpha)
                x = scale1(scalei, g.c.val)
                convert(ARGB32, ARGB{Ufixed8}(x,x,x,scale1(scalei, g.alpha)))
            end
            scale{T}(scalei::$ST{RGB{T}}, g::Gray) = scale(scalei, g.val)
            function scale{T}(scalei::$ST{RGB{T}}, g::Real)
                x = scale1(scalei, g)
                RGB{T}(x,x,x)
            end
            scale{T}(scalei::$ST{ARGB{T}}, g::Gray) = scale(scalei, g.val)
            function scale{T}(scalei::$ST{ARGB{T}}, g::Real)
                x = scale1(scalei, g)
                AlphaColor{RGB{T}, T}(x,x,x)
            end
            function scale{T}(scalei::$ST{ARGB{T}}, g::GrayAlpha)
                x = scale1(scalei, g.c.val)
                AlphaColor{RGB{T}, T}(x,x,x,scale1(scalei, g.alpha))
            end
            # AbstractRGB and abstract ARGB inputs
            scale(scalei::$ST{RGB24}, rgb::AbstractRGB) =
                convert(RGB24, RGB{Ufixed8}(scale1(scalei, rgb.r), scale1(scalei, rgb.g), scale1(scalei, rgb.b)))
            scale{C<:AbstractRGB, TC}(scalei::$ST{ARGB32}, argb::AbstractAlphaColorValue{C,TC}) =
                convert(ARGB32, ARGB{Ufixed8}(scale1(scalei, argb.r), scale1(scalei, argb.g),
                                              scale1(scalei, argb.b), scale1(scalei, argb.alpha)))
            scale{T}(scalei::$ST{RGB{T}}, rgb::AbstractRGB) =
                RGB{T}(scale1(scalei, rgb.r), scale1(scalei, rgb.g), scale1(scalei, rgb.b))
            scale{T, C<:AbstractRGB, TC}(scalei::$ST{ARGB{T}}, argb::AbstractAlphaColorValue{C,TC}) =
                AlphaColor{RGB{T}, T}(scale1(scalei, argb.c.r), scale1(scalei, argb.c.g),
                                      scale1(scalei, argb.c.b), scale1(scalei, argb.alpha))
        end
    end
end

# Apply to any ColorType
scale(scalei, x::ColorValue) = scale(scalei, convert(RGB, x))
scale{C<:ColorValue, TC}(scalei, x::AbstractAlphaColorValue{C, TC}) = scale(scalei, convert(ARGB, x))

## Fallback definitions of scale() for array types

function scale{T}(scalei::ScaleInfo{T}, img::AbstractArray)
    out = similar(img, T)
    scale!(out, scalei, img)
end
function scale{T}(scalei::ScaleInfo{T}, img::AbstractImageIndexed)
    out = Image(Array(T, size(img)), properties(img))
    scale!(out, scalei, img)
end

@ngenerate N typeof(out) function scale!{T,T1,N}(out::AbstractArray{T,N}, scalei::ScaleInfo{T}, img::AbstractArray{T1,N})
    si = take(scalei, img)
    dimg = data(img)
    dout = data(out)
    @nloops N i dout begin
        @inbounds @nref(N, dout, i) = scale(si, @nref(N, dimg, i))
    end
    out
end

take(scalei::ScaleInfo, img::AbstractArray) = scalei
take{T}(scalei::ScaleAutoMinMax{T}, img::AbstractArray) = ScaleMinMax(T, img)
take{To,From}(scalei::ScaleMinMax{To}, img::AbstractArray{From}) = ScaleMinMax(To, convert(From, scalei.min), convert(From, scalei.max), scalei.s)

# Indexed images (colormaps)
@ngenerate N typeof(out) function scale!{T,T1,N}(out::AbstractArray{T,N}, scalei::ScaleInfo{T}, img::AbstractImageIndexed{T1,N})
    dimg = data(img)
    dout = data(out)
    cmap = scale(si, img.cmap)
    @nloops N i dout begin
        @inbounds @nref(N, dout, i) = cmap[@nref(N, dimg, i)]
    end
    out
end



#### ScaleInfo defaults
# Each "client" can define its own methods. "clients" include Ufixed, RGB24/ARGB32, and ImageMagick

const bitshiftto8 = ((Ufixed10, 2), (Ufixed12, 4), (Ufixed14, 6), (Ufixed16, 8))

# scaleinfo{T}(::Type{T}, img::AbstractArray{T}) = ScaleNone(img)
println("Defining scaleinfo methods")
# Grayscale methods
for (T,n) in bitshiftto8
    @eval scaleinfo(::Type{Ufixed8}, img::AbstractArray{$T}) = BitShift{Ufixed8,$n}()
end
scaleinfo{T<:Ufixed,F<:FloatingPoint}(::Type{T}, img::AbstractArray{F}) = ClampMinMax(T, zero(F), one(F))

# Color->RGB24/ARGB32
for C in subtypes(AbstractRGB)
    @eval scaleinfo(::Type{RGB24}, img::AbstractArray{$C{Ufixed8}}) = ScaleNone{RGB24}()
    for (T, n) in bitshiftto8
        @eval scaleinfo(::Type{RGB24}, img::AbstractArray{$C{$T}}) = BitShift{RGB24, $n}()
    end
    @eval scaleinfo{F<:FloatingPoint}(::Type{RGB24}, img::AbstractArray{$C{F}}) = ClampMinMax(RGB24, zero(F), one(F))
    for AC in subtypes(AbstractAlphaColorValue)
        length(AC.parameters) == 2 || continue
        @eval scaleinfo(::Type{ARGB32}, img::AbstractArray{$AC{$C{Ufixed8},Ufixed8}}) = ScaleNone{ARGB32}()
        for (T, n) in bitshiftto8
            @eval scaleinfo(::Type{ARGB32}, img::AbstractArray{$AC{$C{$T},$T}}) = BitShift{ARGB32, $n}()
        end
        @eval scaleinfo{F<:FloatingPoint}(::Type{ARGB32}, img::AbstractArray{$AC{$C{F},F}}) = ClampMinMax(ARGB32, zero(F), one(F))
    end
end
scaleinfo{CV<:ColorValue}(::Type{Uint32}, img::AbstractArray{CV}) = scaleinfo(RGB24, img)
scaleinfo{CV<:AbstractAlphaColorValue}(::Type{Uint32}, img::AbstractArray{CV}) = scaleinfo(ARGB32, img)

# ImageMagick. Converts to RGB and uses Ufixed.
scaleinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{T}) = ScaleNone{T}()
scaleinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{T}) = ClampMinMax(Ufixed8, zero(T), one(T))
for ACV in (ColorValue, AbstractRGB,AbstractGray)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && isleaftype(CV{Float64})) || continue
        @show CV
        CVnew = CV<:AbstractGray ? Gray : RGB
        @eval scaleinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{$CV{T}}) = ScaleNone{$CVnew{T}}()
        @eval scaleinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{$CV{T}}) =
            Clamp01{$CVnew{Ufixed8}}()
        CVnew = CV<:AbstractGray ? Gray : BGR
        for AC in subtypes(AbstractAlphaColorValue)
            length(AC.parameters) == 2 || continue
            @eval scaleinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{$AC{$CV{T},T}}) = ScaleNone{$AC{$CVnew{T},T}}()
            @eval scaleinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{$AC{$CV{T},T}}) = Clamp01{$AC{$CVnew{Ufixed8}, Ufixed8}}()
        end
    end
end


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
