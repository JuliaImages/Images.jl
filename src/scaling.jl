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

scale{T<:ColorType}(scalei::ScaleNone{T}, img::AbstractImageIndexed{T}) = convert(Image{T}, img)
scale{C<:ColorType}(scalei::ScaleNone{C}, img::AbstractImageDirect{C}) = img  # ambiguity resolution
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
abstract AbstractClamp{T} <: ScaleInfo{T}
immutable ClampMin{T,From} <: AbstractClamp{T}
    min::From
end
ClampMin{T,From}(::Type{T}, min::From) = ClampMin{T,From}(min)
immutable ClampMax{T,From} <: AbstractClamp{T}
    max::From
end
ClampMax{T,From}(::Type{T}, max::From) = ClampMax{T,From}(max)
immutable ClampMinMax{T,From} <: AbstractClamp{T}
    min::From
    max::From
end
ClampMinMax{T,From}(::Type{T}, min::From, max::From) = ClampMinMax{T,From}(min,max)
immutable Clamp{T} <: AbstractClamp{T} end  # specialized for clamping colorvalues (e.g., 0 to 1 for RGB, also fractional)

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

scale{To<:Real}(::Clamp{To}, val::Real) = clamp01(To, val)
scale1{CT<:Union(AbstractRGB,AbstractRGBA)}(::Clamp{CT}, val::Real) = clamp01(eltype(CT), val)

# Also available as a stand-alone function
clamp01{T}(::Type{T}, x::Real) = convert(T, min(max(x, zero(x)), one(x)))
clamp01(x::Real) = clamp01(typeof(x), x)
clamp01{To}(::Type{RGB{To}}, x::AbstractRGB) = RGB{To}(clamp01(To, x.r), clamp01(To, x.g), clamp01(To, x.b))
clamp01{T}(x::AbstractRGB{T}) = clamp01(RGB{T}, x)

clamp(x::AbstractRGB) = clamp01(x)

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

immutable ScaleSigned{T, S<:FloatingPoint} <: ScaleInfo{T}
    s::S
end
ScaleSigned{T}(::Type{T}, s::FloatingPoint) = ScaleSigned{T, typeof(s)}(s)

ScaleSigned{T}(::Type{T}, img::AbstractArray) = ScaleSigned(T, 1.0f0/maxabsfinite(img))
ScaleSigned(img::AbstractArray) = ScaleSigned(Float32, img)

scale{T}(scalei::ScaleSigned{T}, val::Real) = clamppm(T, scalei.s*val)
function scale{C<:Union(RGB24, RGB{Ufixed8})}(scalei::ScaleSigned{C}, val::Real)
    x = clamppm(scalei.s*val)
    g = Ufixed8(abs(x))
    ifelse(x >= 0, C(g, zero(Ufixed8), g), C(zero(Ufixed8), g, zero(Ufixed8)))
end

clamppm(x::Real) = ifelse(x >= 0, min(x, one(x)), max(x, -one(x)))

## ScaleAutoMinMax
# Works only on whole arrays, not values

immutable ScaleAutoMinMax{T} <: ScaleInfo{T} end
ScaleAutoMinMax() = ScaleAutoMinMax{Ufixed8}()




# Conversions to RGB{T}, RGBA{T}, RGB24, ARGB32,
# for grayscale, AbstractRGB, and abstract ARGB inputs.
# This essentially "vectorizes" scale using scale1
for SI in (ScaleInfo, AbstractClamp)
    for ST in subtypes(SI)
        ST.abstract && continue
        ST == ScaleSigned && continue  # ScaleSigned gives an RGB from a scalar, so don't "vectorize" it
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
                convert(ARGB32, ARGB{Ufixed8}(scale1(scalei, argb.c.r), scale1(scalei, argb.c.g),
                                              scale1(scalei, argb.c.b), scale1(scalei, argb.alpha)))
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

scale{C<:ColorType,R<:Real}(scalei::ScaleNone{C}, img::AbstractImageDirect{R}) = scalecd(scalei, img)  # ambiguity resolution
scale{C<:ColorType,R<:Real}(scalei::ScaleInfo{C}, img::AbstractImageDirect{R}) = scalecd(scalei, img)
function scalecd{C<:ColorType,R<:Real}(scalei::ScaleInfo{C}, img::AbstractImageDirect{R})
    # For this case we have to check whether color is defined along an array axis
    cd = colordim(img)
    if cd > 0
        dims = setdiff(1:ndims(img), cd)
        out = similar(img, C, size(img)[dims])
        scale!(out, scalei, img, TypeConst{cd})
    else
        out = similar(img, C)
        scale!(out, scalei, img)
    end
    out   # note this isn't type-stable
end

function scale{T<:ColorType}(scalei::ScaleInfo{T}, img::AbstractImageIndexed)
    out = Image(Array(T, size(img)), properties(img))
    scale!(out, scalei, img)
end

@ngenerate N typeof(out) function scale!{T,T1,T2,N}(out::AbstractArray{T,N}, scalei::ScaleInfo{T1}, img::AbstractArray{T2,N})
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

# For when color is encoded along dimension CD
# NC is the number of color channels
# This is a very flexible implementation: color can be stored along any dimension, and it handles conversions to
# many different colorspace representations.
for (CT, NC) in ((Union(AbstractRGB,RGB24), 3), (Union(AbstractRGBA,ARGB32), 4), (Union(GrayAlpha,AGray32), 2))
    for N = 1:4
        N1 = N+1
        @eval begin
function scale!{T<:$CT,T1,CD}(out::AbstractArray{T,$N}, scalei::ScaleInfo{T}, img::AbstractArray{T1,$N1}, ::Type{TypeConst{CD}})
    si = take(scalei, img)
    dimg = data(img)
    dout = data(out)
    # Set up the index along the color axis
    # We really only need dimension CD, but this will suffice
    @nexprs $NC k->(@nexprs $N1 d->(j_k_d = k))
    # Loop over all the elements in the output, performing the conversion on each color component
    @nloops $N i dout d->(d<CD ? (@nexprs $NC k->(j_k_d = i_d)) : (@nexprs $NC k->(j_k_{d+1} = i_d))) begin
        @inbounds @nref($N, dout, i) = @ncall $NC T k->(scale1(si, @nref($N1, dimg, j_k)))
    end
    out
end
        end
    end
end


#### ScaleInfo defaults
# Each "client" can define its own methods. "clients" include Ufixed, RGB24/ARGB32, and ImageMagick

const bitshiftto8 = ((Ufixed10, 2), (Ufixed12, 4), (Ufixed14, 6), (Ufixed16, 8))

# typealias GrayType{T<:Fractional} Union(T, Gray{T})
typealias GrayArray{T<:Fractional} Union(AbstractArray{T}, AbstractArray{Gray{T}})
# note, though, that we need to override for AbstractImage in case the "colorspace" property is defined differently

# scaleinfo{T}(::Type{T}, img::AbstractArray{T}) = ScaleNone(img)
# Grayscale methods
for (T,n) in bitshiftto8
    @eval scaleinfo(::Type{Ufixed8}, img::GrayArray{$T}) = BitShift{Ufixed8,$n}()
    @eval scaleinfo(::Type{Gray{Ufixed8}}, img::GrayArray{$T}) = BitShift{Gray{Ufixed8},$n}()
end
scaleinfo{T<:Ufixed,F<:FloatingPoint}(::Type{T}, img::AbstractArray{F}) = ClampMinMax(T, zero(F), one(F))
scaleinfo{F<:Fractional}(::Type{RGB24}, img::GrayArray{F}) = ClampMinMax(RGB24, zero(F), one(F))
scaleinfo{F<:Fractional}(::Type{ARGB32}, img::AbstractArray{F}) = ClampMinMax(ARGB32, zero(F), one(F))
scaleinfo(::Type{RGB24}, img::AbstractArray{RGB24}) = ScaleNone{RGB24}()
scaleinfo(::Type{ARGB32}, img::AbstractArray{ARGB32}) = ScaleNone{ARGB32}()


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

# Uint32 conversions will use ARGB32 for images that have an alpha channel,
# and RGB24 when not
scaleinfo{CV<:Union(Fractional,ColorValue)}(::Type{Uint32}, img::AbstractArray{CV}) = scaleinfo(RGB24, img)
scaleinfo{CV<:AbstractAlphaColorValue}(::Type{Uint32}, img::AbstractArray{CV}) = scaleinfo(ARGB32, img)
scaleinfo(::Type{Uint32}, img::AbstractArray{Uint32}) = ScaleNone{Uint32}()  # define and use a Ufixed18 if you need 32 bits for your camera!

# ImageMagick. Converts to RGB and uses Ufixed.
scaleinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{T}) = ScaleNone{T}()
scaleinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{T}) = ClampMinMax(Ufixed8, zero(T), one(T))
for ACV in (ColorValue, AbstractRGB,AbstractGray)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && !(CV.abstract)) || continue
        CVnew = CV<:AbstractGray ? Gray : RGB
        @eval scaleinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{$CV{T}}) = ScaleNone{$CVnew{T}}()
        @eval scaleinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{$CV{T}}) =
            Clamp{$CVnew{Ufixed8}}()
        CVnew = CV<:AbstractGray ? Gray : BGR
        for AC in subtypes(AbstractAlphaColorValue)
            (length(AC.parameters) == 2 && !(AC.abstract)) || continue
            @eval scaleinfo{T<:Ufixed}(::Type{ImageMagick}, img::AbstractArray{$AC{$CV{T},T}}) = ScaleNone{$AC{$CVnew{T},T}}()
            @eval scaleinfo{T<:FloatingPoint}(::Type{ImageMagick}, img::AbstractArray{$AC{$CV{T},T}}) = Clamp{$AC{$CVnew{Ufixed8}, Ufixed8}}()
        end
    end
end

# Backwards-compatibility
uint32color(img) = scale(scaleinfo(Uint32, img), img)
uint32color!(buf, img::AbstractArray) = scale!(buf, scaleinfo(Uint32, img), img)
uint32color!(buf, img::AbstractArray, si::ScaleInfo) = scale!(buf, si, img)
uint32color!{T,N}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N}) =
    scale!(buf, scaleinfo(Uint32, img), img)
uint32color!{T,N,N1}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N1}) =
    scale!(buf, scaleinfo(Uint32, img), img, TypeConst{colordim(img)})
uint32color!{T,N}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N}, si::ScaleInfo) =
    scale!(buf, si, img)
uint32color!{T,N,N1}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N1}, si::ScaleInfo) =
    scale!(buf, si, img, TypeConst{colordim(img)})


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
