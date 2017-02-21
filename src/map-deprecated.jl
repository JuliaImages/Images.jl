#### Elementwise manipulations (scaling/clamping/type conversion) ####

# This file exists primarily to handle conversions for display and
# saving to disk. Both of these operations require Normed-valued
# elements, but with display we always want to convert to 8-bit
# whereas saving can handle 16-bit.
# We also can't trust that user images are clamped properly.
# Finally, this supports adjustable contrast limits.

# Structure of MapInfo subtype definitions:
#   - type definition
#   - constructors for scalars
#   - constructors for AbstractArrays
#   - similar  (syntax: similar(mapi, ToType, FromType))
#   - implementation of map() for scalars
#   - implementation of map() for AbstractArrays
# map(mapi::MapInfo{T}, x) should return an object of type T (for x not an array)
# map1(mapi::MapInfo{T}, x) is designed to allow T<:Color to work on
#    scalars x::Fractional

hasparameters{T}(::Type{T}, n) = !isabstract(T) && length(T.parameters) ∈ n

isdefined(:UnionAll) && include_string("""
hasparameters(::UnionAll, n) = false
""")

# Dispatch-based elementwise manipulations
"""
`MapInfo{T}` is an abstract type that encompasses objects designed to
perform intensity or color transformations on pixels.  For example,
before displaying an image in a window, you might need to adjust the
contrast settings; `MapInfo` objects provide a means to describe these
transformations without calculating them immediately.  This delayed
execution can be useful in many contexts.  For example, if you want to
display a movie, it would be quite wasteful to have to first transform
the entire movie; instead, `MapInfo` objects allow one to specify a
transformation to be performed on-the-fly as particular frames are
displayed.

You can create your own custom `MapInfo` objects. For example, given a
grayscale image, you could color "saturated" pixels red using

```jl
immutable ColorSaturated{C<:AbstractRGB} <: MapInfo{C}
end

Base.map{C}(::ColorSaturated{C}, val::Union{Number,Gray}) = ifelse(val == 1, C(1,0,0), C(val,val,val))

imgc = map(ColorSaturated{RGB{N0f8}}(), img)
```

For pre-defined types see `MapNone`, `BitShift`, `ClampMinMax`, `ScaleMinMax`,
`ScaleAutoMinMax`, and `ScaleSigned`.
"""
abstract MapInfo{T}
eltype{T}(mapi::MapInfo{T}) = T

## Centralize calls to map, to reduce the potential for ambiguity
map(mapi::MapInfo, x::Number) = immap(mapi, x)
map(mapi::MapInfo, x::Colorant) = immap(mapi, x)
map(mapi::MapInfo, A::AbstractArray) = immap(mapi, A)

## MapNone
"`MapNone(T)` is a `MapInfo` object that converts `x` to have type `T`."
immutable MapNone{T} <: MapInfo{T}
    function MapNone()
        depwarn("MapNone is deprecated, use x->$T(x)", :MapNone)
        new()
    end
end

# Constructors
MapNone{T}(::Type{T}) = MapNone{T}()
MapNone{T}(val::T) = MapNone{T}()
MapNone{T}(A::AbstractArray{T}) = MapNone{T}()

similar{T}(mapi::MapNone, ::Type{T}, ::Type) = MapNone{T}()

# Implementation
immap{T}(mapi::MapNone{T}, val::Union{Number,Colorant}) = convert(T, val)
map1(mapi::Union{MapNone{RGB24}, MapNone{ARGB32}}, b::Bool) = ifelse(b, 0xffuf8, 0x00uf8)
map1(mapi::Union{MapNone{RGB24},MapNone{ARGB32}}, val::Fractional) = convert(N0f8, val)
map1{CT<:Colorant}(mapi::MapNone{CT}, val::Fractional) = convert(eltype(CT), val)

immap(::MapNone{UInt32}, val::RGB24)  = val.color
immap(::MapNone{UInt32}, val::ARGB32) = val.color
immap(::MapNone{RGB24},  val::UInt32) = reinterpret(RGB24,  val)
immap(::MapNone{ARGB32}, val::UInt32) = reinterpret(ARGB32, val)

# immap{T<:Colorant}(mapi::MapNone{T}, img::AbstractImageIndexed{T}) = convert(Image{T}, img)
# immap{C<:Colorant}(mapi::MapNone{C}, img::AbstractImageDirect{C}) = img  # ambiguity resolution
immap{T}(mapi::MapNone{T}, img::AbstractArray{T}) = img


## BitShift
"""
`BitShift{T,N}` performs a "saturating rightward bit-shift" operation.
It is particularly useful in converting high bit-depth images to 8-bit
images for the purpose of display.  For example,

```
map(BitShift(N0f8, 8), 0xa2d5uf16) === 0xa2uf8
```

converts a `N0f16` to the corresponding `N0f8` by discarding the
least significant byte.  However,

```
map(BitShift(N0f8, 7), 0xa2d5uf16) == 0xffuf8
```

because `0xa2d5>>7 == 0x0145 > typemax(UInt8)`.

When applicable, the main advantage of using `BitShift` rather than
`MapNone` or `ScaleMinMax` is speed.
"""
immutable BitShift{T,N} <: MapInfo{T}
    function BitShift()
        depwarn("BitShift is deprecated, use x->x>>>$N", :BitShift)
        new()
    end
end
BitShift{T}(::Type{T}, n::Int) = BitShift{T,n}()  # note that this is not type-stable

similar{S,T,N}(mapi::BitShift{S,N}, ::Type{T}, ::Type) = BitShift{T,N}()

# Implementation
immutable BS{N} end
_immap{T<:Unsigned,N}(::Type{T}, ::Type{BS{N}}, val::Unsigned) = (v = val>>>N; tm = oftype(val, typemax(T)); convert(T, ifelse(v > tm, tm, v)))
_immap{T<:Normed,N}(::Type{T}, ::Type{BS{N}}, val::Normed) = reinterpret(T, _immap(FixedPointNumbers.rawtype(T), BS{N}, reinterpret(val)))
immap{T<:Real,N}(mapi::BitShift{T,N}, val::Real) = _immap(T, BS{N}, val)
immap{T<:Real,N}(mapi::BitShift{T,N}, val::Gray) = _immap(T, BS{N}, val.val)
immap{T<:Real,N}(mapi::BitShift{Gray{T},N}, val::Gray) = Gray(_immap(T, BS{N}, val.val))
map1{N}(mapi::Union{BitShift{RGB24,N},BitShift{ARGB32,N}}, val::Unsigned) = _immap(UInt8, BS{N}, val)
map1{N}(mapi::Union{BitShift{RGB24,N},BitShift{ARGB32,N}}, val::Normed) = _immap(N0f8, BS{N}, val)
map1{CT<:Colorant,N}(mapi::BitShift{CT,N}, val::Normed) = _immap(eltype(CT), BS{N}, val)


## Clamp types
# The Clamp types just enforce bounds, but do not scale or offset

# Types and constructors
abstract AbstractClamp{T} <: MapInfo{T}
"""
`ClampMin(T, minvalue)` is a `MapInfo` object that clamps pixel values
to be greater than or equal to `minvalue` before converting to type `T`.

See also: `ClampMax`, `ClampMinMax`.
"""
immutable ClampMin{T,From} <: AbstractClamp{T}
    min::From

    function ClampMin(min)
        depwarn("ClampMin is deprecated, use x->max(x, $min)", :ClampMin)
        new(min)
    end
end
ClampMin{T,From}(::Type{T}, min::From) = ClampMin{T,From}(min)
ClampMin{T}(min::T) = ClampMin{T,T}(min)
"""
`ClampMax(T, maxvalue)` is a `MapInfo` object that clamps pixel values
to be less than or equal to `maxvalue` before converting to type `T`.

See also: `ClampMin`, `ClampMinMax`.
"""
immutable ClampMax{T,From} <: AbstractClamp{T}
    max::From

    function ClampMax(max)
        depwarn("ClampMax is deprecated, use x->min(x, $max)", :ClampMax)
        new(max)
    end
end
ClampMax{T,From}(::Type{T}, max::From) = ClampMax{T,From}(max)
ClampMax{T}(max::T) = ClampMax{T,T}(max)
immutable ClampMinMax{T,From} <: AbstractClamp{T}
    min::From
    max::From

    function ClampMinMax(min, max)
        depwarn("ClampMinMax is deprecated, use x->clamp(x, $min, $max)", :ClampMinMax)
        new(min, max)
    end
end
"""
`ClampMinMax(T, minvalue, maxvalue)` is a `MapInfo` object that clamps
pixel values to be between `minvalue` and `maxvalue` before converting
to type `T`.

See also: `ClampMin`, `ClampMax`, and `Clamp`.
"""
ClampMinMax{T,From}(::Type{T}, min::From, max::From) = ClampMinMax{T,From}(min,max)
ClampMinMax{T}(min::T, max::T) = ClampMinMax{T,T}(min,max)
"""
`Clamp(C)` is a `MapInfo` object that clamps color values to be within
gamut.  For example,

```
map(Clamp(RGB{N0f8}), RGB(1.2, -0.4, 0.6)) === RGB{N0f8}(1, 0, 0.6)
```
"""
immutable Clamp{T} <: AbstractClamp{T}
    function Clamp()
        depwarn("Clamp is deprecated, use a colorspace-specific function (clamp01 for gray/RGB)", :Clamp)
        new()
    end
end
Clamp{T}(::Type{T}) = Clamp{T}()

similar{T,F}(mapi::ClampMin, ::Type{T}, ::Type{F}) = ClampMin{T,F}(convert(F, mapi.min))
similar{T,F}(mapi::ClampMax, ::Type{T}, ::Type{F}) = ClampMax{T,F}(convert(F, mapi.max))
similar{T,F}(mapi::ClampMinMax, ::Type{T}, ::Type{F}) = ClampMin{T,F}(convert(F, mapi.min), convert(F, mapi.max))
similar{T,F}(mapi::Clamp, ::Type{T}, ::Type{F}) = Clamp{T}()

# Implementation
immap{T<:Real,F<:Real}(mapi::ClampMin{T,F}, val::F) = convert(T, max(val, mapi.min))
immap{T<:Real,F<:Real}(mapi::ClampMax{T,F}, val::F) = convert(T, min(val, mapi.max))
immap{T<:Real,F<:Real}(mapi::ClampMinMax{T,F}, val::F) = convert(T,min(max(val, mapi.min), mapi.max))
immap{T<:Fractional,F<:Real}(mapi::ClampMin{Gray{T},F}, val::F) = convert(Gray{T}, max(val, mapi.min))
immap{T<:Fractional,F<:Real}(mapi::ClampMax{Gray{T},F}, val::F) = convert(Gray{T}, min(val, mapi.max))
immap{T<:Fractional,F<:Real}(mapi::ClampMinMax{Gray{T},F}, val::F) = convert(Gray{T},min(max(val, mapi.min), mapi.max))
immap{T<:Fractional,F<:Fractional}(mapi::ClampMin{Gray{T},F}, val::Gray{F}) = convert(Gray{T}, max(val, mapi.min))
immap{T<:Fractional,F<:Fractional}(mapi::ClampMax{Gray{T},F}, val::Gray{F}) = convert(Gray{T}, min(val, mapi.max))
immap{T<:Fractional,F<:Fractional}(mapi::ClampMinMax{Gray{T},F}, val::Gray{F}) = convert(Gray{T},min(max(val, mapi.min), mapi.max))
immap{T<:Fractional,F<:Fractional}(mapi::ClampMin{Gray{T},Gray{F}}, val::Gray{F}) = convert(Gray{T}, max(val, mapi.min))
immap{T<:Fractional,F<:Fractional}(mapi::ClampMax{Gray{T},Gray{F}}, val::Gray{F}) = convert(Gray{T}, min(val, mapi.max))
immap{T<:Fractional,F<:Fractional}(mapi::ClampMinMax{Gray{T},Gray{F}}, val::Gray{F}) = convert(Gray{T},min(max(val, mapi.min), mapi.max))
map1{T<:Union{RGB24,ARGB32},F<:Fractional}(mapi::ClampMin{T,F}, val::F) = convert(N0f8, max(val, mapi.min))
map1{T<:Union{RGB24,ARGB32},F<:Fractional}(mapi::ClampMax{T,F}, val::F) = convert(N0f8, min(val, mapi.max))
map1{T<:Union{RGB24,ARGB32},F<:Fractional}(mapi::ClampMinMax{T,F}, val::F) = convert(N0f8,min(max(val, mapi.min), mapi.max))
map1{CT<:Colorant,F<:Fractional}(mapi::ClampMin{CT,F}, val::F) = convert(eltype(CT), max(val, mapi.min))
map1{CT<:Colorant,F<:Fractional}(mapi::ClampMax{CT,F}, val::F) = convert(eltype(CT), min(val, mapi.max))
map1{CT<:Colorant,F<:Fractional}(mapi::ClampMinMax{CT,F}, val::F) = convert(eltype(CT), min(max(val, mapi.min), mapi.max))

immap{To<:Real}(::Clamp{To}, val::Real) = clamp01(To, val)
immap{To<:Real}(::Clamp{Gray{To}}, val::AbstractGray) = Gray(clamp01(To, val.val))
immap{To<:Real}(::Clamp{Gray{To}}, val::Real) = Gray(clamp01(To, val))
map1{CT<:AbstractRGB}(::Clamp{CT}, val::Real) = clamp01(eltype(CT), val)
map1{P<:TransparentRGB}(::Clamp{P}, val::Real) = clamp01(eltype(P), val)

# Also available as a stand-alone function
function ImageCore.clamp01{T}(::Type{T}, x::Real)
    depwarn("clamp01(T, x) is deprecated, use x->T(clamp01(x))", :clamp01)
    T(clamp01(x))
end

# clamp is generic for any colorspace; this version does the right thing for any RGB type
clamp(x::Union{AbstractRGB, TransparentRGB}) = clamp01(x)

## ScaleMinMax
"""
`ScaleMinMax(T, min, max, [scalefactor])` is a `MapInfo` object that
clamps the image at the specified `min`/`max` values, subtracts the
`min` value, scales the result by multiplying by `scalefactor`, and
finally converts to type `T`.  If `scalefactor` is not specified, it
defaults to scaling the interval `[min,max]` to `[0,1]`.

Alternative constructors include `ScaleMinMax(T, img)` for which
`min`, `max`, and `scalefactor` are computed from the minimum and
maximum values found in `img`.

See also: `ScaleMinMaxNaN`, `ScaleAutoMinMax`, `MapNone`, `BitShift`.
"""
immutable ScaleMinMax{To,From,S<:AbstractFloat} <: MapInfo{To}
    min::From
    max::From
    s::S

    function ScaleMinMax(min, max, s)
        depwarn("ScaleMinMax is deprecated, use scaleminmax([$To,] $min, $max)", :ScaleMinMax)
        min >= max && error("min must be smaller than max")
        new(min, max, s)
    end
end

ScaleMinMax{To,From}(::Type{To}, min::From, max::From, s::AbstractFloat) = ScaleMinMax{To,From,typeof(s)}(min, max, s)
ScaleMinMax{To,From}(::Type{To}, min::From, max::From, s) = ScaleMinMax(To, min, max, convert_float(To, Float32, s))
convert_float{To<:AbstractFloat,T}(::Type{To}, ::Type{T}, s) = convert(To, s)
convert_float{To,T}(::Type{To}, ::Type{T}, s) = convert(T, s)
ScaleMinMax{To<:Union{Fractional,Colorant},From}(::Type{To}, mn::From, mx::From) = ScaleMinMax(To, mn, mx, 1.0f0/(convert(Float32, mx)-convert(Float32, mn)))

# ScaleMinMax constructors that take AbstractArray input
ScaleMinMax{To,From<:Real}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(convert(Float32, convert(From, mx))-convert(Float32,convert(From, mn))))
ScaleMinMax{To,From<:Real}(::Type{To}, img::AbstractArray{Gray{From}}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(convert(Float32, convert(From,mx))-convert(Float32, convert(From,mn))))
ScaleMinMax{To,From<:Real,R<:Real}(::Type{To}, img::AbstractArray{From}, mn::Gray{R}, mx::Gray{R}) = ScaleMinMax(To, convert(From,mn.val), convert(From,mx.val), 1.0f0/(convert(Float32, convert(From,mx.val))-convert(Float32, convert(From,mn.val))))
ScaleMinMax{To,From<:Real,R<:Real}(::Type{To}, img::AbstractArray{Gray{From}}, mn::Gray{R}, mx::Gray{R}) = ScaleMinMax(To, convert(From,mn.val), convert(From,mx.val), 1.0f0/(convert(Float32, convert(From,mx.val))-convert(Float32, convert(From,mn.val))))
ScaleMinMax{To}(::Type{To}, img::AbstractArray) = ScaleMinMax(To, img, minfinite(img), maxfinite(img))
ScaleMinMax{To<:Real,CV<:AbstractRGB}(::Type{To}, img::AbstractArray{CV}) = (imgr = channelview(img); ScaleMinMax(To, minfinite(imgr), maxfinite(imgr)))
ScaleMinMax{To<:Colorant,CV<:AbstractRGB}(::Type{To}, img::AbstractArray{CV}) = (imgr = channelview(img); ScaleMinMax(To, minfinite(imgr), maxfinite(imgr)))

similar{T,F,To,From,S}(mapi::ScaleMinMax{To,From,S}, ::Type{T}, ::Type{F}) = ScaleMinMax{T,F,S}(convert(F,mapi.min), convert(F.mapi.max), mapi.s)

# these functions are moved to ImageTransformations
convertsafely{T<:AbstractFloat}(::Type{T}, val) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::Integer) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::AbstractFloat) = round(T, val)
convertsafely{T}(::Type{T}, val) = convert(T, val)

# Implementation
function immap{To<:RealLike,From<:RealLike}(mapi::ScaleMinMax{To,From}, val::Union{Real,Colorant})
    t = clamp(gray(val), gray(mapi.min), gray(mapi.max))
    f = mapi.s*t - mapi.s*mapi.min  # better than mapi.s*(t-mapi.min) (overflow)
    convertsafely(To, f)
end
# function immap{To<:RealLike,From<:RealLike}(mapi::ScaleMinMax{To,From}, val::Union{Real,Colorant})
#     immap(mapi, convert(From, val))
# end
function map1{To<:Union{RGB24,ARGB32},From<:Real}(mapi::ScaleMinMax{To,From}, val::From)
    t = clamp(val, mapi.min, mapi.max)
    f = mapi.s*t - mapi.s*mapi.min
    convert(N0f8, f)
end
function map1{To<:Colorant,From<:Real}(mapi::ScaleMinMax{To,From}, val::From)
    t = clamp(val, mapi.min, mapi.max)
    f = mapi.s*t - mapi.s*mapi.min
    convertsafely(eltype(To), f)
end
function map1{To<:Union{RGB24,ARGB32},From<:Real}(mapi::ScaleMinMax{To,From}, val::Union{Real,Colorant})
    map1(mapi, convert(From, val))
end
function map1{To<:Colorant,From<:Real}(mapi::ScaleMinMax{To,From}, val::Union{Real,Colorant})
    map1(mapi, convert(From, val))
end

## ScaleSigned
"""
`ScaleSigned(T, scalefactor)` is a `MapInfo` object designed for
visualization of images where the pixel's sign has special meaning.
It multiplies the pixel value by `scalefactor`, then clamps to the
interval `[-1,1]`. If `T` is a floating-point type, it stays in this
representation.  If `T` is an `AbstractRGB`, then it is encoded as a
magenta (positive)/green (negative) image, with the intensity of the
color proportional to the clamped absolute value.
"""
immutable ScaleSigned{T, S<:AbstractFloat} <: MapInfo{T}
    s::S

    function ScaleSigned(s)
        depwarn("ScaleSigned is deprecated, use scalesigned", :ScaleSigned)
        new(s)
    end
end
ScaleSigned{T}(::Type{T}, s::AbstractFloat) = ScaleSigned{T, typeof(s)}(s)

ScaleSigned{T}(::Type{T}, img::AbstractArray) = ScaleSigned(T, 1.0f0/maxabsfinite(img))
ScaleSigned(img::AbstractArray) = ScaleSigned(Float32, img)

similar{T,To,S}(mapi::ScaleSigned{To,S}, ::Type{T}, ::Type) = ScaleSigned{T,S}(mapi.s)

immap{T}(mapi::ScaleSigned{T}, val::Real) = convert(T, clamppm(mapi.s*val))
function immap{C<:AbstractRGB}(mapi::ScaleSigned{C}, val::Real)
    x = clamppm(mapi.s*val)
    g = N0f8(abs(x))
    ifelse(x >= 0, C(g, zero(N0f8), g), C(zero(N0f8), g, zero(N0f8)))
end

clamppm(x::Real) = ifelse(x >= 0, min(x, one(x)), max(x, -one(x)))

## ScaleAutoMinMax
# Works only on whole arrays, not values
"""
`ScaleAutoMinMax(T)` constructs a `MapInfo` object that causes images
to be dynamically scaled to their specific min/max values, using the
same algorithm for `ScaleMinMax`. When displaying a movie, the min/max
will be recalculated for each frame, so this can result in
inconsistent contrast scaling.
"""
immutable ScaleAutoMinMax{T} <: MapInfo{T}
    function ScaleAutoMinMax()
        depwarn("ScaleAutoMinMax is deprecated, use scaleminmax as an argument to takemap", :ScaleAutoMinMax)
        new()
    end
end
ScaleAutoMinMax{T}(::Type{T}) = ScaleAutoMinMax{T}()
ScaleAutoMinMax() = ScaleAutoMinMax{N0f8}()

similar{T}(mapi::ScaleAutoMinMax, ::Type{T}, ::Type) = ScaleAutoMinMax{T}()

## NaN-nulling mapping
"""
`ScaleMinMaxNaN(smm)` constructs a `MapInfo` object from a
`ScaleMinMax` object `smm`, with the additional property that `NaN`
values map to zero.

See also: `ScaleMinMax`.
"""
immutable ScaleMinMaxNaN{To,From,S} <: MapInfo{To}
    smm::ScaleMinMax{To,From,S}
    function ScaleMinMaxNaN(smm)
        depwarn("ScaleMinMaxNaN is deprecated, use scaleminmax in conjunction with clamp01nan or x->ifelse(isnan(x), zero(x), x)", :ScaleMinMaxNaN)
        new(smm)
    end
end

"""
`Clamp01NaN(T)` or `Clamp01NaN(img)` constructs a `MapInfo` object
that clamps grayscale or color pixels to the interval `[0,1]`, sending
`NaN` pixels to zero.
"""
immutable Clamp01NaN{T} <: MapInfo{T}
    function Clamp01NaN()
        depwarn("Clamp01NaN is deprecated, use clamp01nan", :Clamp01NaN)
        new()
    end
end

Clamp01NaN{T}(A::AbstractArray{T}) = Clamp01NaN{T}()

# Implementation
similar{T,F,To,From,S}(mapi::ScaleMinMaxNaN{To,From,S}, ::Type{T}, ::Type{F}) = ScaleMinMaxNaN{T,F,S}(similar(mapi.smm, T, F))
similar{T}(mapi::Clamp01NaN, ::Type{T}, ::Type) = Clamp01NaN{T}()

immap{To}(smmn::ScaleMinMaxNaN{To}, g::Number) = isnan(g) ? zero(To) : immap(smmn.smm, g)
immap{To}(smmn::ScaleMinMaxNaN{To}, g::Gray) = isnan(g) ? zero(To) : immap(smmn.smm, g)

function immap{T<:RGB}(::Clamp01NaN{T}, c::AbstractRGB)
    r, g, b = red(c), green(c), blue(c)
    if isnan(r) || isnan(g) || isnan(b)
        return T(0,0,0)
    end
    T(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1))
end
function immap{T<:Union{Fractional,Gray}}(::Clamp01NaN{T}, c::Union{Fractional,AbstractGray})
    g = gray(c)
    if isnan(g)
        return T(0)
    end
    T(clamp(g, 0, 1))
end

# Conversions to RGB{T}, RGBA{T}, RGB24, ARGB32,
# for grayscale, AbstractRGB, and abstract ARGB inputs.
# This essentially "vectorizes" map over a single pixel's color channels using map1
for SI in (MapInfo, AbstractClamp)
    for ST in subtypes(SI)
        isabstract(ST) && continue
        ST == ScaleSigned && continue  # ScaleSigned gives an RGB from a scalar, so don't "vectorize" it
        @eval begin
            # Grayscale and GrayAlpha inputs
            immap(mapi::$ST{RGB24}, g::Gray) = immap(mapi, g.val)
            immap(mapi::$ST{RGB24}, g::Real) = (x = map1(mapi, g); convert(RGB24, RGB{N0f8}(x,x,x)))
            function immap(mapi::$ST{RGB24}, g::AbstractFloat)
                if isfinite(g)
                    x = map1(mapi, g)
                    convert(RGB24, RGB{N0f8}(x,x,x))
                else
                    RGB24(0)
                end
            end
            immap{G<:Gray}(mapi::$ST{RGB24}, g::TransparentColor{G}) = immap(mapi, gray(g))
            immap(mapi::$ST{ARGB32}, g::Gray) = immap(mapi, g.val)
            function immap(mapi::$ST{ARGB32}, g::Real)
                x = map1(mapi, g)
                convert(ARGB32, ARGB{N0f8}(x,x,x,0xffuf8))
            end
            function immap{G<:Gray}(mapi::$ST{ARGB32}, g::TransparentColor{G})
                x = map1(mapi, gray(g))
                convert(ARGB32, ARGB{N0f8}(x,x,x,map1(mapi, g.alpha)))
            end
        end
        for O in (:RGB, :BGR)
            @eval begin
                immap{T}(mapi::$ST{$O{T}}, g::Gray) = immap(mapi, g.val)
                function immap{T}(mapi::$ST{$O{T}}, g::Real)
                    x = map1(mapi, g)
                    $O{T}(x,x,x)
                end
            end
        end
        for OA in (:RGBA, :ARGB, :BGRA)
            exAlphaGray = ST == MapNone ? :nothing : quote
                function immap{T,G<:Gray}(mapi::$ST{$OA{T}}, g::TransparentColor{G})
                    x = map1(mapi, gray(g))
                    $OA{T}(x,x,x,map1(mapi, g.alpha))
                end  # avoids an ambiguity warning with MapNone definitions
            end
            @eval begin
                immap{T}(mapi::$ST{$OA{T}}, g::Gray) = immap(mapi, g.val)
                function immap{T}(mapi::$ST{$OA{T}}, g::Real)
                    x = map1(mapi, g)
                    $OA{T}(x,x,x)
                end
                $exAlphaGray
            end
        end
        @eval begin
            # AbstractRGB and abstract ARGB inputs
            immap(mapi::$ST{RGB24}, rgb::AbstractRGB) =
                convert(RGB24, RGB{N0f8}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb))))
            immap{C<:AbstractRGB, TC}(mapi::$ST{RGB24}, argb::TransparentColor{C,TC}) =
                convert(RGB24, RGB{N0f8}(map1(mapi, red(argb)), map1(mapi, green(argb)),
                                            map1(mapi, blue(argb))))
            immap{C<:AbstractRGB, TC}(mapi::$ST{ARGB32}, argb::TransparentColor{C,TC}) =
                convert(ARGB32, ARGB{N0f8}(map1(mapi, red(argb)), map1(mapi, green(argb)),
                                              map1(mapi, blue(argb)), map1(mapi, alpha(argb))))
            immap(mapi::$ST{ARGB32}, rgb::AbstractRGB) =
                convert(ARGB32, ARGB{N0f8}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb))))
        end
        for O in (:RGB, :BGR)
            @eval begin
                immap{T}(mapi::$ST{$O{T}}, rgb::AbstractRGB) =
                    $O{T}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb)))
                immap{T,C<:AbstractRGB, TC}(mapi::$ST{$O{T}}, argb::TransparentColor{C,TC}) =
                    $O{T}(map1(mapi, red(argb)), map1(mapi, green(argb)), map1(mapi, blue(argb)))
            end
        end
        for OA in (:RGBA, :ARGB, :BGRA)
            @eval begin
                immap{T, C<:AbstractRGB, TC}(mapi::$ST{$OA{T}}, argb::TransparentColor{C,TC}) =
                    $OA{T}(map1(mapi, red(argb)), map1(mapi, green(argb)),
                            map1(mapi, blue(argb)), map1(mapi, alpha(argb)))
                immap{T}(mapi::$ST{$OA{T}}, argb::ARGB32) = immap(mapi, convert(RGBA{N0f8}, argb))
                immap{T}(mapi::$ST{$OA{T}}, rgb::AbstractRGB) =
                    $OA{T}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb)))
                immap{T}(mapi::$ST{$OA{T}}, rgb::RGB24) = immap(mapi, convert(RGB{N0f8}, argb))
            end
        end
    end
end

## Fallback definitions of map() for array types

function immap{T}(mapi::MapInfo{T}, img::AbstractArray)
    out = similar(img, T)
    map!(mapi, out, img)
end

# immap{C<:Colorant,R<:Real}(mapi::MapNone{C}, img::AbstractImageDirect{R}) = mapcd(mapi, img)  # ambiguity resolution
# immap{C<:Colorant,R<:Real}(mapi::MapInfo{C}, img::AbstractImageDirect{R}) = mapcd(mapi, img)
# function mapcd{C<:Colorant,R<:Real}(mapi::MapInfo{C}, img::AbstractImageDirect{R})
#     # For this case we have to check whether color is defined along an array axis
#     cd = colordim(img)
#     if cd > 0
#         dims = setdiff(1:ndims(img), cd)
#         out = similar(img, C, size(img)[dims])
#         map!(mapi, out, img, Val{cd})
#     else
#         out = similar(img, C)
#         map!(mapi, out, img)
#     end
#     out   # note this isn't type-stable
# end

# function immap{T<:Colorant}(mapi::MapInfo{T}, img::AbstractImageIndexed)
#     out = Image(Array(T, size(img)), properties(img))
#     map!(mapi, out, img)
# end

map!{T,T1,T2,N}(mapi::MapInfo{T1}, out::AbstractArray{T,N}, img::AbstractArray{T2,N}) =
    _map_a!(mapi, out, img)
function _map_a!{T,T1,T2,N}(mapi::MapInfo{T1}, out::AbstractArray{T,N}, img::AbstractArray{T2,N})
    mi = take(mapi, img)
    dimg = data(img)
    dout = data(out)
    size(dout) == size(dimg) || throw(DimensionMismatch())
    if eltype(dout) == UInt32 && isa(immap(mi, first(dimg)), Union{RGB24,ARGB32})
        for I in eachindex(dout, dimg)
            @inbounds dout[I] = immap(mi, dimg[I]).color
        end
    else
        for I in eachindex(dout, dimg)
            @inbounds dout[I] = immap(mi, dimg[I])
        end
    end
    out
end

take(mapi::MapInfo, img::AbstractArray) = mapi
take{T}(mapi::ScaleAutoMinMax{T}, img::AbstractArray) = ScaleMinMax(T, img)

# Indexed images (colormaps)
map!{T,T1,N}(mapi::MapInfo{T}, out::AbstractArray{T,N}, img::IndirectArray{T1,N}) =
    _mapindx!(mapi, out, img)
function _mapindx!{T,T1,N}(mapi::MapInfo{T}, out::AbstractArray{T,N}, img::IndirectArray{T1,N})
    dimg = data(img)
    dout = data(out)
    colmap = immap(mapi, dimg.values)
    for I in eachindex(dout, dimg)
        @inbounds dout[I] = colmap[dimg.index[I]]
    end
    out
end


#### MapInfo defaults
# Each "client" can define its own methods. "clients" include Normed,
# RGB24/ARGB32, and ImageMagick

const bitshiftto8 = ((N6f10, 2), (N4f12, 4), (N2f14, 6), (N0f16, 8))

# typealias GrayType{T<:Fractional} Union{T, Gray{T}}
typealias GrayArray{T<:Union{Fractional,Bool}} Union{AbstractArray{T}, AbstractArray{Gray{T}}}
# note, though, that we need to override for AbstractImage in case the
# "colorspace" property is defined differently

# mapinfo{T<:Union{Real,Colorant}}(::Type{T}, img::AbstractArray{T}) = MapNone(img)
"""
`mapi = mapinf(T, img)` returns a `MapInfo` object that is deemed
appropriate for converting pixels of `img` to be of type `T`. `T` can
either be a specific type (e.g., `RGB24`), or you can specify an
abstract type like `Clamp` and it will return one of the `Clamp`
family of `MapInfo` objects.

You can define your own rules for `mapinfo`.  For example, the
`ImageMagick` package defines methods for how pixels values should be
converted before saving images to disk.
"""
mapinfo{T<:Normed}(::Type{T}, img::AbstractArray{T}) = MapNone(img)
mapinfo{T<:AbstractFloat}(::Type{T}, img::AbstractArray{T}) = MapNone(img)

# Grayscale methods
mapinfo(::Type{N0f8}, img::GrayArray{Bool}) = MapNone{N0f8}()
mapinfo(::Type{N0f8}, img::GrayArray{N0f8}) = MapNone{N0f8}()
mapinfo(::Type{Gray{N0f8}}, img::GrayArray{N0f8}) = MapNone{Gray{N0f8}}()
mapinfo(::Type{GrayA{N0f8}}, img::AbstractArray{GrayA{N0f8}}) = MapNone{GrayA{N0f8}}()
for (T,n) in bitshiftto8
    @eval mapinfo(::Type{N0f8}, img::GrayArray{$T}) = BitShift{N0f8,$n}()
    @eval mapinfo(::Type{Gray{N0f8}}, img::GrayArray{$T}) = BitShift{Gray{N0f8},$n}()
    @eval mapinfo(::Type{GrayA{N0f8}}, img::AbstractArray{GrayA{$T}}) = BitShift{GrayA{N0f8},$n}()
end
mapinfo{T<:Normed,F<:AbstractFloat}(::Type{T}, img::GrayArray{F}) = ClampMinMax(T, zero(F), one(F))
mapinfo{T<:Normed,F<:AbstractFloat}(::Type{Gray{T}}, img::GrayArray{F}) = ClampMinMax(Gray{T}, zero(F), one(F))
mapinfo{T<:AbstractFloat, R<:Real}(::Type{T}, img::AbstractArray{R}) = MapNone(T)

mapinfo(::Type{RGB24}, img::Union{AbstractArray{Bool}, BitArray}) = MapNone{RGB24}()
mapinfo(::Type{ARGB32}, img::Union{AbstractArray{Bool}, BitArray}) = MapNone{ARGB32}()
mapinfo{F<:Fractional}(::Type{RGB24}, img::GrayArray{F}) = ClampMinMax(RGB24, zero(F), one(F))
mapinfo{F<:Fractional}(::Type{ARGB32}, img::AbstractArray{F}) = ClampMinMax(ARGB32, zero(F), one(F))

# Color->Color methods
mapinfo(::Type{RGB{N0f8}}, img) = MapNone{RGB{N0f8}}()
mapinfo(::Type{RGBA{N0f8}}, img) = MapNone{RGBA{N0f8}}()
for (T,n) in bitshiftto8
    @eval mapinfo(::Type{RGB{N0f8}}, img::AbstractArray{RGB{$T}}) = BitShift{RGB{N0f8},$n}()
    @eval mapinfo(::Type{RGBA{N0f8}}, img::AbstractArray{RGBA{$T}}) = BitShift{RGBA{N0f8},$n}()
end
mapinfo{F<:Fractional}(::Type{RGB{N0f8}}, img::AbstractArray{RGB{F}}) = Clamp(RGB{N0f8})
mapinfo{F<:Fractional}(::Type{RGBA{N0f8}}, img::AbstractArray{RGBA{F}}) = Clamp(RGBA{N0f8})



# Color->RGB24/ARGB32
mapinfo(::Type{RGB24}, img::AbstractArray{RGB24}) = MapNone{RGB24}()
mapinfo(::Type{ARGB32}, img::AbstractArray{ARGB32}) = MapNone{ARGB32}()
for C in tuple(subtypes(AbstractRGB)..., Gray)
    C == RGB24 && continue
    @eval mapinfo(::Type{RGB24}, img::AbstractArray{$C{N0f8}}) = MapNone{RGB24}()
    @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$C{N0f8}}) = MapNone{ARGB32}()
    for (T, n) in bitshiftto8
        @eval mapinfo(::Type{RGB24}, img::AbstractArray{$C{$T}}) = BitShift{RGB24, $n}()
        @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$C{$T}}) = BitShift{ARGB32, $n}()
    end
    @eval mapinfo{F<:AbstractFloat}(::Type{RGB24}, img::AbstractArray{$C{F}}) = ClampMinMax(RGB24, zero(F), one(F))
    @eval mapinfo{F<:AbstractFloat}(::Type{ARGB32}, img::AbstractArray{$C{F}}) = ClampMinMax(ARGB32, zero(F), one(F))
    for AC in subtypes(TransparentColor)
        hasparameters(AC, 2) || continue
        @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$AC{$C{N0f8},N0f8}}) = MapNone{ARGB32}()
        @eval mapinfo(::Type{RGB24}, img::AbstractArray{$AC{$C{N0f8},N0f8}}) = MapNone{RGB24}()
        for (T, n) in bitshiftto8
            @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$AC{$C{$T},$T}}) = BitShift{ARGB32, $n}()
            @eval mapinfo(::Type{RGB24}, img::AbstractArray{$AC{$C{$T},$T}}) = BitShift{RGB24, $n}()
        end
        @eval mapinfo{F<:AbstractFloat}(::Type{ARGB32}, img::AbstractArray{$AC{$C{F},F}}) = ClampMinMax(ARGB32, zero(F), one(F))
        @eval mapinfo{F<:AbstractFloat}(::Type{RGB24}, img::AbstractArray{$AC{$C{F},F}}) = ClampMinMax(RGB24, zero(F), one(F))
    end
end

mapinfo{CT<:Colorant}(::Type{RGB24},  img::AbstractArray{CT}) = MapNone{RGB24}()
mapinfo{CT<:Colorant}(::Type{ARGB32}, img::AbstractArray{CT}) = MapNone{ARGB32}()


# UInt32 conversions will use ARGB32 for images that have an alpha channel,
# and RGB24 when not
mapinfo{CV<:Union{Fractional,Color,AbstractGray}}(::Type{UInt32}, img::AbstractArray{CV}) = mapinfo(RGB24, img)
mapinfo{CV<:TransparentColor}(::Type{UInt32}, img::AbstractArray{CV}) = mapinfo(ARGB32, img)
mapinfo(::Type{UInt32}, img::Union{AbstractArray{Bool},BitArray}) = mapinfo(RGB24, img)
mapinfo(::Type{UInt32}, img::AbstractArray{UInt32}) = MapNone{UInt32}()


# Clamping mapinfo client. Converts to RGB and uses Normed, clamping
# floating-point values to [0,1].
mapinfo{T<:Normed}(::Type{Clamp}, img::AbstractArray{T}) = MapNone{T}()
mapinfo{T<:AbstractFloat}(::Type{Clamp}, img::AbstractArray{T}) = ClampMinMax(N0f8, zero(T), one(T))
let handled = Set()
for ACV in (Color, AbstractRGB)
    for CV in subtypes(ACV)
        hasparameters(CV, 1) || continue
        CVnew = CV<:AbstractGray ? Gray : RGB
        @eval mapinfo{T<:Normed}(::Type{Clamp}, img::AbstractArray{$CV{T}}) = MapNone{$CVnew{T}}()
        @eval mapinfo{CV<:$CV}(::Type{Clamp}, img::AbstractArray{CV}) = Clamp{$CVnew{N0f8}}()
        CVnew = CV<:AbstractGray ? Gray : BGR
        AC, CA       = alphacolor(CV), coloralpha(CV)
        if AC in handled
            continue
        end
        push!(handled, AC)
        ACnew, CAnew = alphacolor(CVnew), coloralpha(CVnew)
        @eval begin
            mapinfo{T<:Normed}(::Type{Clamp}, img::AbstractArray{$AC{T}}) = MapNone{$ACnew{T}}()
            mapinfo{P<:$AC}(::Type{Clamp}, img::AbstractArray{P}) = Clamp{$ACnew{N0f8}}()
            mapinfo{T<:Normed}(::Type{Clamp}, img::AbstractArray{$CA{T}}) = MapNone{$CAnew{T}}()
            mapinfo{P<:$CA}(::Type{Clamp}, img::AbstractArray{P}) = Clamp{$CAnew{N0f8}}()
        end
    end
end
end
mapinfo(::Type{Clamp}, img::AbstractArray{RGB24}) = MapNone{RGB{N0f8}}()
mapinfo(::Type{Clamp}, img::AbstractArray{ARGB32}) = MapNone{BGRA{N0f8}}()


"""
```
imgsc = sc(img)
imgsc = sc(img, min, max)
```

Applies default or specified `ScaleMinMax` mapping to the image.
"""
sc(img::AbstractArray) = immap(ScaleMinMax(N0f8, img), img)
sc(img::AbstractArray, mn::Real, mx::Real) = immap(ScaleMinMax(N0f8, img, mn, mx), img)

ufixedsc{T<:Normed}(::Type{T}, img::AbstractArray) = immap(mapinfo(T, img), img)
ufixed8sc(img::AbstractArray) = ufixedsc(N0f8, img)
