#### Elementwise manipulations (scaling/clamping/type conversion) ####

# This file exists primarily to handle conversions for display and
# saving to disk. Both of these operations require UFixed-valued
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

imgc = map(ColorSaturated{RGB{U8}}(), img)
```

For pre-defined types see `MapNone`, `BitShift`, `ClampMinMax`, `ScaleMinMax`,
`ScaleAutoMinMax`, and `ScaleSigned`.
"""
abstract MapInfo{T}
eltype{T}(mapi::MapInfo{T}) = T


## MapNone
"`MapNone(T)` is a `MapInfo` object that converts `x` to have type `T`."
immutable MapNone{T} <: MapInfo{T}; end

# Constructors
MapNone{T}(::Type{T}) = MapNone{T}()
MapNone{T}(val::T) = MapNone{T}()
MapNone{T}(A::AbstractArray{T}) = MapNone{T}()

similar{T}(mapi::MapNone, ::Type{T}, ::Type) = MapNone{T}()

# Implementation
map{T}(mapi::MapNone{T}, val::Union{Number,Colorant}) = convert(T, val)
map1(mapi::Union{MapNone{RGB24}, MapNone{ARGB32}}, b::Bool) = ifelse(b, 0xffuf8, 0x00uf8)
map1(mapi::Union{MapNone{RGB24},MapNone{ARGB32}}, val::Fractional) = convert(UFixed8, val)
map1{CT<:Colorant}(mapi::MapNone{CT}, val::Fractional) = convert(eltype(CT), val)

map{T<:Colorant}(mapi::MapNone{T}, img::AbstractImageIndexed{T}) = convert(Image{T}, img)
map{C<:Colorant}(mapi::MapNone{C}, img::AbstractImageDirect{C}) = img  # ambiguity resolution
map{T}(mapi::MapNone{T}, img::AbstractArray{T}) = img


## BitShift
"""
`BitShift{T,N}` performs a "saturating rightward bit-shift" operation.
It is particularly useful in converting high bit-depth images to 8-bit
images for the purpose of display.  For example,

```
map(BitShift(UFixed8, 8), 0xa2d5uf16) === 0xa2uf8
```

converts a `UFixed16` to the corresponding `UFixed8` by discarding the
least significant byte.  However,

```
map(BitShift(UFixed8, 7), 0xa2d5uf16) == 0xffuf8
```

because `0xa2d5>>7 == 0x0145 > typemax(UInt8)`.

When applicable, the main advantage of using `BitShift` rather than
`MapNone` or `ScaleMinMax` is speed.
"""
immutable BitShift{T,N} <: MapInfo{T} end
BitShift{T}(::Type{T}, n::Int) = BitShift{T,n}()  # note that this is not type-stable

similar{S,T,N}(mapi::BitShift{S,N}, ::Type{T}, ::Type) = BitShift{T,N}()

# Implementation
immutable BS{N} end
_map{T<:Unsigned,N}(::Type{T}, ::Type{BS{N}}, val::Unsigned) = (v = val>>>N; tm = oftype(val, typemax(T)); convert(T, ifelse(v > tm, tm, v)))
_map{T<:UFixed,N}(::Type{T}, ::Type{BS{N}}, val::UFixed) = reinterpret(T, _map(FixedPointNumbers.rawtype(T), BS{N}, reinterpret(val)))
map{T<:Real,N}(mapi::BitShift{T,N}, val::Real) = _map(T, BS{N}, val)
map{T<:Real,N}(mapi::BitShift{Gray{T},N}, val::Gray) = Gray(_map(T, BS{N}, val.val))
map1{N}(mapi::Union{BitShift{RGB24,N},BitShift{ARGB32,N}}, val::Unsigned) = _map(UInt8, BS{N}, val)
map1{N}(mapi::Union{BitShift{RGB24,N},BitShift{ARGB32,N}}, val::UFixed) = _map(UFixed8, BS{N}, val)
map1{CT<:Colorant,N}(mapi::BitShift{CT,N}, val::UFixed) = _map(eltype(CT), BS{N}, val)


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
end
ClampMax{T,From}(::Type{T}, max::From) = ClampMax{T,From}(max)
ClampMax{T}(max::T) = ClampMax{T,T}(max)
immutable ClampMinMax{T,From} <: AbstractClamp{T}
    min::From
    max::From
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
map(Clamp(RGB{U8}), RGB(1.2, -0.4, 0.6)) === RGB{U8}(1, 0, 0.6)
```
"""
immutable Clamp{T} <: AbstractClamp{T} end
Clamp{T}(::Type{T}) = Clamp{T}()

similar{T,F}(mapi::ClampMin, ::Type{T}, ::Type{F}) = ClampMin{T,F}(convert(F, mapi.min))
similar{T,F}(mapi::ClampMax, ::Type{T}, ::Type{F}) = ClampMax{T,F}(convert(F, mapi.max))
similar{T,F}(mapi::ClampMinMax, ::Type{T}, ::Type{F}) = ClampMin{T,F}(convert(F, mapi.min), convert(F, mapi.max))
similar{T,F}(mapi::Clamp, ::Type{T}, ::Type{F}) = Clamp{T}()

# Implementation
map{T<:Real,F<:Real}(mapi::ClampMin{T,F}, val::F) = convert(T, max(val, mapi.min))
map{T<:Real,F<:Real}(mapi::ClampMax{T,F}, val::F) = convert(T, min(val, mapi.max))
map{T<:Real,F<:Real}(mapi::ClampMinMax{T,F}, val::F) = convert(T,min(max(val, mapi.min), mapi.max))
map{T<:Fractional,F<:Real}(mapi::ClampMin{Gray{T},F}, val::F) = convert(Gray{T}, max(val, mapi.min))
map{T<:Fractional,F<:Real}(mapi::ClampMax{Gray{T},F}, val::F) = convert(Gray{T}, min(val, mapi.max))
map{T<:Fractional,F<:Real}(mapi::ClampMinMax{Gray{T},F}, val::F) = convert(Gray{T},min(max(val, mapi.min), mapi.max))
map{T<:Fractional,F<:Fractional}(mapi::ClampMin{Gray{T},F}, val::Gray{F}) = convert(Gray{T}, max(val, mapi.min))
map{T<:Fractional,F<:Fractional}(mapi::ClampMax{Gray{T},F}, val::Gray{F}) = convert(Gray{T}, min(val, mapi.max))
map{T<:Fractional,F<:Fractional}(mapi::ClampMinMax{Gray{T},F}, val::Gray{F}) = convert(Gray{T},min(max(val, mapi.min), mapi.max))
map{T<:Fractional,F<:Fractional}(mapi::ClampMin{Gray{T},Gray{F}}, val::Gray{F}) = convert(Gray{T}, max(val, mapi.min))
map{T<:Fractional,F<:Fractional}(mapi::ClampMax{Gray{T},Gray{F}}, val::Gray{F}) = convert(Gray{T}, min(val, mapi.max))
map{T<:Fractional,F<:Fractional}(mapi::ClampMinMax{Gray{T},Gray{F}}, val::Gray{F}) = convert(Gray{T},min(max(val, mapi.min), mapi.max))
map1{T<:Union{RGB24,ARGB32},F<:Fractional}(mapi::ClampMin{T,F}, val::F) = convert(UFixed8, max(val, mapi.min))
map1{T<:Union{RGB24,ARGB32},F<:Fractional}(mapi::ClampMax{T,F}, val::F) = convert(UFixed8, min(val, mapi.max))
map1{T<:Union{RGB24,ARGB32},F<:Fractional}(mapi::ClampMinMax{T,F}, val::F) = convert(UFixed8,min(max(val, mapi.min), mapi.max))
map1{CT<:Colorant,F<:Fractional}(mapi::ClampMin{CT,F}, val::F) = convert(eltype(CT), max(val, mapi.min))
map1{CT<:Colorant,F<:Fractional}(mapi::ClampMax{CT,F}, val::F) = convert(eltype(CT), min(val, mapi.max))
map1{CT<:Colorant,F<:Fractional}(mapi::ClampMinMax{CT,F}, val::F) = convert(eltype(CT), min(max(val, mapi.min), mapi.max))

map{To<:Real}(::Clamp{To}, val::Real) = clamp01(To, val)
map{To<:Real}(::Clamp{Gray{To}}, val::AbstractGray) = Gray(clamp01(To, val.val))
map{To<:Real}(::Clamp{Gray{To}}, val::Real) = Gray(clamp01(To, val))
map1{CT<:AbstractRGB}(::Clamp{CT}, val::Real) = clamp01(eltype(CT), val)
map1{P<:TransparentRGB}(::Clamp{P}, val::Real) = clamp01(eltype(P), val)

# Also available as a stand-alone function
clamp01{T}(::Type{T}, x::Real) = convert(T, min(max(x, zero(x)), one(x)))
clamp01(x::Real) = clamp01(typeof(x), x)
clamp01(x::Colorant) = clamp01(typeof(x), x)
clamp01{Cdest<:AbstractRGB   }(::Type{Cdest}, x::AbstractRGB)    = (To = eltype(Cdest);
    Cdest(clamp01(To, red(x)), clamp01(To, green(x)), clamp01(To, blue(x))))
clamp01{Pdest<:TransparentRGB}(::Type{Pdest}, x::TransparentRGB) = (To = eltype(Pdest);
    Pdest(clamp01(To, red(x)), clamp01(To, green(x)), clamp01(To, blue(x)), clamp01(To, alpha(x))))

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
        min >= max && error("min must be smaller than max")
        new(min, max, s)
    end
end

ScaleMinMax{To,From}(::Type{To}, min::From, max::From, s::AbstractFloat) = ScaleMinMax{To,From,typeof(s)}(min, max, s)
ScaleMinMax{To<:Union{Fractional,Colorant},From}(::Type{To}, mn::From, mx::From) = ScaleMinMax(To, mn, mx, 1.0f0/(convert(Float32, mx)-convert(Float32, mn)))

# ScaleMinMax constructors that take AbstractArray input
ScaleMinMax{To,From<:Real}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(convert(Float32, convert(From, mx))-convert(Float32,convert(From, mn))))
ScaleMinMax{To,From<:Real}(::Type{To}, img::AbstractArray{Gray{From}}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(convert(Float32, convert(From,mx))-convert(Float32, convert(From,mn))))
ScaleMinMax{To,From<:Real,R<:Real}(::Type{To}, img::AbstractArray{From}, mn::Gray{R}, mx::Gray{R}) = ScaleMinMax(To, convert(From,mn.val), convert(From,mx.val), 1.0f0/(convert(Float32, convert(From,mx.val))-convert(Float32, convert(From,mn.val))))
ScaleMinMax{To,From<:Real,R<:Real}(::Type{To}, img::AbstractArray{Gray{From}}, mn::Gray{R}, mx::Gray{R}) = ScaleMinMax(To, convert(From,mn.val), convert(From,mx.val), 1.0f0/(convert(Float32, convert(From,mx.val))-convert(Float32, convert(From,mn.val))))
ScaleMinMax{To}(::Type{To}, img::AbstractArray) = ScaleMinMax(To, img, minfinite(img), maxfinite(img))
ScaleMinMax{To,CV<:AbstractRGB}(::Type{To}, img::AbstractArray{CV}) = (imgr = reinterpret(eltype(CV), img); ScaleMinMax(To, minfinite(imgr), maxfinite(imgr)))

similar{T,F,To,From,S}(mapi::ScaleMinMax{To,From,S}, ::Type{T}, ::Type{F}) = ScaleMinMax{T,F,S}(convert(F,mapi.min), convert(F.mapi.max), mapi.s)

# Implementation
function map{To<:Union{Real,AbstractGray},From<:Union{Real,AbstractGray}}(mapi::ScaleMinMax{To,From}, val::From)
    g = gray(val)
    t = ifelse(g < mapi.min, zero(From), ifelse(g > mapi.max, mapi.max-mapi.min, g-mapi.min))
    convert(To, mapi.s*t)
end
function map{To<:Union{Real,AbstractGray},From<:Union{Real,AbstractGray}}(mapi::ScaleMinMax{To,From}, val::Union{Real,Colorant})
    map(mapi, convert(From, val))
end
function map1{To<:Union{RGB24,ARGB32},From<:Real}(mapi::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < mapi.min, zero(From), ifelse(val > mapi.max, mapi.max-mapi.min, val-mapi.min))
    convert(UFixed8, mapi.s*t)
end
function map1{To<:Colorant,From<:Real}(mapi::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < mapi.min, zero(From), ifelse(val > mapi.max, mapi.max-mapi.min, val-mapi.min))
    convert(eltype(To), mapi.s*t)
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
end
ScaleSigned{T}(::Type{T}, s::AbstractFloat) = ScaleSigned{T, typeof(s)}(s)

ScaleSigned{T}(::Type{T}, img::AbstractArray) = ScaleSigned(T, 1.0f0/maxabsfinite(img))
ScaleSigned(img::AbstractArray) = ScaleSigned(Float32, img)

similar{T,To,S}(mapi::ScaleSigned{To,S}, ::Type{T}, ::Type) = ScaleSigned{T,S}(mapi.s)

map{T}(mapi::ScaleSigned{T}, val::Real) = convert(T, clamppm(mapi.s*val))
function map{C<:AbstractRGB}(mapi::ScaleSigned{C}, val::Real)
    x = clamppm(mapi.s*val)
    g = UFixed8(abs(x))
    ifelse(x >= 0, C(g, zero(UFixed8), g), C(zero(UFixed8), g, zero(UFixed8)))
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
immutable ScaleAutoMinMax{T} <: MapInfo{T} end
ScaleAutoMinMax{T}(::Type{T}) = ScaleAutoMinMax{T}()
ScaleAutoMinMax() = ScaleAutoMinMax{UFixed8}()

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
end

"""
`Clamp01NaN(T)` or `Clamp01NaN(img)` constructs a `MapInfo` object
that clamps grayscale or color pixels to the interval `[0,1]`, sending
`NaN` pixels to zero.
"""
immutable Clamp01NaN{T} <: MapInfo{T} end

Clamp01NaN{T}(A::AbstractArray{T}) = Clamp01NaN{T}()

# Implementation
similar{T,F,To,From,S}(mapi::ScaleMinMaxNaN{To,From,S}, ::Type{T}, ::Type{F}) = ScaleMinMaxNaN{T,F,S}(similar(mapi.smm, T, F))
similar{T}(mapi::Clamp01NaN, ::Type{T}, ::Type) = Clamp01NaN{T}()

Base.map{To}(smmn::ScaleMinMaxNaN{To}, g::Number) = isnan(g) ? zero(To) : map(smmn.smm, g)
Base.map{To}(smmn::ScaleMinMaxNaN{To}, g::Gray) = isnan(g) ? zero(To) : map(smmn.smm, g)

function Base.map{T<:RGB}(::Clamp01NaN{T}, c::AbstractRGB)
    r, g, b = red(c), green(c), blue(c)
    if isnan(r) || isnan(g) || isnan(b)
        return T(0,0,0)
    end
    T(clamp(r, 0, 1), clamp(g, 0, 1), clamp(b, 0, 1))
end
function Base.map{T<:Union{Fractional,Gray}}(::Clamp01NaN{T}, c::Union{Fractional,AbstractGray})
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
        ST.abstract && continue
        ST == ScaleSigned && continue  # ScaleSigned gives an RGB from a scalar, so don't "vectorize" it
        @eval begin
            # Grayscale and GrayAlpha inputs
            map(mapi::$ST{RGB24}, g::Gray) = map(mapi, g.val)
            map(mapi::$ST{RGB24}, g::Real) = (x = map1(mapi, g); convert(RGB24, RGB{UFixed8}(x,x,x)))
            function map(mapi::$ST{RGB24}, g::AbstractFloat)
                if isfinite(g)
                    x = map1(mapi, g)
                    convert(RGB24, RGB{UFixed8}(x,x,x))
                else
                    RGB24(0)
                end
            end
            map{G<:Gray}(mapi::$ST{RGB24}, g::TransparentColor{G}) = map(mapi, gray(g))
            map(mapi::$ST{ARGB32}, g::Gray) = map(mapi, g.val)
            function map(mapi::$ST{ARGB32}, g::Real)
                x = map1(mapi, g)
                convert(ARGB32, ARGB{UFixed8}(x,x,x,0xffuf8))
            end
            function map{G<:Gray}(mapi::$ST{ARGB32}, g::TransparentColor{G})
                x = map1(mapi, gray(g))
                convert(ARGB32, ARGB{UFixed8}(x,x,x,map1(mapi, g.alpha)))
            end
        end
        for O in (:RGB, :BGR)
            @eval begin
                map{T}(mapi::$ST{$O{T}}, g::Gray) = map(mapi, g.val)
                function map{T}(mapi::$ST{$O{T}}, g::Real)
                    x = map1(mapi, g)
                    $O{T}(x,x,x)
                end
            end
        end
        for OA in (:RGBA, :ARGB, :BGRA)
            exAlphaGray = ST == MapNone ? :nothing : quote
                function map{T,G<:Gray}(mapi::$ST{$OA{T}}, g::TransparentColor{G})
                    x = map1(mapi, gray(g))
                    $OA{T}(x,x,x,map1(mapi, g.alpha))
                end  # avoids an ambiguity warning with MapNone definitions
            end
            @eval begin
                map{T}(mapi::$ST{$OA{T}}, g::Gray) = map(mapi, g.val)
                function map{T}(mapi::$ST{$OA{T}}, g::Real)
                    x = map1(mapi, g)
                    $OA{T}(x,x,x)
                end
                $exAlphaGray
            end
        end
        @eval begin
            # AbstractRGB and abstract ARGB inputs
            map(mapi::$ST{RGB24}, rgb::AbstractRGB) =
                convert(RGB24, RGB{UFixed8}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb))))
            map{C<:AbstractRGB, TC}(mapi::$ST{RGB24}, argb::TransparentColor{C,TC}) =
                convert(RGB24, RGB{UFixed8}(map1(mapi, red(argb)), map1(mapi, green(argb)),
                                            map1(mapi, blue(argb))))
            map{C<:AbstractRGB, TC}(mapi::$ST{ARGB32}, argb::TransparentColor{C,TC}) =
                convert(ARGB32, ARGB{UFixed8}(map1(mapi, red(argb)), map1(mapi, green(argb)),
                                              map1(mapi, blue(argb)), map1(mapi, alpha(argb))))
            map(mapi::$ST{ARGB32}, rgb::AbstractRGB) =
                convert(ARGB32, ARGB{UFixed8}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb))))
        end
        for O in (:RGB, :BGR)
            @eval begin
                map{T}(mapi::$ST{$O{T}}, rgb::AbstractRGB) =
                    $O{T}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb)))
                map{T,C<:AbstractRGB, TC}(mapi::$ST{$O{T}}, argb::TransparentColor{C,TC}) =
                    $O{T}(map1(mapi, red(argb)), map1(mapi, green(argb)), map1(mapi, blue(argb)))
            end
        end
        for OA in (:RGBA, :ARGB, :BGRA)
            @eval begin
                map{T, C<:AbstractRGB, TC}(mapi::$ST{$OA{T}}, argb::TransparentColor{C,TC}) =
                    $OA{T}(map1(mapi, red(argb)), map1(mapi, green(argb)),
                            map1(mapi, blue(argb)), map1(mapi, alpha(argb)))
                map{T}(mapi::$ST{$OA{T}}, argb::ARGB32) = map(mapi, convert(RGBA{UFixed8}, argb))
                map{T}(mapi::$ST{$OA{T}}, rgb::AbstractRGB) =
                    $OA{T}(map1(mapi, red(rgb)), map1(mapi, green(rgb)), map1(mapi, blue(rgb)))
                map{T}(mapi::$ST{$OA{T}}, rgb::RGB24) = map(mapi, convert(RGB{UFixed8}, argb))
            end
        end
    end
end

# # Apply to any Colorant
# map(f::Callable, x::Color) = f(x)
# map(mapi, x::Color) = map(mapi, convert(RGB, x))
# map{C<:Color, TC}(f::Callable, x::TransparentColor{C, TC}) = f(convert(ARGB, x))
# map{C<:Color, TC}(mapi, x::TransparentColor{C, TC}) = map(mapi, convert(ARGB, x))

## Fallback definitions of map() for array types

function map{T}(mapi::MapInfo{T}, img::AbstractArray)
    out = similar(img, T)
    map!(mapi, out, img)
end

map{C<:Colorant,R<:Real}(mapi::MapNone{C}, img::AbstractImageDirect{R}) = mapcd(mapi, img)  # ambiguity resolution
map{C<:Colorant,R<:Real}(mapi::MapInfo{C}, img::AbstractImageDirect{R}) = mapcd(mapi, img)
function mapcd{C<:Colorant,R<:Real}(mapi::MapInfo{C}, img::AbstractImageDirect{R})
    # For this case we have to check whether color is defined along an array axis
    cd = colordim(img)
    if cd > 0
        dims = setdiff(1:ndims(img), cd)
        out = similar(img, C, size(img)[dims])
        map!(mapi, out, img, TypeConst{cd})
    else
        out = similar(img, C)
        map!(mapi, out, img)
    end
    out   # note this isn't type-stable
end

function map{T<:Colorant}(mapi::MapInfo{T}, img::AbstractImageIndexed)
    out = Image(Array(T, size(img)), properties(img))
    map!(mapi, out, img)
end

map!{T,T1,T2,N}(mapi::MapInfo{T1}, out::AbstractArray{T,N}, img::AbstractArray{T2,N}) =
    _map_a!(mapi, out, img)
function _map_a!{T,T1,T2,N}(mapi::MapInfo{T1}, out::AbstractArray{T,N}, img::AbstractArray{T2,N})
    mi = take(mapi, img)
    dimg = data(img)
    dout = data(out)
    size(dout) == size(dimg) || throw(DimensionMismatch())
    for I in eachindex(dout, dimg)
        @inbounds dout[I] = map(mi, dimg[I])
    end
    out
end

take(mapi::MapInfo, img::AbstractArray) = mapi
take{T}(mapi::ScaleAutoMinMax{T}, img::AbstractArray) = ScaleMinMax(T, img)

# Indexed images (colormaps)
map!{T,T1,N}(mapi::MapInfo{T}, out::AbstractArray{T,N}, img::AbstractImageIndexed{T1,N}) =
    _mapindx!(mapi, out, img)
function _mapindx!{T,T1,N}(mapi::MapInfo{T}, out::AbstractArray{T,N}, img::AbstractImageIndexed{T1,N})
    dimg = data(img)
    dout = data(out)
    cmap = map(mapi, img.cmap)
    for I in eachindex(dout, dimg)
        @inbounds dout[I] = cmap[dimg[I]]
    end
    out
end

# For when color is encoded along dimension CD
# NC is the number of color channels
# This is a very flexible implementation: color can be stored along any dimension, and it handles conversions to
# many different colorspace representations.
for (CT, NC) in ((Union{AbstractRGB,RGB24}, 3), (Union{RGBA,ARGB,ARGB32}, 4), (Union{AGray,GrayA,AGray32}, 2))
    for N = 1:4
        N1 = N+1
        @eval begin
function map!{T<:$CT,T1,T2,CD}(mapi::MapInfo{T}, out::AbstractArray{T1,$N}, img::AbstractArray{T2,$N1}, ::Type{TypeConst{CD}})
    mi = take(mapi, img)
    dimg = data(img)
    dout = data(out)
    # Set up the index along the color axis
    # We really only need dimension CD, but this will suffice
    @nexprs $NC k->(@nexprs $N1 d->(j_k_d = k))
    # Loop over all the elements in the output, performing the conversion on each color component
    @nloops $N i dout d->(d<CD ? (@nexprs $NC k->(j_k_d = i_d)) : (@nexprs $NC k->(j_k_{d+1} = i_d))) begin
        @inbounds @nref($N, dout, i) = @ncall $NC T k->(map1(mi, @nref($N1, dimg, j_k)))
    end
    out
end
        end
    end
end


#### MapInfo defaults
# Each "client" can define its own methods. "clients" include UFixed,
# RGB24/ARGB32, and ImageMagick

const bitshiftto8 = ((UFixed10, 2), (UFixed12, 4), (UFixed14, 6), (UFixed16, 8))

# typealias GrayType{T<:Fractional} Union{T, Gray{T}}
typealias GrayArray{T<:Fractional} Union{AbstractArray{T}, AbstractArray{Gray{T}}}
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
mapinfo{T<:UFixed}(::Type{T}, img::AbstractArray{T}) = MapNone(img)
mapinfo{T<:AbstractFloat}(::Type{T}, img::AbstractArray{T}) = MapNone(img)

# Grayscale methods
mapinfo(::Type{UFixed8}, img::GrayArray{UFixed8}) = MapNone{UFixed8}()
mapinfo(::Type{Gray{UFixed8}}, img::GrayArray{UFixed8}) = MapNone{Gray{UFixed8}}()
mapinfo(::Type{GrayA{UFixed8}}, img::AbstractArray{GrayA{UFixed8}}) = MapNone{GrayA{UFixed8}}()
for (T,n) in bitshiftto8
    @eval mapinfo(::Type{UFixed8}, img::GrayArray{$T}) = BitShift{UFixed8,$n}()
    @eval mapinfo(::Type{Gray{UFixed8}}, img::GrayArray{$T}) = BitShift{Gray{UFixed8},$n}()
    @eval mapinfo(::Type{GrayA{UFixed8}}, img::AbstractArray{GrayA{$T}}) = BitShift{GrayA{UFixed8},$n}()
end
mapinfo{T<:UFixed,F<:AbstractFloat}(::Type{T}, img::GrayArray{F}) = ClampMinMax(T, zero(F), one(F))
mapinfo{T<:UFixed,F<:AbstractFloat}(::Type{Gray{T}}, img::GrayArray{F}) = ClampMinMax(Gray{T}, zero(F), one(F))
mapinfo{T<:AbstractFloat, R<:Real}(::Type{T}, img::AbstractArray{R}) = MapNone(T)

mapinfo(::Type{RGB24}, img::Union{AbstractArray{Bool}, BitArray}) = MapNone{RGB24}()
mapinfo(::Type{ARGB32}, img::Union{AbstractArray{Bool}, BitArray}) = MapNone{ARGB32}()
mapinfo{F<:Fractional}(::Type{RGB24}, img::GrayArray{F}) = ClampMinMax(RGB24, zero(F), one(F))
mapinfo{F<:Fractional}(::Type{ARGB32}, img::AbstractArray{F}) = ClampMinMax(ARGB32, zero(F), one(F))

# Color->Color methods
mapinfo(::Type{RGB{UFixed8}}, img) = MapNone{RGB{UFixed8}}()
mapinfo(::Type{RGBA{UFixed8}}, img) = MapNone{RGBA{UFixed8}}()
for (T,n) in bitshiftto8
    @eval mapinfo(::Type{RGB{UFixed8}}, img::AbstractArray{RGB{$T}}) = BitShift{RGB{UFixed8},$n}()
    @eval mapinfo(::Type{RGBA{UFixed8}}, img::AbstractArray{RGBA{$T}}) = BitShift{RGBA{UFixed8},$n}()
end
mapinfo{F<:Fractional}(::Type{RGB{UFixed8}}, img::AbstractArray{RGB{F}}) = Clamp(RGB{UFixed8})
mapinfo{F<:Fractional}(::Type{RGBA{UFixed8}}, img::AbstractArray{RGBA{F}}) = Clamp(RGBA{UFixed8})



# Color->RGB24/ARGB32
mapinfo(::Type{RGB24}, img::AbstractArray{RGB24}) = MapNone{RGB24}()
mapinfo(::Type{ARGB32}, img::AbstractArray{ARGB32}) = MapNone{ARGB32}()
for C in tuple(subtypes(AbstractRGB)..., Gray)
    C == RGB24 && continue
    @eval mapinfo(::Type{RGB24}, img::AbstractArray{$C{UFixed8}}) = MapNone{RGB24}()
    @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$C{UFixed8}}) = MapNone{ARGB32}()
    for (T, n) in bitshiftto8
        @eval mapinfo(::Type{RGB24}, img::AbstractArray{$C{$T}}) = BitShift{RGB24, $n}()
        @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$C{$T}}) = BitShift{ARGB32, $n}()
    end
    @eval mapinfo{F<:AbstractFloat}(::Type{RGB24}, img::AbstractArray{$C{F}}) = ClampMinMax(RGB24, zero(F), one(F))
    @eval mapinfo{F<:AbstractFloat}(::Type{ARGB32}, img::AbstractArray{$C{F}}) = ClampMinMax(ARGB32, zero(F), one(F))
    for AC in subtypes(TransparentColor)
        length(AC.parameters) == 2 || continue
        @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$AC{$C{UFixed8},UFixed8}}) = MapNone{ARGB32}()
        @eval mapinfo(::Type{RGB24}, img::AbstractArray{$AC{$C{UFixed8},UFixed8}}) = MapNone{RGB24}()
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


# Clamping mapinfo client. Converts to RGB and uses UFixed, clamping
# floating-point values to [0,1].
mapinfo{T<:UFixed}(::Type{Clamp}, img::AbstractArray{T}) = MapNone{T}()
mapinfo{T<:AbstractFloat}(::Type{Clamp}, img::AbstractArray{T}) = ClampMinMax(UFixed8, zero(T), one(T))
const evaled_types = []

for ACV in (Color, AbstractRGB)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && !(CV.abstract)) || continue
        CVnew = CV<:AbstractGray ? Gray : RGB
        @eval mapinfo{T<:UFixed}(::Type{Clamp}, img::AbstractArray{$CV{T}}) = MapNone{$CVnew{T}}()
        @eval mapinfo{CV<:$CV}(::Type{Clamp}, img::AbstractArray{CV}) = Clamp{$CVnew{UFixed8}}()
        CVnew = CV<:AbstractGray ? Gray : BGR
        AC, CA       = alphacolor(CV), coloralpha(CV)
        ACnew, CAnew = alphacolor(CVnew), coloralpha(CVnew)
        if !in(AC, evaled_types)
            @eval begin
                mapinfo{T<:UFixed}(::Type{Clamp}, img::AbstractArray{$AC{T}}) = MapNone{$ACnew{T}}()
                mapinfo{P<:$AC}(::Type{Clamp}, img::AbstractArray{P}) = Clamp{$ACnew{UFixed8}}()
            end
            push!(evaled_types, AC)
        end
        if !in(CA, evaled_types)
            @eval begin
                mapinfo{T<:UFixed}(::Type{Clamp}, img::AbstractArray{$CA{T}}) = MapNone{$CAnew{T}}()
                mapinfo{P<:$CA}(::Type{Clamp}, img::AbstractArray{P}) = Clamp{$CAnew{UFixed8}}()
            end
            push!(evaled_types, CA)
        end
    end
end
mapinfo(::Type{Clamp}, img::AbstractArray{RGB24}) = MapNone{RGB{UFixed8}}()
mapinfo(::Type{Clamp}, img::AbstractArray{ARGB32}) = MapNone{BGRA{UFixed8}}()


# Backwards-compatibility
uint32color(img) = map(mapinfo(UInt32, img), img)
uint32color!(buf, img::AbstractArray) = map!(mapinfo(UInt32, img), buf, img)
uint32color!(buf, img::AbstractArray, mi::MapInfo) = map!(mi, buf, img)
uint32color!{T,N}(buf::Array{UInt32,N}, img::AbstractImageDirect{T,N}) =
    map!(mapinfo(UInt32, img), buf, img)
uint32color!{T,N,N1}(buf::Array{UInt32,N}, img::AbstractImageDirect{T,N1}) =
    map!(mapinfo(UInt32, img), buf, img, TypeConst{colordim(img)})
uint32color!{T,N}(buf::Array{UInt32,N}, img::AbstractImageDirect{T,N}, mi::MapInfo) =
    map!(mi, buf, img)
uint32color!{T,N,N1}(buf::Array{UInt32,N}, img::AbstractImageDirect{T,N1}, mi::MapInfo) =
    map!(mi, buf, img, TypeConst{colordim(img)})

"""
```
imgsc = sc(img)
imgsc = sc(img, min, max)
```

Applies default or specified `ScaleMinMax` mapping to the image.
"""
sc(img::AbstractArray) = map(ScaleMinMax(UFixed8, img), img)
sc(img::AbstractArray, mn::Real, mx::Real) = map(ScaleMinMax(UFixed8, img, mn, mx), img)

for (fn,T) in ((:float32, Float32), (:float64, Float64), (:ufixed8, UFixed8),
               (:ufixed10, UFixed10), (:ufixed12, UFixed12), (:ufixed14, UFixed14),
               (:ufixed16, UFixed16))
    @eval begin
        function $fn{C<:Colorant}(A::AbstractArray{C})
            newC = eval(C.name.name){$T}
            convert(Array{newC}, A)
        end
        $fn{C<:Colorant}(img::AbstractImage{C}) = shareproperties(img, $fn(data(img)))
    end
end


ufixedsc{T<:UFixed}(::Type{T}, img::AbstractImageDirect) = map(mapinfo(T, img), img)
ufixed8sc(img::AbstractImageDirect) = ufixedsc(UFixed8, img)
