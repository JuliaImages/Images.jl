#### Elementwise manipulations (scaling/clamping/type conversion) ####

# This file exists primarily to handle conversions for display and
# saving to disk. Both of these operations require Ufixed-valued
# elements, but with display we always want to convert to 8-bit
# whereas saving can handle 16-bit.
# We also can't trust that user images are clamped properly.
# Finally, this supports adjustable contrast limits.

# Structure of MapInfo subtype definitions:
#   - type definition
#   - constructors for scalars
#   - constructors for AbstractArrays
#   - implementation of map() for scalars
#   - implementation of map() for AbstractArrays
# map(mapi::MapInfo{T}, x) should return an object of type T (for x not an array)
# map1(mapi::MapInfo{T}, x) is designed to allow T<:ColorValue to work on
#    scalars x::Fractional

## MapNone
# At most, this does conversions

# Dispatch-based elementwise manipulations
abstract MapInfo{T}

immutable MapNone{T} <: MapInfo{T}; end

# Constructors
MapNone{T}(::Type{T}) = MapNone{T}()
MapNone{T}(val::T) = MapNone{T}()
MapNone{T}(A::AbstractArray{T}) = MapNone{T}()

# Implementation
map{T<:Real}(mapi::MapNone{T}, val::Real) = convert(T, val)
map{C<:ColorValue}(mapi::MapNone{C}, c::ColorValue) = convert(C, c)
map1(mapi::Union(MapNone{RGB24},MapNone{ARGB32}), val::Fractional) = convert(Ufixed8, val)
map1{CT<:ColorType}(mapi::MapNone{CT}, val::Fractional) = convert(eltype(CT), val)

map{T<:ColorType}(mapi::MapNone{T}, img::AbstractImageIndexed{T}) = convert(Image{T}, img)
map{C<:ColorType}(mapi::MapNone{C}, img::AbstractImageDirect{C}) = img  # ambiguity resolution
map{T}(mapi::MapNone{T}, img::AbstractArray{T}) = img


## BitShift
# This is really a "saturating bitshift", for example
#    map(BitShift{Uint8,7}(), 0xf0ff) == 0xff rather than 0xe1 even though 0xf0ff>>>7 == 0x01e1

immutable BitShift{T,N} <: MapInfo{T} end
BitShift{T}(::Type{T}, n::Int) = BitShift{T,n}()  # note that this is not type-stable

# Implementation
immutable BS{N} end
_map{T<:Unsigned,N}(::Type{T}, ::Type{BS{N}}, val::Unsigned) = (v = val>>>N; tm = oftype(val, typemax(T)); convert(T, ifelse(v > tm, tm, v)))
_map{T<:Ufixed,N}(::Type{T}, ::Type{BS{N}}, val::Ufixed) = reinterpret(T, _map(FixedPointNumbers.rawtype(T), BS{N}, reinterpret(val)))
map{T<:Real,N}(mapi::BitShift{T,N}, val::Real) = _map(T, BS{N}, val)
map1{N}(mapi::Union(BitShift{RGB24,N},BitShift{ARGB32,N}), val::Unsigned) = _map(Uint8, BS{N}, val)
map1{N}(mapi::Union(BitShift{RGB24,N},BitShift{ARGB32,N}), val::Ufixed) = _map(Ufixed8, BS{N}, val)
map1{CT<:ColorType,N}(mapi::BitShift{CT,N}, val::Ufixed) = _map(eltype(CT), BS{N}, val)


## Clamp types
# The Clamp types just enforce bounds, but do not scale or offset

# Types and constructors
abstract AbstractClamp{T} <: MapInfo{T}
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
Clamp{T}(::Type{T}) = Clamp{T}()

# Implementation
map{T<:Real,F<:Real}(mapi::ClampMin{T,F}, val::F) = convert(T, max(val, mapi.min))
map{T<:Real,F<:Real}(mapi::ClampMax{T,F}, val::F) = convert(T, min(val, mapi.max))
map{T<:Real,F<:Real}(mapi::ClampMinMax{T,F}, val::F) = convert(T,min(max(val, mapi.min), mapi.max))
map1{T<:Union(RGB24,ARGB32),F<:Fractional}(mapi::ClampMin{T,F}, val::F) = convert(Ufixed8, max(val, mapi.min))
map1{T<:Union(RGB24,ARGB32),F<:Fractional}(mapi::ClampMax{T,F}, val::F) = convert(Ufixed8, min(val, mapi.max))
map1{T<:Union(RGB24,ARGB32),F<:Fractional}(mapi::ClampMinMax{T,F}, val::F) = convert(Ufixed8,min(max(val, mapi.min), mapi.max))
map1{CT<:ColorType,F<:Fractional}(mapi::ClampMin{CT,F}, val::F) = convert(eltype(CT), max(val, mapi.min))
map1{CT<:ColorType,F<:Fractional}(mapi::ClampMax{CT,F}, val::F) = convert(eltype(CT), min(val, mapi.max))
map1{CT<:ColorType,F<:Fractional}(mapi::ClampMinMax{CT,F}, val::F) = convert(eltype(CT), min(max(val, mapi.min), mapi.max))

map{To<:Real}(::Clamp{To}, val::Real) = clamp01(To, val)
map{To<:Real}(::Clamp{Gray{To}}, val::AbstractGray) = Gray(clamp01(To, val.val))
map1{CT<:AbstractRGB}(::Clamp{CT}, val::Real) = clamp01(eltype(CT), val)
map1{CT<:AbstractRGB,T}(::Clamp{AlphaColor{CT,T}}, val::Real) = clamp01(T, val)
map1{CT<:AbstractRGB,T}(::Clamp{AlphaColorValue{CT,T}}, val::Real) = clamp01(T, val)

# Also available as a stand-alone function
clamp01{T}(::Type{T}, x::Real) = convert(T, min(max(x, zero(x)), one(x)))
clamp01(x::Real) = clamp01(typeof(x), x)
for CV in subtypes(AbstractRGB)
    @eval begin
        clamp01{T}(x::$CV{T}) = clamp01($CV{T}, x)
        clamp01{To}(::Type{$CV{To}}, x::AbstractRGB) = $CV{To}(clamp01(To, x.r), clamp01(To, x.g), clamp01(To, x.b))
        clamp01{T}(x::AlphaColor{$CV{T},T}) = clamp01(AlphaColor{$CV{T},T}, x)
        clamp01{T}(x::AlphaColorValue{$CV{T},T}) = clamp01(AlphaColorValue{$CV{T},T}, x)
        clamp01{To,CV2<:AbstractRGB}(::Type{AlphaColor{$CV{To},To}}, x::AlphaColor{CV2}) =
            AlphaColor{$CV{To},To}(clamp01(To, x.c.r), clamp01(To, x.c.g), clamp01(To, x.c.b), clamp01(To, x.alpha))
        clamp01{To,CV2<:AbstractRGB}(::Type{AlphaColorValue{$CV{To},To}}, x::AlphaColorValue{CV2}) =
            AlphaColorValue{$CV{To},To}(clamp01(To, x.c.r), clamp01(To, x.c.g), clamp01(To, x.c.b), clamp01(To, x.alpha))
        clamp01{To,CV2<:AbstractRGB}(::Type{AlphaColor{$CV{To},To}}, x::AlphaColorValue{CV2}) =
            AlphaColor{$CV{To},To}(clamp01(To, x.c.r), clamp01(To, x.c.g), clamp01(To, x.c.b), clamp01(To, x.alpha))
        clamp01{To,CV2<:AbstractRGB}(::Type{AlphaColorValue{$CV{To},To}}, x::AlphaColor{CV2}) =
            AlphaColorValue{$CV{To},To}(clamp01(To, x.c.r), clamp01(To, x.c.g), clamp01(To, x.c.b), clamp01(To, x.alpha))
    end
end

# clamp is generic for any colorspace; this version does the right thing for any RGB type
clamp(x::AbstractRGB) = clamp01(x)
for CV in subtypes(AbstractRGB)
    @eval begin
        clamp{T}(x::AlphaColorValue{$CV{T},T}) = clamp01(x)
        clamp{T}(x::AlphaColor{$CV{T},T}) = clamp01(x)
    end
end

## ScaleMinMax
# This clamps, subtracts the min value, then scales

immutable ScaleMinMax{To,From,S<:FloatingPoint} <: MapInfo{To}
    min::From
    max::From
    s::S
end

ScaleMinMax{To,From}(::Type{To}, min::From, max::From, s::FloatingPoint) = ScaleMinMax{To,From,typeof(s)}(min, max, s)
ScaleMinMax{To<:Union(Fractional,ColorType),From<:Real}(::Type{To}, mn::From, mx::From) = ScaleMinMax(To, mn, mx, 1.0f0/(float32(mx)-float32(mn)))

# ScaleMinMax constructors that take AbstractArray input
ScaleMinMax{To,From}(::Type{To}, img::AbstractArray{From}, mn::Real, mx::Real) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(float32(convert(From,mx))-float32(convert(From,mn))))
ScaleMinMax{To,From,R<:Real}(::Type{To}, img::AbstractArray{From}, mn::Gray{R}, mx::Gray{R}) = ScaleMinMax(To, convert(From,mn), convert(From,mx), 1.0f0/(float32(convert(From,mx.val))-float32(convert(From,mn.val))))
ScaleMinMax{To}(::Type{To}, img::AbstractArray) = ScaleMinMax(To, img, minfinite(img), maxfinite(img))

function map{To<:Real,From<:Union(Real,Gray)}(mapi::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < mapi.min, zero(From), ifelse(val > mapi.max, mapi.max-mapi.min, val-mapi.min))
    convert(To, mapi.s*t)
end
function map1{To<:Union(RGB24,ARGB32),From<:Real}(mapi::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < mapi.min, zero(From), ifelse(val > mapi.max, mapi.max-mapi.min, val-mapi.min))
    convert(Ufixed8, mapi.s*t)
end
function map1{To<:Union(RGB24,ARGB32),From<:Fractional}(mapi::ScaleMinMax{To,Gray{From}}, val::From)
    t = ifelse(val < mapi.min.val, zero(From), ifelse(val > mapi.max.val, mapi.max.val-mapi.min.val, val-mapi.min.val))
    convert(Ufixed8, mapi.s*t)
end
function map1{To<:ColorType,From<:Real}(mapi::ScaleMinMax{To,From}, val::From)
    t = ifelse(val < mapi.min, zero(From), ifelse(val > mapi.max, mapi.max-mapi.min, val-mapi.min))
    convert(eltype(To), mapi.s*t)
end


## ScaleSigned
# Multiplies by a scaling factor and then clamps to the range [-1,1].
# Intended for positive/negative coloring

immutable ScaleSigned{T, S<:FloatingPoint} <: MapInfo{T}
    s::S
end
ScaleSigned{T}(::Type{T}, s::FloatingPoint) = ScaleSigned{T, typeof(s)}(s)

ScaleSigned{T}(::Type{T}, img::AbstractArray) = ScaleSigned(T, 1.0f0/maxabsfinite(img))
ScaleSigned(img::AbstractArray) = ScaleSigned(Float32, img)

map{T}(mapi::ScaleSigned{T}, val::Real) = convert(T, clamppm(mapi.s*val))
function map{C<:Union(RGB24, RGB{Ufixed8})}(mapi::ScaleSigned{C}, val::Real)
    x = clamppm(mapi.s*val)
    g = Ufixed8(abs(x))
    ifelse(x >= 0, C(g, zero(Ufixed8), g), C(zero(Ufixed8), g, zero(Ufixed8)))
end

clamppm(x::Real) = ifelse(x >= 0, min(x, one(x)), max(x, -one(x)))

## ScaleAutoMinMax
# Works only on whole arrays, not values

immutable ScaleAutoMinMax{T} <: MapInfo{T} end
ScaleAutoMinMax{T}(::Type{T}) = ScaleAutoMinMax{T}()
ScaleAutoMinMax() = ScaleAutoMinMax{Ufixed8}()




# Conversions to RGB{T}, RGBA{T}, RGB24, ARGB32,
# for grayscale, AbstractRGB, and abstract ARGB inputs.
# This essentially "vectorizes" map using map1
for SI in (MapInfo, AbstractClamp)
    for ST in subtypes(SI)
        ST.abstract && continue
        ST == ScaleSigned && continue  # ScaleSigned gives an RGB from a scalar, so don't "vectorize" it
        @eval begin
            # Grayscale and GrayAlpha inputs
            map(mapi::$ST{RGB24}, g::Gray) = map(mapi, g.val)
            map(mapi::$ST{RGB24}, g::Real) = (x = map1(mapi, g); convert(RGB24, RGB{Ufixed8}(x,x,x)))
            map{T}(mapi::$ST{RGB24}, g::GrayAlpha{T}) = map(mapi, g.c.val)
            map(mapi::$ST{ARGB32}, g::Gray) = map(mapi, g.val)
            function map(mapi::$ST{ARGB32}, g::Real)
                x = map1(mapi, g)
                convert(ARGB32, ARGB{Ufixed8}(x,x,x,0xffuf8))
            end
            function map{T}(mapi::$ST{ARGB32}, g::GrayAlpha{T})
                x = map1(mapi, g.c.val)
                convert(ARGB32, ARGB{Ufixed8}(x,x,x,map1(mapi, g.alpha)))
            end
            map{T}(mapi::$ST{RGB{T}}, g::Gray) = map(mapi, g.val)
            function map{T}(mapi::$ST{RGB{T}}, g::Real)
                x = map1(mapi, g)
                RGB{T}(x,x,x)
            end
            map{T}(mapi::$ST{ARGB{T}}, g::Gray) = map(mapi, g.val)
            function map{T}(mapi::$ST{ARGB{T}}, g::Real)
                x = map1(mapi, g)
                ARGB{T}(x,x,x)
            end
            function map{T,S}(mapi::$ST{ARGB{T}}, g::GrayAlpha{S})
                x = map1(mapi, g.c.val)
                ARGB{T}(x,x,x,map1(mapi, g.alpha))
            end
            map{T}(mapi::$ST{RGBA{T}}, g::Gray) = map(mapi, g.val)
            function map{T}(mapi::$ST{RGBA{T}}, g::Real)
                x = map1(mapi, g)
                RGBA{T}(x,x,x)
            end
            function map{T,S}(mapi::$ST{RGBA{T}}, g::GrayAlpha{S})
                x = map1(mapi, g.c.val)
                RGBA{T}(x,x,x,map1(mapi, g.alpha))
            end
            # AbstractRGB and abstract ARGB inputs
            map(mapi::$ST{RGB24}, rgb::AbstractRGB) =
                convert(RGB24, RGB{Ufixed8}(map1(mapi, rgb.r), map1(mapi, rgb.g), map1(mapi, rgb.b)))
            map{C<:AbstractRGB, TC}(mapi::$ST{RGB24}, argb::AbstractAlphaColorValue{C,TC}) =
                convert(RGB24, RGB{Ufixed8}(map1(mapi, argb.c.r), map1(mapi, argb.c.g),
                                            map1(mapi, argb.c.b)))
            map{C<:AbstractRGB, TC}(mapi::$ST{ARGB32}, argb::AbstractAlphaColorValue{C,TC}) =
                convert(ARGB32, ARGB{Ufixed8}(map1(mapi, argb.c.r), map1(mapi, argb.c.g),
                                              map1(mapi, argb.c.b), map1(mapi, argb.alpha)))
            map(mapi::$ST{ARGB32}, rgb::AbstractRGB) =
                convert(ARGB32, ARGB{Ufixed8}(map1(mapi, rgb.r), map1(mapi, rgb.g), map1(mapi, rgb.b)))
            map{T}(mapi::$ST{RGB{T}}, rgb::AbstractRGB) =
                RGB{T}(map1(mapi, rgb.r), map1(mapi, rgb.g), map1(mapi, rgb.b))
            map{T,C<:AbstractRGB, TC}(mapi::$ST{RGB{T}}, argb::AbstractAlphaColorValue{C,TC}) =
                RGB{T}(map1(mapi, argb.c.r), map1(mapi, argb.c.g), map1(mapi, argb.c.b))
            map{T, C<:AbstractRGB, TC}(mapi::$ST{ARGB{T}}, argb::AbstractAlphaColorValue{C,TC}) =
                ARGB{T}(map1(mapi, argb.c.r), map1(mapi, argb.c.g),
                        map1(mapi, argb.c.b), map1(mapi, argb.alpha))
            map{T, C<:AbstractRGB, TC}(mapi::$ST{RGBA{T}}, argb::AbstractAlphaColorValue{C,TC}) =
                RGBA{T}(map1(mapi, argb.c.r), map1(mapi, argb.c.g),
                        map1(mapi, argb.c.b), map1(mapi, argb.alpha))
            map{T}(mapi::$ST{ARGB{T}}, rgb::AbstractRGB) =
                ARGB{T}(map1(mapi, rgb.r), map1(mapi, rgb.g), map1(mapi, rgb.b))
            map{T}(mapi::$ST{RGBA{T}}, rgb::AbstractRGB) =
                RGBA{T}(map1(mapi, rgb.r), map1(mapi, rgb.g), map1(mapi, rgb.b))
        end
    end
end

# # Apply to any ColorType
# map(f::Callable, x::ColorValue) = f(x)
# map(mapi, x::ColorValue) = map(mapi, convert(RGB, x))
# map{C<:ColorValue, TC}(f::Callable, x::AbstractAlphaColorValue{C, TC}) = f(convert(ARGB, x))
# map{C<:ColorValue, TC}(mapi, x::AbstractAlphaColorValue{C, TC}) = map(mapi, convert(ARGB, x))

## Fallback definitions of map() for array types

function map{T}(mapi::MapInfo{T}, img::AbstractArray)
    out = similar(img, T)
    map!(mapi, out, img)
end

map{C<:ColorType,R<:Real}(mapi::MapNone{C}, img::AbstractImageDirect{R}) = mapcd(mapi, img)  # ambiguity resolution
map{C<:ColorType,R<:Real}(mapi::MapInfo{C}, img::AbstractImageDirect{R}) = mapcd(mapi, img)
function mapcd{C<:ColorType,R<:Real}(mapi::MapInfo{C}, img::AbstractImageDirect{R})
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

function map{T<:ColorType}(mapi::MapInfo{T}, img::AbstractImageIndexed)
    out = Image(Array(T, size(img)), properties(img))
    map!(mapi, out, img)
end

@ngenerate N typeof(out) function map!{T,T1,T2,N}(mapi::MapInfo{T1}, out::AbstractArray{T,N}, img::AbstractArray{T2,N})
    mi = take(mapi, img)
    dimg = data(img)
    dout = data(out)
    @nloops N i dout begin
        @inbounds @nref(N, dout, i) = map(mi, @nref(N, dimg, i))
    end
    out
end

take(mapi::MapInfo, img::AbstractArray) = mapi
take{T}(mapi::ScaleAutoMinMax{T}, img::AbstractArray) = ScaleMinMax(T, img)
take{To,From}(mapi::ScaleMinMax{To}, img::AbstractArray{From}) = ScaleMinMax(To, convert(From, mapi.min), convert(From, mapi.max), mapi.s)

# Indexed images (colormaps)
@ngenerate N typeof(out) function map!{T,T1,N}(mapi::MapInfo{T}, out::AbstractArray{T,N}, img::AbstractImageIndexed{T1,N})
    dimg = data(img)
    dout = data(out)
    cmap = map(mapi, img.cmap)
    @nloops N i dout begin
        @inbounds @nref(N, dout, i) = cmap[@nref(N, dimg, i)]
    end
    out
end

# For when color is encoded along dimension CD
# NC is the number of color channels
# This is a very flexible implementation: color can be stored along any dimension, and it handles conversions to
# many different colorspace representations.
for (CT, NC) in ((Union(AbstractRGB,RGB24), 3), (Union(RGBA,ARGB,ARGB32), 4), (Union(GrayAlpha,AGray32), 2))
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
# Each "client" can define its own methods. "clients" include Ufixed, RGB24/ARGB32, and ImageMagick

const bitshiftto8 = ((Ufixed10, 2), (Ufixed12, 4), (Ufixed14, 6), (Ufixed16, 8))

# typealias GrayType{T<:Fractional} Union(T, Gray{T})
typealias GrayArray{T<:Fractional} Union(AbstractArray{T}, AbstractArray{Gray{T}})
# note, though, that we need to override for AbstractImage in case the "colorspace" property is defined differently

# mapinfo{T}(::Type{T}, img::AbstractArray{T}) = MapNone(img)
# Grayscale methods
for (T,n) in bitshiftto8
    @eval mapinfo(::Type{Ufixed8}, img::GrayArray{$T}) = BitShift{Ufixed8,$n}()
    @eval mapinfo(::Type{Gray{Ufixed8}}, img::GrayArray{$T}) = BitShift{Gray{Ufixed8},$n}()
end
mapinfo{T<:Ufixed,F<:FloatingPoint}(::Type{T}, img::AbstractArray{F}) = ClampMinMax(T, zero(F), one(F))
mapinfo{T<:FloatingPoint, R<:Real}(::Type{T}, img::AbstractArray{R}) = MapNone(T)
mapinfo{F<:Fractional}(::Type{RGB24}, img::GrayArray{F}) = ClampMinMax(RGB24, zero(F), one(F))
mapinfo{F<:Fractional}(::Type{ARGB32}, img::AbstractArray{F}) = ClampMinMax(ARGB32, zero(F), one(F))
mapinfo(::Type{RGB24}, img::AbstractArray{RGB24}) = MapNone{RGB24}()
mapinfo(::Type{ARGB32}, img::AbstractArray{ARGB32}) = MapNone{ARGB32}()


# Color->RGB24/ARGB32
for C in tuple(subtypes(AbstractRGB)..., Gray)
    @eval mapinfo(::Type{RGB24}, img::AbstractArray{$C{Ufixed8}}) = MapNone{RGB24}()
    @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$C{Ufixed8}}) = MapNone{ARGB32}()
    for (T, n) in bitshiftto8
        @eval mapinfo(::Type{RGB24}, img::AbstractArray{$C{$T}}) = BitShift{RGB24, $n}()
        @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$C{$T}}) = BitShift{ARGB32, $n}()
    end
    @eval mapinfo{F<:FloatingPoint}(::Type{RGB24}, img::AbstractArray{$C{F}}) = ClampMinMax(RGB24, zero(F), one(F))
    @eval mapinfo{F<:FloatingPoint}(::Type{ARGB32}, img::AbstractArray{$C{F}}) = ClampMinMax(ARGB32, zero(F), one(F))
    for AC in subtypes(AbstractAlphaColorValue)
        length(AC.parameters) == 2 || continue
        @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$AC{$C{Ufixed8},Ufixed8}}) = MapNone{ARGB32}()
        @eval mapinfo(::Type{RGB24}, img::AbstractArray{$AC{$C{Ufixed8},Ufixed8}}) = MapNone{RGB24}()
        for (T, n) in bitshiftto8
            @eval mapinfo(::Type{ARGB32}, img::AbstractArray{$AC{$C{$T},$T}}) = BitShift{ARGB32, $n}()
            @eval mapinfo(::Type{RGB24}, img::AbstractArray{$AC{$C{$T},$T}}) = BitShift{RGB24, $n}()
        end
        @eval mapinfo{F<:FloatingPoint}(::Type{ARGB32}, img::AbstractArray{$AC{$C{F},F}}) = ClampMinMax(ARGB32, zero(F), one(F))
        @eval mapinfo{F<:FloatingPoint}(::Type{RGB24}, img::AbstractArray{$AC{$C{F},F}}) = ClampMinMax(RGB24, zero(F), one(F))
    end
end

mapinfo{CT<:ColorType}(::Type{RGB24},  img::AbstractArray{CT}) = MapNone{RGB24}()
mapinfo{CT<:ColorType}(::Type{ARGB32}, img::AbstractArray{CT}) = MapNone{ARGB32}()


# Uint32 conversions will use ARGB32 for images that have an alpha channel,
# and RGB24 when not
mapinfo{CV<:Union(Fractional,ColorValue)}(::Type{Uint32}, img::AbstractArray{CV}) = mapinfo(RGB24, img)
mapinfo{CV<:AbstractAlphaColorValue}(::Type{Uint32}, img::AbstractArray{CV}) = mapinfo(ARGB32, img)
mapinfo(::Type{Uint32}, img::AbstractArray{Uint32}) = MapNone{Uint32}()

# RGB{Ufixed8} fallback
mapinfo(::Type{RGB{Ufixed8}}, img) = MapNone{RGB{Ufixed8}}()

# ImageMagick client is defined in io.jl

# Backwards-compatibility
uint32color(img) = map(mapinfo(Uint32, img), img)
uint32color!(buf, img::AbstractArray) = map!(mapinfo(Uint32, img), buf, img)
uint32color!(buf, img::AbstractArray, mi::MapInfo) = map!(mi, buf, img)
uint32color!{T,N}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N}) =
    map!(mapinfo(Uint32, img), buf, img)
uint32color!{T,N,N1}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N1}) =
    map!(mapinfo(Uint32, img), buf, img, TypeConst{colordim(img)})
uint32color!{T,N}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N}, mi::MapInfo) =
    map!(mi, buf, img)
uint32color!{T,N,N1}(buf::Array{Uint32,N}, img::AbstractImageDirect{T,N1}, mi::MapInfo) =
    map!(mi, buf, img, TypeConst{colordim(img)})


sc(img::AbstractArray) = map(ScaleMinMax(img), img)
sc(img::AbstractArray, mn::Real, mx::Real) = map(ScaleMinMax(img, mn, mx), img)

convert{T}(::Type{AbstractImageDirect{T,2}},M::Tridiagonal) = error("Not defined") # prevent ambiguity warning

convert{T<:Real,S<:Real}(::Type{Image{T}}, img::AbstractImageDirect{S}) = map(mapinfo(T, img), img)

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


ufixedsc{T<:Ufixed}(::Type{T}, img::AbstractImageDirect) = map(mapinfo(T, img), img)
ufixed8sc(img::AbstractImageDirect) = ufixedsc(Ufixed8, img)
