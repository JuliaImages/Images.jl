module ColorTypes

using Color, FixedPointNumbers
import Color: Fractional, _convert
import Base: ==, clamp, convert, length, one, promote_array_type, promote_rule, zero

export ARGB, BGR, RGB1, RGB4, BGRA, AbstractGray, Gray, GrayAlpha, Gray24, AGray32, YIQ, AlphaColor, ColorType

typealias ColorType Union(ColorValue, AbstractAlphaColorValue)

# An alpha-channel-first memory layout
immutable AlphaColor{C<:ColorValue, T<:Fractional} <: AbstractAlphaColorValue{C,T}
    alpha::T
    c::C

    AlphaColor(x1::Real, x2::Real, x3::Real, alpha::Real = 1.0) = new(alpha, C(x1, x2, x3))
    AlphaColor(c::ColorValue, alpha::Real) = new(alpha, c)
end
AlphaColor{T<:Fractional}(c::ColorValue{T}, alpha::T = one(T)) = AlphaColor{typeof(c),T}(c, alpha)

typealias ARGB{T} AlphaColor{RGB{T}, T}


# Little-endian RGB (useful for BGRA & Cairo)
immutable BGR{T<:Fractional} <: AbstractRGB{T}
    b::T
    g::T
    r::T

    BGR(r::Real, g::Real, b::Real) = new(b, g, r)
end
BGR(r::Integer, g::Integer, b::Integer) = BGR{Float64}(r, g, b)
BGR(r::Fractional, g::Fractional, b::Fractional) = (T = promote_type(typeof(r), typeof(g), typeof(b)); BGR{T}(r, g, b))

typealias BGRA{T} AlphaColorValue{BGR{T}, T}


# Some readers return a byte for an alpha channel even if it's not meaningful
immutable RGB1{T<:Fractional} <: AbstractRGB{T}
    alphadummy::T
    r::T
    g::T
    b::T

    RGB1(r::Real, g::Real, b::Real) = new(one(T), r, g, b)
end
RGB1(r::Integer, g::Integer, b::Integer) = RGB1{Float64}(r, g, b)
RGB1(r::Fractional, g::Fractional, b::Fractional) = (T = promote_type(typeof(r), typeof(g), typeof(b)); RGB1{T}(r, g, b))

immutable RGB4{T<:Fractional} <: AbstractRGB{T}
    r::T
    g::T
    b::T
    alphadummy::T

    RGB4(r::Real, g::Real, b::Real) = new(r, g, b, one(T))
end
RGB4(r::Integer, g::Integer, b::Integer) = RGB4{Float64}(r, g, b)
RGB4(r::Fractional, g::Fractional, b::Fractional) = (T = promote_type(typeof(r), typeof(g), typeof(b)); RGB4{T}(r, g, b))


# Sometimes you want to be explicit about grayscale. Also needed for GrayAlpha.
abstract AbstractGray{T} <: ColorValue{T}
immutable Gray{T<:Fractional} <: AbstractGray{T}
    val::T
end
convert{T}(::Type{Gray{T}}, x::Gray{T}) = x
convert{T,S}(::Type{Gray{T}}, x::Gray{S}) = Gray{T}(x.val)
convert{T<:Real}(::Type{T}, x::Gray) = convert(T, x.val)
convert{T}(::Type{Gray{T}}, x::Real) = Gray{T}(x)

convert{T}(::Type{Gray{T}}, x::AbstractRGB) = convert(Gray{T}, 0.299*x.r + 0.587*x.g + 0.114*x.b)  # Rec 601 luma conversion

zero{T}(::Type{Gray{T}}) = Gray{T}(zero(T))
 one{T}(::Type{Gray{T}}) = Gray{T}(one(T))

immutable Gray24 <: ColorValue{Uint8}
    color::Uint32
end
Gray24() = Gray24(0)
Gray24(val::Uint8) = (g = uint32(val); g<<16 | g<<8 | g)
Gray24(val::Ufixed8) = Gray24(reinterpret(val))

convert(::Type{Uint32}, g::Gray24) = g.color


typealias GrayAlpha{T} AlphaColorValue{Gray{T}, T}

immutable AGray32 <: AbstractAlphaColorValue{Gray24, Uint8}
    color::Uint32
end
AGray32() = AGray32(0)
AGray32(val::Uint8, alpha::Uint8) = (g = uint32(val); uint32(alpha)<<24 | g<<16 | g<<8 | g)
AGray32(val::Ufixed8, alpha::Ufixed8) = AGray32(reinterpret(val), reinterpret(alpha))

convert(::Type{Uint32}, g::AGray32) = g.color


convert(::Type{RGB}, x::Gray) = RGB(x.val, x.val, x.val)
convert{T}(::Type{RGB{T}}, x::Gray) = (g = convert(T, x.val); RGB{T}(g, g, g))

# YIQ (NTSC)
immutable YIQ{T<:FloatingPoint} <: ColorValue{T}
    y::T
    i::T
    q::T

    YIQ(y::Real, i::Real, q::Real) = new(y, i, q)
end
YIQ(y::FloatingPoint, i::FloatingPoint, q::FloatingPoint) = (T = promote_type(typeof(y), typeof(i), typeof(q)); YIQ{T}(y, i, q))

clamp{T}(c::YIQ{T}) = YIQ{T}(clamp(c.y, zero(T), one(T)),
                             clamp(c.i, convert(T,-0.5957), convert(T,0.5957)),
                             clamp(c.q, convert(T,-0.5226), convert(T,0.5226)))

function convert{T}(::Type{YIQ{T}}, c::AbstractRGB)
    rgb = clamp(c)
    YIQ{T}(0.299*rgb.r+0.587*rgb.g+0.114*rgb.b,
           0.595716*rgb.r-0.274453*rgb.g-0.321263*rgb.b,
           0.211456*rgb.r-0.522591*rgb.g+0.311135*rgb.b)
end
convert{T}(::Type{YIQ}, c::AbstractRGB{T}) = convert(YIQ{T}, c)

function _convert{T}(::Type{RGB{T}}, c::YIQ)
    cc = clamp(c)
    RGB{T}(cc.y+0.9563*cc.i+0.6210*cc.q,
           cc.y-0.2721*cc.i-0.6474*cc.q,
           cc.y-1.1070*cc.i+1.7046*cc.q)
end

## Generic algorithms

length(cv::ColorType) = div(sizeof(cv), sizeof(eltype(cv)))
# Because this can be called as `length(RGB)`, we might need to fill in a default element type.
# But the compiler chokes if we ask it to create RGB{Float64}{Float64}, even if that's inside
# the non-evaluated branch of a ternary expression, so we have to be sneaky about this.
length{CV<:ColorValue}(::Type{CV}) = _length(CV, eltype(CV))
_length{CV<:ColorValue}(::Type{CV}, ::Type{Any}) = length(CV{Float64})
_length{CV<:ColorValue}(::Type{CV}, ::DataType)  = div(sizeof(CV), sizeof(eltype(CV)))
length{CV,T}(::Type{AlphaColorValue{CV,T}}) = length(CV)+1
length{CV,T}(::Type{AlphaColor{CV,T}}) = length(CV)+1

# Return types for arithmetic operations
multype(a::Type,b::Type) = typeof(one(a)*one(b))
sumtype(a::Type,b::Type) = typeof(one(a)+one(b))
divtype(a::Type,b::Type) = typeof(one(a)/one(b))

# Math on ColorValues. These implementations encourage inlining and,
# for the case of Ufixed types, nearly halve the number of multiplications.
for CV in subtypes(AbstractRGB)
    @eval begin
        (*){R<:Real,T}(f::R, c::$CV{T}) = $CV{multype(R,T)}(f*c.r, f*c.g, f*c.b)
        function (*){R<:FloatingPoint,T<:Ufixed}(f::R, c::$CV{T})
            fs = f/reinterpret(one(T))
            $CV{multype(R,T)}(fs*reinterpret(c.r), fs*reinterpret(c.g), fs*reinterpret(c.b))
        end
        function (*){R<:Ufixed,T<:Ufixed}(f::R, c::$CV{T})
            fs = reinterpret(f)/widen(reinterpret(one(T)))^2
            $CV{multype(R,T)}(fs*reinterpret(c.r), fs*reinterpret(c.g), fs*reinterpret(c.b))
        end
        (*)(c::$CV, f::Real) = (*)(f, c)
        (.*)(f::Real, c::$CV) = (*)(f, c)
        (.*)(c::$CV, f::Real) = (*)(f, c)
        (/)(c::$CV, f::Real) = (one(f)/f)*c
        (/)(c::$CV, f::Integer) = (one(eltype(c))/f)*c
        (./)(c::$CV, f::Real) = (/)(c, f)
        function (/){R<:FloatingPoint,T<:Ufixed}(c::$CV{T}, f::R)
            fs = one(R)/(f*reinterpret(one(T)))
            $CV{divtype(R,T)}(fs*reinterpret(c.r), fs*reinterpret(c.g), fs*reinterpret(c.b))
        end
        (+){S,T}(a::$CV{S}, b::$CV{T}) = $CV{sumtype(S,T)}(a.r+b.r, a.g+b.g, a.b+b.b)
    end
end

# To help type inference
for ACV in (ColorValue, AbstractRGB)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && !(CV.abstract)) || continue
        @eval promote_array_type{T<:Real,S<:Real}(::Type{T}, ::Type{$CV{S}}) = $CV{promote_type(T, S)}
        @eval promote_rule{T<:Fractional,S<:Fractional}(::Type{$CV{T}}, ::Type{$CV{S}}) = $CV{promote_type(T, S)}
        for AC in subtypes(AbstractAlphaColorValue)
            (length(AC.parameters) == 2 && !(AC.abstract)) || continue
            @eval promote_array_type{T<:Real,S<:Real}(::Type{T}, ::Type{$AC{$CV{S},S}}) = (TS = promote_type(T, S); $AC{$CV{TS}, TS})
            @eval promote_rule{T<:Fractional,S<:Fractional}(::Type{$CV{T}}, ::Type{$CV{S}}) = $CV{promote_type(T, S)}
        end
    end
end

for (CV, CVstr, fields) in ((BGR,  "BGR",  (:(c.r),:(c.g),:(c.b))),
                            (RGB1, "RGB1", (:(c.r),:(c.g),:(c.b))),
                            (RGB4, "RGB4", (:(c.r),:(c.g),:(c.b))),
                            (ARGB, "ARGB", (:(c.c.r),:(c.c.g),:(c.c.b),:(c.alpha))),
                            (BGRA, "BGRA", (:(c.c.r),:(c.c.g),:(c.c.b),:(c.alpha))),
                            (Gray, "Gray", (:(c.val),)),
                            (GrayAlpha, "GrayAlpha", (:(c.c.val),:(c.alpha))))
    Color.makeshow(CV, CVstr, fields)
end

for T in (RGB24, ARGB32, Gray24, AGray32)
    @eval begin
        ==(x::Uint32, y::$T) = x == convert(Uint32, y)
        ==(x::$T, y::Uint32) = ==(y, x)
    end
end
=={T}(x::Gray{T}, y::Gray{T}) = x.val == y.val
=={T}(x::T, y::Gray{T}) = x == convert(T, y)
=={T}(x::Gray{T}, y::T) = ==(y, x)

end
