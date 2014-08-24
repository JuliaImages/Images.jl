module ColorTypes

using Color
import Color.Fractional
import Base: convert, length, promote_array_type

export ARGB, BGR, RGB1, RGB4, BGRA, AbstractGray, Gray, GrayAlpha, AlphaColor, ColorType

typealias ColorType Union(ColorValue, AbstractAlphaColorValue)

# An alpha-channel-first memory layout
immutable AlphaColor{C<:ColorValue, T<:Number} <: AbstractAlphaColorValue{C,T}
    alpha::T
    c::C

    AlphaColor(x1::T, x2::T, x3::T, alpha::T) = new(alpha, C(x1, x2, x3))
    AlphaColor(c::C, alpha::T) = new(alpha, c)
end
AlphaColor{T<:Fractional}(c::ColorValue{T}, alpha::T = one(T)) = AlphaColor{T}(c, alpha)

typealias ARGB{T} AlphaColor{RGB{T}, T}


# Little-endian RGB (useful for BGRA & Cairo)
immutable BGR{T<:Fractional} <: AbstractRGB{T}
    b::T
    g::T
    r::T

    BGR(r::Number, g::Number, b::Number) = new(b, g, r)
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

    RGB1(r::Number, g::Number, b::Number) = new(one(T), r, g, b)
end
RGB1(r::Integer, g::Integer, b::Integer) = RGB1{Float64}(r, g, b)
RGB1(r::Fractional, g::Fractional, b::Fractional) = (T = promote_type(typeof(r), typeof(g), typeof(b)); RGB1{T}(r, g, b))

immutable RGB4{T<:Fractional} <: AbstractRGB{T}
    r::T
    g::T
    b::T
    alphadummy::T

    RGB4(r::Number, g::Number, b::Number) = new(r, g, b, one(T))
end
RGB4(r::Integer, g::Integer, b::Integer) = RGB4{Float64}(r, g, b)
RGB4(r::Fractional, g::Fractional, b::Fractional) = (T = promote_type(typeof(r), typeof(g), typeof(b)); RGB4{T}(r, g, b))


# Sometimes you want to be explicit about grayscale. Also needed for GrayAlpha.
abstract AbstractGray{T} <: ColorValue{T}
immutable Gray{T<:Fractional} <: AbstractGray{T}
    val::T
end

typealias GrayAlpha{T} AlphaColorValue{Gray{T}, T}

convert(::Type{RGB}, x::Gray) = RGB(x.val, x.val, x.val)
convert{T}(::Type{RGB{T}}, x::Gray) = (g = convert(T, x.val); RGB{T}(g, g, g))

length(cv::ColorType) = div(sizeof(cv), sizeof(eltype(cv)))
# Because this can be called as `length(RGB)`, we might need to fill in a default element type.
# But the compiler chokes if we ask it to create RGB{Float64}{Float64}, even if that's inside
# the non-evaluated branch of a ternary expression, so we have to be sneaky about this.
length{CV<:ColorType}(::Type{CV}) = _length(CV, eltype(CV))
_length{CV<:ColorType}(::Type{CV}, ::Type{Any}) = length(CV{Float64})
_length{CV<:ColorType}(::Type{CV}, ::DataType)  = div(sizeof(CV), sizeof(eltype(CV)))

# Math on ColorValues
(*)(f::Real, c::RGB) = RGB(f*c.r, f*c.g, f*c.b)
(*)(c::RGB, f::Real) = (*)(f, c)
(.*)(f::Real, c::RGB) = RGB(f*c.r, f*c.g, f*c.b)
(.*)(c::RGB, f::Real) = (*)(f, c)
(/)(c::RGB, f::Real) = (1.0/f)*c
(+)(a::RGB, b::RGB) = RGB(a.r+b.r, a.g+b.g, a.b+b.b)

# To help type inference
for ACV in (ColorValue, AbstractRGB)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && isleaftype(CV{Float64})) || continue
        @eval promote_array_type{T<:Real,S<:Real}(::Type{T}, ::Type{$CV{S}}) = $CV{promote_type(T, S)}
        for AC in subtypes(AbstractAlphaColorValue)
            length(AC.parameters) == 2 || continue
            @eval promote_array_type{T<:Real,S<:Real}(::Type{T}, ::Type{$AC{$CV{S},S}}) = (TS = promote_type(T, S); $AC{$CV{TS}, TS})
        end
    end
end

end
