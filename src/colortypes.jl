module ColorTypes

using Color
import Color.Fractional

export RGBA, ARGB, BGRA, GrayAlpha

typealias RGBA{T} AlphaColorValue{RGB{T}, T}

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


# For encoding GrayAlpha
immutable Gray{T<:Fractional} <: ColorValue{T}
    val::T
end

typealias GrayAlpha{T} AlphaColorValue{Gray{T}, T}

end
