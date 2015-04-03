module ColorTypes

using Color, FixedPointNumbers, Compat, Base.Cartesian
import Color: Fractional, _convert

import Base: ==, abs, abs2, clamp, convert, div, isfinite, isinf,
    isnan, isless, length, one, promote_array_type, promote_rule, zero,
    trunc, floor, round, ceil, bswap,
    mod, rem

export ARGB, BGR, RGB1, RGB4, BGRA, AbstractGray, Gray, GrayAlpha, Gray24, AGray32, YIQ, AlphaColor, ColorType
export noeltype, sumsq

typealias ColorType Union(ColorValue, AbstractAlphaColorValue)

if VERSION < v"0.4.0-dev+1827"
    const unsafe_trunc = itrunc
else
    const unsafe_trunc = Base.unsafe_trunc
end

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

for CV in subtypes(AbstractRGB)
    CV == RGB && continue
    @eval begin
        convert{T}(::Type{$CV{T}}, c::$CV{T}) = c
        convert(::Type{$CV}, c::$CV) = c
        convert{T}(::Type{$CV{T}}, c::$CV) = $CV{T}(convert(T, c.r), convert(T, c.g), convert(T, c.b))
        convert{T<:Fractional}(::Type{$CV}, c::ColorValue{T}) = convert($CV{T}, c)
        convert(::Type{$CV}, c::AbstractRGB) = $CV(c.r, c.g, c.b)
        convert{T}(::Type{$CV{T}}, c::AbstractRGB) = $CV{T}(c.r, c.g, c.b)
    end
end
convert(::Type{RGB}, c::AbstractRGB) = RGB(c.r, c.g, c.b)
convert{T}(::Type{RGB{T}}, c::AbstractRGB) = RGB{T}(c.r, c.g, c.b)


# Sometimes you want to be explicit about grayscale. Also needed for GrayAlpha.
abstract AbstractGray{T} <: ColorValue{T}
immutable Gray{T<:Fractional} <: AbstractGray{T}
    val::T
end
convert{T}(::Type{Gray{T}}, x::Gray{T}) = x
convert{T,S}(::Type{Gray{T}}, x::Gray{S}) = Gray{T}(x.val)
convert{T<:Real}(::Type{T}, x::Gray) = convert(T, x.val)
convert{T}(::Type{Gray{T}}, x::Real) = Gray{T}(x)

# Rec 601 luma conversion
convert{T<:Ufixed}(::Type{Gray{T}}, x::AbstractRGB{T}) = Gray{T}(T(unsafe_trunc(FixedPointNumbers.rawtype(T), 0.299f0*reinterpret(x.r) + 0.587f0*reinterpret(x.g) + 0.114f0*reinterpret(x.b)), 0))
convert{T}(::Type{Gray{T}}, x::AbstractRGB) = convert(Gray{T}, 0.299f0*x.r + 0.587f0*x.g + 0.114f0*x.b)

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

zero{T}(::Type{GrayAlpha{T}}) = AlphaColorValue{Gray{T},T}(zero(Gray{T}),zero(T))
 one{T}(::Type{GrayAlpha{T}}) = AlphaColorValue{Gray{T},T}(one(Gray{T}),one(T))


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

# Y'CbCr
immutable YCbCr{T<:FloatingPoint} <: ColorValue{T}
    y::T
    cb::T
    cr::T

    YCbCr(y::Real, cb::Real, cr::Real) = new(y, cb, cr)
end
function YCbCr(y::FloatingPoint, cb::FloatingPoint, cr::FloatingPoint)
    T = promote_type(typeof(y), typeof(cb), typeof(cr))
    YCbCr{T}(y, cb, cr)
end

clamp{T}(c::YCbCr{T}) = YCbCr{T}(clamp(c.y, convert(T,16), convert(T,235)),
                                 clamp(c.cb, convert(T,16), convert(T,240)),
                                 clamp(c.cr, convert(T,16), convert(T,240)))

function convert{T}(::Type{YCbCr{T}}, c::AbstractRGB)
    rgb = clamp(c)
    YCbCr{T}(16+65.481*rgb.r+128.553*rgb.g+24.966*rgb.b,
             128-37.797*rgb.r-74.203*rgb.g+112*rgb.b,
             128+112*rgb.r-93.786*rgb.g-18.214*rgb.b)
end
convert{T}(::Type{YCbCr}, c::AbstractRGB{T}) = convert(YCbCr{T}, c)

function _convert{T}(::Type{RGB{T}}, c::YCbCr)
    cc = clamp(c)
    ny = cc.y - 16
    ncb = cc.cb - 128
    ncr = cc.cr - 128
    RGB{T}(0.004567ny - 1.39135e-7ncb + 0.0062586ncr,
           0.004567ny - 0.00153646ncb - 0.0031884ncr,
           0.004567ny + 0.00791058ncb - 2.79201e-7ncr)
end

# HSI
immutable HSI{T<:FloatingPoint} <: ColorValue{T}
    h::T
    s::T
    i::T

    HSI(h::Real, s::Real, i::Real) = new(h, s, i)
end
function HSI(h::FloatingPoint, s::FloatingPoint, i::FloatingPoint)
    T = promote_type(typeof(h), typeof(s), typeof(i))
    HSI{T}(h, s, i)
end

function convert{T}(::Type{HSI{T}}, c::AbstractRGB)
    rgb = clamp(c)
    α = (2rgb.r - rgb.g - rgb.b)/2
    β = 0.8660254*(rgb.g - rgb.b)
    h = atan2(β, α)
    i = (rgb.r + rgb.g + rgb.b)/3
    s = 1-min(rgb.r, rgb.g, rgb.b)/i
    s = ifelse(i > 0, s, zero(s))
    HSI{T}(h, s, i)
end
convert{T}(::Type{HSI}, c::AbstractRGB{T}) = convert(HSI{T}, c)

# TODO: HSI->RGB

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

for ACV in (ColorValue, AbstractRGB, AbstractGray, AbstractAlphaColorValue)
    for CV in subtypes(ACV)
        length(CV.parameters) == 1 || continue
        @eval noeltype{T}(::Type{$CV{T}}) = $CV
    end
end

for f in (:trunc, :floor, :round, :ceil, :abs, :abs2, :isfinite, :isnan, :isinf, :bswap)
    @eval $f{T}(g::Gray{T}) = Gray{T}($f(g.val))
    @eval @vectorize_1arg Gray $f
end
for f in (:trunc, :floor, :round, :ceil)
    @eval $f{T<:Integer}(::Type{T}, g::Gray) = Gray{T}($f(T, g.val))
    @eval $f{T<:Integer,G<:Gray,Ti}(::Type{T}, A::SparseMatrixCSC{G,Ti}) = error("not defined") # fix ambiguity warning
    # Resolve ambiguities with Compat versions
    if VERSION < v"0.3.99"
        @eval $f{T<:Integer,G<:Gray}(::Type{T}, A::AbstractArray{G,1}) = [($f)(A[i]) for i = 1:length(A)]
        @eval $f{T<:Integer,G<:Gray}(::Type{T}, A::AbstractArray{G,2}) = [($f)(A[i,j]) for i = 1:size(A,1), j = 1:size(A,2)]
    end
    # The next resolve ambiguities with floatfuncs.jl definitions
    if VERSION < v"0.4.0-dev+3847"
        @eval $f{T<:Integer,G<:Gray}(::Type{T}, A::AbstractArray{G}) = reshape([($f)(A[i]) for i = 1:length(A)], size(A))
    end
end

for f in (:mod, :rem, :mod1)
    @eval $f(x::Gray, m::Gray) = Gray($f(x.val, m.val))
end

# Return types for arithmetic operations
multype(a::Type,b::Type) = typeof(one(a)*one(b))
sumtype(a::Type,b::Type) = typeof(one(a)+one(b))
divtype(a::Type,b::Type) = typeof(one(a)/one(b))

# Math on ColorValues. These implementations encourage inlining and,
# for the case of Ufixed types, nearly halve the number of multiplications.
for CV in subtypes(AbstractRGB)
    @eval begin
        (*){R<:Real,T}(f::R, c::$CV{T}) = $CV{multype(R,T)}(f*c.r, f*c.g, f*c.b)
        (*){R<:Real,T}(f::R, c::AlphaColorValue{$CV{T},T}) =
            AlphaColorValue{$CV{multype(R,T)},multype(R,T)}(f*c.c.r, f*c.c.g, f*c.c.b, f*c.alpha)
        (*){R<:Real,T}(f::R, c::AlphaColor{$CV{T},T}) =
            AlphaColor{$CV{multype(R,T)},multype(R,T)}(f*c.c.r, f*c.c.g, f*c.c.b, f*c.alpha)
        function (*){R<:FloatingPoint,T<:Ufixed}(f::R, c::$CV{T})
            fs = f/reinterpret(one(T))
            $CV{multype(R,T)}(fs*reinterpret(c.r), fs*reinterpret(c.g), fs*reinterpret(c.b))
        end
        function (*){R<:Ufixed,T<:Ufixed}(f::R, c::$CV{T})
            fs = reinterpret(f)/widen(reinterpret(one(T)))^2
            $CV{multype(R,T)}(fs*reinterpret(c.r), fs*reinterpret(c.g), fs*reinterpret(c.b))
        end
        function (/){R<:FloatingPoint,T<:Ufixed}(c::$CV{T}, f::R)
            fs = one(R)/(f*reinterpret(one(T)))
            $CV{divtype(R,T)}(fs*reinterpret(c.r), fs*reinterpret(c.g), fs*reinterpret(c.b))
        end
        (+){S,T}(a::$CV{S}, b::$CV{T}) = $CV{sumtype(S,T)}(a.r+b.r, a.g+b.g, a.b+b.b)
        (-){S,T}(a::$CV{S}, b::$CV{T}) = $CV{sumtype(S,T)}(a.r-b.r, a.g-b.g, a.b-b.b)
        (+){S,T}(a::AlphaColorValue{$CV{S},S}, b::AlphaColorValue{$CV{T},T}) =
            AlphaColorValue{$CV{sumtype(S,T)},sumtype(S,T)}(a.c.r+b.c.r, a.c.g+b.c.g, a.c.b+b.c.b, a.alpha+b.alpha)
        (+){S,T}(a::AlphaColor{$CV{S},S}, b::AlphaColor{$CV{T},T}) =
            AlphaColor{$CV{sumtype(S,T)},sumtype(S,T)}(a.c.r+b.c.r, a.c.g+b.c.g, a.c.b+b.c.b, a.alpha+b.alpha)
        (-){S,T}(a::AlphaColorValue{$CV{S},S}, b::AlphaColorValue{$CV{T},T}) =
            AlphaColorValue{$CV{sumtype(S,T)},sumtype(S,T)}(a.c.r-b.c.r, a.c.g-b.c.g, a.c.b-b.c.b, a.alpha-b.alpha)
        (-){S,T}(a::AlphaColor{$CV{S},S}, b::AlphaColor{$CV{T},T}) =
            AlphaColor{$CV{sumtype(S,T)},sumtype(S,T)}(a.c.r-b.c.r, a.c.g-b.c.g, a.c.b-b.c.b, a.alpha-b.alpha)
        function (.+){T}(A::AbstractArray{$CV{T}}, b::AbstractRGB)
            bT = convert($CV{T}, b)
            out = similar(A)
            add!(out, A, bT)
        end
        function (.+){T}(A::Union(AbstractArray{AlphaColorValue{$CV{T},T}}, AbstractArray{AlphaColor{$CV{T},T}}), b::AbstractAlphaColorValue)
            bT = convert($CV{T}, b)
            out = similar(A)
            add!(out, A, bT)
        end
        (.+){T}(b::AbstractRGB, A::AbstractArray{$CV{T}}) = (.+)(A, b)
        (.+){T}(b::AbstractAlphaColorValue, A::Union(AbstractArray{AlphaColorValue{$CV{T},T}}, AbstractArray{AlphaColor{$CV{T},T}})) =
            (.+)(A, b)
        function (.-){T}(A::AbstractArray{$CV{T}}, b::AbstractRGB)
            bT = convert($CV{T}, b)
            out = similar(A)
            sub!(out, A, bT)
        end
        function (.-){T}(A::Union(AbstractArray{AlphaColorValue{$CV{T},T}}, AbstractArray{AlphaColor{$CV{T},T}}), b::AbstractAlphaColorValue)
            bT = convert($CV{T}, b)
            out = similar(A)
            sub!(out, A, bT)
        end
        function (.-){T}(b::AbstractRGB, A::AbstractArray{$CV{T}})
            bT = convert($CV{T}, b)
            out = similar(A)
            sub!(out, bT, A)
        end
        function (.-){T}(b::AbstractAlphaColorValue, A::Union(AbstractArray{AlphaColorValue{$CV{T},T}}, AbstractArray{AlphaColor{$CV{T},T}}))
            bT = convert($CV{T}, b)
            out = similar(A)
            sub!(out, bT, A)
        end
        function (.*){T<:Number}(A::AbstractArray{T}, b::AbstractRGB)
            bT = typeof(b*one(T))
            out = similar(A, bT)
            mul!(out, A, b)
        end
        (.*){T<:Number}(b::AbstractRGB, A::AbstractArray{T}) = A.*b
        (*){T<:Number}(A::AbstractArray{T}, b::AbstractRGB) = A.*b
        (*){T<:Number}(b::AbstractRGB, A::AbstractArray{T}) = A.*b
        isfinite{T<:Ufixed}(c::$CV{T}) = true
        isfinite{T<:Ufixed}(c::AbstractAlphaColorValue{$CV{T},T}) = true
        isfinite{T<:FloatingPoint}(c::$CV{T}) = isfinite(c.r) && isfinite(c.g) && isfinite(c.b)
        isfinite{T<:FloatingPoint}(c::AbstractAlphaColorValue{$CV{T},T}) = isfinite(c.c.r) && isfinite(c.c.g) && isfinite(c.c.b) && isfinite(c.alpha)
        isnan{T<:Ufixed}(c::$CV{T}) = false
        isnan{T<:Ufixed}(c::AbstractAlphaColorValue{$CV{T},T}) = false
        isnan{T<:FloatingPoint}(c::$CV{T}) = isnan(c.r) || isnan(c.g) || isnan(c.b)
        isnan{T<:FloatingPoint}(c::AbstractAlphaColorValue{$CV{T},T}) = isnan(c.c.r) || isnan(c.c.g) || isnan(c.c.b) || isnan(c.alpha)
        isinf{T<:Ufixed}(c::$CV{T}) = false
        isinf{T<:Ufixed}(c::AbstractAlphaColorValue{$CV{T},T}) = false
        isinf{T<:FloatingPoint}(c::$CV{T}) = isinf(c.r) || isinf(c.g) || isinf(c.b)
        isinf{T<:FloatingPoint}(c::AbstractAlphaColorValue{$CV{T},T}) = isinf(c.c.r) || isinf(c.c.g) || isinf(c.c.b) || isinf(c.alpha)
        abs(c::$CV) = abs(c.r)+abs(c.g)+abs(c.b) # should this have a different name?
        abs{T<:Ufixed}(c::$CV{T}) = float32(c.r)+float32(c.g)+float32(c.b) # should this have a different name?
        sumsq(c::$CV) = c.r^2+c.g^2+c.b^2
        sumsq{T<:Ufixed}(c::$CV{T}) = float32(c.r)^2+float32(c.g)^2+float32(c.b)^2
        one{T<:ColorType}(::T) = one(T)
        one{T}(::Type{$CV{T}}) = $CV{T}(one(T),one(T),one(T))
        one{T}(::Type{AlphaColorValue{$CV{T},T}}) = AlphaColorValue{$CV{T},T}(one(T),one(T),one(T),one(T))
        one{T}(::Type{AlphaColor{$CV{T},T}}) = AlphaColorValue{$CV{T},T}(one(T),one(T),one(T),one(T))
        zero{T<:ColorType}(::T) = zero(T)
        zero{T}(::Type{$CV{T}}) = $CV{T}(zero(T),zero(T),zero(T))
        zero{T}(::Type{AlphaColorValue{$CV{T},T}}) = AlphaColorValue{$CV{T},T}(zero(T),zero(T),zero(T),zero(T))
        zero{T}(::Type{AlphaColor{$CV{T},T}}) = AlphaColorValue{$CV{T},T}(zero(T),zero(T),zero(T),zero(T))
    end
end
(*)(c::AbstractRGB, f::Real) = (*)(f, c)
(*){CV<:AbstractRGB}(c::AlphaColorValue{CV}, f::Real) = (*)(f, c)
(*){CV<:AbstractRGB}(c::AlphaColor{CV}, f::Real) = (*)(f, c)
(.*)(f::Real, c::AbstractRGB) = (*)(f, c)
(.*){CV<:AbstractRGB}(f::Real, c::AlphaColorValue{CV}) = (*)(f, c)
(.*){CV<:AbstractRGB}(f::Real, c::AlphaColor{CV}) = (*)(f, c)
(.*)(c::AbstractRGB, f::Real) = (*)(f, c)
(.*){CV<:AbstractRGB}(c::AlphaColorValue{CV}, f::Real) = (*)(f, c)
(.*){CV<:AbstractRGB}(c::AlphaColor{CV}, f::Real) = (*)(f, c)
(/)(c::AbstractRGB, f::Real) = (one(f)/f)*c
(/){CV<:AbstractRGB}(c::AlphaColorValue{CV}, f::Real) = (one(f)/f)*c
(/){CV<:AbstractRGB}(c::AlphaColor{CV}, f::Real) = (one(f)/f)*c
(/)(c::AbstractRGB, f::Integer) = (one(eltype(c))/f)*c
(/){CV<:AbstractRGB}(c::AlphaColorValue{CV}, f::Integer) = (one(eltype(c))/f)*c
(/){CV<:AbstractRGB}(c::AlphaColor{CV}, f::Integer) = (one(eltype(c))/f)*c
(./)(c::AbstractRGB, f::Real) = (/)(c, f)
(./){CV<:AbstractRGB}(c::AlphaColorValue{CV}, f::Real) = (/)(c, f)
(./){CV<:AbstractRGB}(c::AlphaColor{CV}, f::Real) = (/)(c, f)
(+){CV<:AbstractRGB}(A::AbstractArray{CV}, b::AbstractRGB) = (.+)(A, b)
(+){CV<:AbstractRGB,T,CV2<:AbstractRGB}(A::AbstractArray{AlphaColorValue{CV,T}}, b::AbstractAlphaColorValue{CV2}) = (.+)(A, b)
(+){CV<:AbstractRGB,T,CV2<:AbstractRGB}(A::AbstractArray{AlphaColor{CV,T}}, b::AbstractAlphaColorValue{CV2}) = (.+)(A, b)
(-){CV<:AbstractRGB}(A::AbstractArray{CV}, b::AbstractRGB) = (.-)(A, b)
(-){CV<:AbstractRGB,T,CV2<:AbstractRGB}(A::AbstractArray{AlphaColorValue{CV,T}}, b::AbstractAlphaColorValue{CV2}) = (.-)(A, b)
(-){CV<:AbstractRGB,T,CV2<:AbstractRGB}(A::AbstractArray{AlphaColor{CV,T}}, b::AbstractAlphaColorValue{CV2}) = (.-)(A, b)
(+){CV<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{CV}) = (.+)(b, A)
(+){CV<:AbstractRGB,T,CV2<:AbstractRGB}(b::AbstractAlphaColorValue{CV2}, A::AbstractArray{AlphaColorValue{CV,T}}) = (.+)(A, b)
(+){CV<:AbstractRGB,T,CV2<:AbstractRGB}(b::AbstractAlphaColorValue{CV2}, A::AbstractArray{AlphaColor{CV,T}}) = (.+)(A, b)
(-){CV<:AbstractRGB}(b::AbstractRGB, A::AbstractArray{CV}) = (.-)(b, A)
(-){CV<:AbstractRGB,T,CV2<:AbstractRGB}(b::AbstractAlphaColorValue{CV2}, A::AbstractArray{AlphaColorValue{CV,T}}) = (.-)(A, b)
(-){CV<:AbstractRGB,T,CV2<:AbstractRGB}(b::AbstractAlphaColorValue{CV2}, A::AbstractArray{AlphaColor{CV,T}}) = (.-)(A, b)

# Math on Gray
for CV in subtypes(AbstractGray)
    @eval begin
        (*){R<:Real,T}(f::R, c::$CV{T}) = $CV{multype(R,T)}(f*c.val)
        (*){R<:Real,T}(f::R, c::AlphaColorValue{$CV{T},T}) = AlphaColorValue($CV{multype(R,T)}(f*c.c.val), f*c.alpha)
        (*){R<:Real,T}(f::R, c::AlphaColor{$CV{T},T}) = AlphaColor($CV{multype(R,T)}(f*c.c.val), f*c.alpha)
        (*)(c::$CV, f::Real) = (*)(f, c)
        (.*)(f::Real, c::$CV) = (*)(f, c)
        (.*)(c::$CV, f::Real) = (*)(f, c)
        (.*){T}(f::Real, c::AlphaColorValue{$CV{T},T}) = (*)(f, c)
        (.*){T}(f::Real, c::AlphaColor{$CV{T},T}) = (*)(f, c)
        (.*){T}(c::AlphaColorValue{$CV{T},T}, f::Real) = (*)(f, c)
        (.*){T}(c::AlphaColor{$CV{T},T}, f::Real) = (*)(f, c)
        (/)(c::$CV, f::Real) = (one(f)/f)*c
        (/){R<:Real,T}(c::AlphaColorValue{$CV{T},T}, f::R) = AlphaColorValue($CV{divtype(R,T)}(c.c.val/f), c.alpha/f)
        (/){R<:Real,T}(c::AlphaColor{$CV{T},T}, f::R) = AlphaColor($CV{divtype(R,T)}(c.c.val/f), c.alpha/f)
        (/)(c::$CV, f::Integer) = (one(eltype(c))/f)*c
        (./)(c::$CV, f::Real) = (/)(c, f)
        (./){T}(c::AlphaColorValue{$CV{T},T}, f::Real) = (/)(c, f)
        (./){T}(c::AlphaColor{$CV{T},T}, f::Real) = (/)(c, f)
        (+){S,T}(a::$CV{S}, b::$CV{T}) = $CV{sumtype(S,T)}(a.val+b.val)
        (+){S,T}(a::AlphaColorValue{$CV{S},S}, b::AlphaColorValue{$CV{T},T}) =
            AlphaColorValue($CV{sumtype(S,T)}(a.c.val+b.c.val), a.alpha+b.alpha)
        (+){S,T}(a::AlphaColor{$CV{S},S}, b::AlphaColor{$CV{T},T}) =
            AlphaColor($CV{sumtype(S,T)}(a.c.val+b.c.val), a.alpha+b.alpha)
        (-){S,T}(a::$CV{S}, b::$CV{T}) = $CV{sumtype(S,T)}(a.val-b.val)
        (*){S,T}(a::$CV{S}, b::$CV{T}) = $CV{multype(S,T)}(a.val*b.val)
        (+)(A::AbstractArray{$CV}, b::AbstractGray) = (.+)(A, b)
        (-)(A::AbstractArray{$CV}, b::AbstractGray) = (.-)(A, b)
        (+)(b::AbstractGray, A::AbstractArray{$CV}) = (.+)(b, A)
        (-)(b::AbstractGray, A::AbstractArray{$CV}) = (.-)(b, A)
        function (.+){T}(A::AbstractArray{$CV{T}}, b::AbstractGray)
            bT = convert($CV{T}, b)
            out = similar(A)
            add!(out, A, bT)
        end
        (.+){T}(b::AbstractGray, A::AbstractArray{$CV{T}}) = (.+)(A, b)
        function (.-){T}(A::AbstractArray{$CV{T}}, b::AbstractGray)
            bT = convert($CV{T}, b)
            out = similar(A)
            sub!(out, A, bT)
        end
        function (.-){T}(A::AbstractArray{$CV{T}}, b::AbstractGray)
            bT = convert($CV{T}, b)
            out = similar(A)
            sub!(out, A, bT)
        end
        function (.-){T}(b::AbstractGray, A::AbstractArray{$CV{T}})
            bT = convert($CV{T}, b)
            out = similar(A)
            sub!(out, bT, A)
        end
        (*)(A::AbstractArray, b::AbstractGray) = A .* b
        (*)(b::AbstractGray, A::AbstractArray) = b .* A
        (/)(A::AbstractArray, b::AbstractGray) = A ./ b
        function (.*){T,S}(A::AbstractArray{$CV{T}}, b::AbstractGray{S})
            Tout = multype(T,S)
            out = similar(A, $CV{Tout})
            mul!(out, A, b)
        end
        (.*){T,S}(b::AbstractGray{S}, A::AbstractArray{$CV{T}}) = A .* b
        function (./){T,S}(A::AbstractArray{$CV{T}}, b::AbstractGray{S})
            Tout = divtype(T,S)
            out = similar(A, $CV{Tout})
            div!(out, A, b)
        end
        isfinite{T<:Ufixed}(c::$CV{T}) = true
        isfinite{T<:FloatingPoint}(c::$CV{T}) = isfinite(c.val)
        isnan{T<:Ufixed}(c::$CV{T}) = false
        isnan{T<:FloatingPoint}(c::$CV{T}) = isnan(c.val)
        isinf{T<:Ufixed}(c::$CV{T}) = false
        isinf{T<:FloatingPoint}(c::$CV{T}) = isinf(c.val)
        abs(c::$CV) = abs(c.val) # should this have a different name?
        abs{T<:Ufixed}(c::$CV{T}) = float32(c.val) # should this have a different name?
        sumsq(c::$CV) = c.val^2
        sumsq{T<:Ufixed}(c::$CV{T}) = float32(c.val)^2

        (<)(c::$CV, r::Real) = c.val < r
        (<)(r::Real, c::$CV) = r < c.val
        isless(c::$CV, r::Real) = c.val < r
        isless(r::Real, c::$CV) = r < c.val
        (<)(a::$CV, b::AbstractGray) = a.val < b.val
    end
end
(/)(a::AbstractGray, b::AbstractGray) = a.val/b.val
div(a::AbstractGray, b::AbstractGray) = div(a.val, b.val)
(+)(a::AbstractGray, b::Number) = a.val+b
(-)(a::AbstractGray, b::Number) = a.val-b
(+)(a::Number, b::AbstractGray) = a+b.val
(-)(a::Number, b::AbstractGray) = a-b.val
(.+)(a::AbstractGray, b::Number) = a.val+b
(.-)(a::AbstractGray, b::Number) = a.val-b
(.+)(a::Number, b::AbstractGray) = a+b.val
(.-)(a::Number, b::AbstractGray) = a-b.val

@ngenerate N typeof(out) function add!{T,N}(out, A::AbstractArray{T,N}, b)
    @inbounds begin
        @nloops N i A begin
            @nref(N, out, i) = @nref(N, A, i) + b
        end
    end
    out
end
# need a separate sub! because of unsigned types
@ngenerate N typeof(out) function sub!{T,N}(out, A::AbstractArray{T,N}, b::ColorType)  # TODO: change to b::T when julia #8045 fixed
    @inbounds begin
        @nloops N i A begin
            @nref(N, out, i) = @nref(N, A, i) - b
        end
    end
    out
end
@ngenerate N typeof(out) function sub!{T,N}(out, b::ColorType, A::AbstractArray{T,N})
    @inbounds begin
        @nloops N i A begin
            @nref(N, out, i) = b - @nref(N, A, i)
        end
    end
    out
end
@ngenerate N typeof(out) function mul!{T,N}(out, A::AbstractArray{T,N}, b)
    @inbounds begin
        @nloops N i A begin
            @nref(N, out, i) = @nref(N, A, i) * b
        end
    end
    out
end
@ngenerate N typeof(out) function div!{T,N}(out, A::AbstractArray{T,N}, b)
    @inbounds begin
        @nloops N i A begin
            @nref(N, out, i) = @nref(N, A, i) / b
        end
    end
    out
end

sumsq(x::Real) = x^2

# To help type inference
for ACV in (ColorValue, AbstractRGB, AbstractGray)
    for CV in subtypes(ACV)
        (length(CV.parameters) == 1 && !(CV.abstract)) || continue
        @eval promote_array_type{T<:Real,S<:Real}(::Type{T}, ::Type{$CV{S}}) = $CV{promote_type(T, S)}
        @eval promote_rule{T<:Fractional,S<:Fractional}(::Type{$CV{T}}, ::Type{$CV{S}}) = $CV{promote_type(T, S)}
        @eval promote_rule{T<:Fractional,S<:Fractional}(::Type{T}, ::Type{$CV{S}}) = promote_type(T, S)
        for AC in subtypes(AbstractAlphaColorValue)
            (length(AC.parameters) == 2 && !(AC.abstract)) || continue
            @eval promote_array_type{T<:Real,S<:Real}(::Type{T}, ::Type{$AC{$CV{S},S}}) = (TS = promote_type(T, S); $AC{$CV{TS}, TS})
            @eval promote_rule{T<:Fractional,S<:Fractional}(::Type{$CV{T}}, ::Type{$CV{S}}) = $CV{promote_type(T, S)}
            @eval promote_rule{T<:Fractional,S<:Integer}(::Type{$CV{T}}, ::Type{S}) = $CV{promote_type(T, S)} # for Array{RGB}./Array{Int}
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
