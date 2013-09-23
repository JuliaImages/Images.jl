#### Scaling/clipping/type conversion ####

function scale{T}(scalei::ScaleInfo{T}, img::Union(StridedArray,AbstractImageDirect))
    out = similar(img, T)
    scale!(out, scalei, img)
end

function scale!{T}(out, scalei::ScaleInfo{T}, img::Union(StridedArray,AbstractImageDirect))
    dimg = data(img)
    dout = data(out)
    for i = 1:length(dimg)
        dout[i] = scale(scalei, dimg[i])  # FIXME subarrays?
    end
    out
end

type ScaleNone{T} <: ScaleInfo{T}; end

scale{T<:Number}(scalei::ScaleNone{T}, val::T) = val
scale{T,S<:Number}(scalei::ScaleNone{T}, val::S) = convert(T, val)
scale{T<:Integer,S<:FloatingPoint}(scalei::ScaleNone{T}, val::S) = convert(T, round(val))
scale(scalei::ScaleNone{Uint8}, val::Uint16) = convert(Uint8, val>>>8)
scale(scalei::ScaleNone{Uint32}, val::RGB) = convert(Uint32, convert(RGB24, clip(val)))

clip(v::RGB) = RGB(min(1.0,v.r),min(1.0,v.g),min(1.0,v.b))
function clip!(A::Array{RGB})
    for i = 1:length(A)
        A[i] = clip(A[i])
    end
    A
end

type BitShift{T,N} <: ScaleInfo{T} end

scale{T,N}(scalei::BitShift{T,N}, val::Integer) = convert(T, val>>>N)

# The Clip types just enforce bounds, but do not scale or
# subtract the minimum
type ClipMin{T,From} <: ScaleInfo{T}
    min::From
end
ClipMin{T,From}(::Type{T}, min::From) = ClipMin{T,From}(min)
type ClipMax{T,From} <: ScaleInfo{T}
    max::From
end
ClipMax{T,From}(::Type{T}, max::From) = ClipMax{T,From}(max)
type ClipMinMax{T,From} <: ScaleInfo{T}
    min::From
    max::From
end
ClipMinMax{T,From}(::Type{T}, min::From, max::From) = ClipMinMax{T,From}(min,max)

scale{T<:Number}(scalei::ClipMin{T,T}, val::T) = max(val, scalei.min)
scale(scalei::ClipMin{Uint8,Uint16}, val::Uint16) = (max(val, scalei.min)>>>8) & 0xff
scale{T<:Number}(scalei::ClipMax{T,T}, val::T) = min(val, scalei.max)
scale(scalei::ClipMax{Uint8,Uint16}, val::Uint16) = (min(val, scalei.max)>>>8) & 0xff
scale{T<:Number}(scalei::ClipMinMax{T,T}, val::T) = min(max(val, scalei.min), scalei.max)
scale{T<:Number,F<:Number}(scalei::ClipMinMax{T,F}, val::F) = convert(T,min(max(val, scalei.min), scalei.max))
# scale(scalei::ClipMinMax{Uint8,Uint16}, val::Uint16) = (min(max(val, scalei.min), scalei.max)>>>8) & 0xff

# This scales and subtracts the min value
type ScaleMinMax{To,From} <: ScaleInfo{To}
    min::From
    max::From
    s::Float64
end

function scale{To<:Integer,From<:Number}(scalei::ScaleMinMax{To,From}, val::From)
    # Clip to range min:max and subtract min
    t::From = (val > scalei.min) ? ((val < scalei.max) ? val-scalei.min : scalei.max-scalei.min) : zero(From)
    convert(To, iround(scalei.s*t))
end

function scale{To<:Number,From<:Number}(scalei::ScaleMinMax{To,From}, val::From)
    t::From = (val > scalei.min) ? ((val < scalei.max) ? val-scalei.min : scalei.max-scalei.min) : zero(From)
    convert(To, scalei.s*t)
end

function scaleinfo{To<:Unsigned,From<:Unsigned}(::Type{To}, img::AbstractArray{From})
    l = limits(img)
    if l[1] == typemin(From) && l[2] == typemax(From)
        return ScaleNone{To}()
    end
    ScaleMinMax{To,From}(l[1],l[2],typemax(To)/(l[2]-l[1]))
end

function scaleinfo{To<:Unsigned,From<:FloatingPoint}(::Type{To}, img::AbstractArray{From})
    l = limits(img)
    if !isinf(l[1]) && !isinf(l[2])
        return ScaleMinMax{To,From}(l[1],l[2],typemax(To)/(l[2]-l[1]))
    else
        return ScaleNone{To}()
    end
end

function scaleinfo(::Type{RGB}, img::AbstractArray)
    l = limits(img)
    if !isinf(l[1]) && !isinf(l[2])
        return ScaleMinMax{Float64,eltype(img)}(l[1],l[2],1.0/(l[2]-l[1]))
    else
        return ScaleNone{Float64}()
    end
end

# Multiplies by a scaling factor and then clips to the range [-1,1].
# Intended for positive/negative coloring
type ScaleSigned <: ScaleInfo{Float64}
    s::Float64
end

function scale(scalei::ScaleSigned, val::Real)
    sval::Float64 = scalei.s*val
    return sval>1.0 ? 1.0 : (sval<-1.0 ? -1.0 : sval)
end

scaledefault{T<:Unsigned}(img::AbstractArray{T}) = limits(img)
function scaledefault{T<:FloatingPoint}(img::AbstractArray{T})
    l = limits(img)
    if isinf(l[1]) || isinf(l[2])
        if isa(l, Tuple)
            l = (0.0,255.0)
        else
            l[1] = 0
            l[2] = 255
        end
    end
    l
end

minfinite(A::AbstractArray) = min(A)
function minfinite{T<:FloatingPoint}(A::AbstractArray{T})
    ret = nan(T)
    for a in A
        ret = isfinite(a) ? (ret < a ? ret : a) : ret
    end
    ret
end

maxfinite(A::AbstractArray) = max(A)
function maxfinite{T<:FloatingPoint}(A::AbstractArray{T})
    ret = nan(T)
    for a in A
        ret = isfinite(a) ? (ret > a ? ret : a) : ret
    end
    ret
end

scaleminmax{To<:Integer,From}(::Type{To}, img::AbstractArray{From}, mn::Number, mx::Number) = ScaleMinMax{To,From}(convert(From,mn), convert(From,mx), float64(typemax(To)/(mx-mn)))
scaleminmax{To<:FloatingPoint,From}(::Type{To}, img::AbstractArray{From}, mn::Number, mx::Number) = ScaleMinMax{To,From}(convert(From,mn), convert(From,mx), 1.0/(mx-mn))
scaleminmax{To}(::Type{To}, img::AbstractArray) = scaleminmax(To, img, minfinite(img), maxfinite(img))
scaleminmax(img::AbstractArray) = scaleminmax(Uint8, img)
scaleminmax(img::AbstractArray, mn::Number, mx::Number) = scaleminmax(Uint8, img, mn, mx)
scaleminmax{To<:Number,From<:Number}(::Type{To}, mn::From, mx::From) = ScaleMinMax{To,From}(mn, mx, 255.0/(mx-mn))

sc(img::AbstractArray) = scale(scaleminmax(img), img)
sc(img::AbstractArray, mn::Number, mx::Number) = scale(scaleminmax(img, mn, mx), img)

convert{I<:AbstractImageDirect}(::Type{I}, img::Union(StridedArray,AbstractImageDirect)) = scale(ScaleNone{eltype(I)}(), img)

function convert(::Type{Image{RGB}}, img::Union(StridedArray,AbstractImageDirect))
    cs = colorspace(img)
    if !(cs == "RGB" || cs == "RGBA")
        error("Only RGB and RGBA colorspaces supported currently")
    end
    scalei = scaleinfo(RGB, img)
    cd = colordim(img)
    d = data(img)
    p = parent(d)
    sz = size(img)
    szout = sz[setdiff(1:ndims(img), cd)]
    dout = Array(RGB, szout)
    if colordim(img) == 1
        s = stride(d,2)
        for i in 0:length(dout)-1
            dout[i+1] = RGB(scale(scalei,d[i*s+1]), scale(scalei,d[i*s+2]), scale(scalei,d[i*s+3]))
        end
    elseif cd == ndims(img)
        s = stride(d,cd)
        for i in 1:length(dout)
            dout[i] = RGB(scale(scalei,d[i]), scale(scalei,d[i+s]), scale(scalei,d[i+2s]))
        end
    else
        error("Not yet implemented")
    end
    p = properties(img)
    delete!(p, "colordim")
    delete!(p, "limits")
    delete!(p, "colorspace")
    Image(dout, p)
end
