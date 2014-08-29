# An array type for colorized overlays of grayscale images
immutable Overlay{T,N,NC,AT<:(AbstractArray...),SIT<:(ScaleInfo...)} <: AbstractArray{RGB{T},N}
    channels::AT   # this holds the grayscale arrays
    colors::NTuple{NC,RGB{T}}
    scalei::SIT

    function Overlay(channels::(AbstractArray...), colors, scalei::(ScaleInfo...))
        length(channels) == NC || error("Number of channels must match number of colors")
        length(scalei) == NC   || error("Number of ConvertInfo objects must match number of colors")
        for i = 2:NC
            size(channels[i]) == size(channels[1]) || error("All arrays must have the same size")
        end
        new(channels, colors, scalei)
    end
end
Overlay(channels::(AbstractArray...), colors::AbstractVector, scalei::(ScaleInfo...)) =
    Overlay(channels,tuple(colors...),scalei)
Overlay{NC,T}(channels::(AbstractArray...), colors::NTuple{NC,RGB{T}}, scalei::(ScaleInfo...)) =
    Overlay{T,ndims(channels[1]),NC,typeof(channels),typeof(scalei)}(channels,colors,scalei)

function Overlay(channels::(AbstractArray...), colors,
                 clim = ntuple(length(channels), i->(zero(eltype(channels[i])), one(eltype(channels[i])))))
    n = length(channels)
    for i = 1:n
        if length(clim[i]) != 2
            error("clim must be a 2-vector")
        end
    end
    scalei = ntuple(n, i->ScaleMinMax(Float32, channels[i], clim[i][1], clim[i][2]))
    Overlay(channels, colors, scalei)
end

# Returns the overlay as an image, if possible
function OverlayImage(channels::(AbstractArray...), colors::(ColorType...), arg)
    ovr = Overlay(channels, colors, arg)
    local prop
    haveprop = false
    for i = 1:length(channels)
        if isa(channels[i], AbstractImage)
            prop = copy(properties(channels[i]))
            haveprop = true
            break
        end
    end
    if !haveprop
        prop = properties(channels[1])
    end
    haskey(prop, "colorspace") && delete!(prop, "colorspace")
    haskey(prop, "colordim")   && delete!(prop, "colordim")
    haskey(prop, "limits")     && delete!(prop, "limits")
    Image(ovr, prop)
end

for NC = 1:3
    NCm = NC-1
    @eval begin
@nsplat K 1:4 function getindex{T,N,AT,SIT}(O::Overlay{T,N,$NC,AT,SIT}, indexes::NTuple{K,Real}...)
    @inbounds begin
        sc = O.scalei[$NC]
        ch = O.channels[$NC]
        cl = O.colors[$NC]
    end
    out = scale(sc, getindex(ch, indexes...)) * cl  # one of them needs a bounds-check
    @inbounds @nexprs $NCm c->begin
        sc = O.scalei[c]
        ch = O.channels[c]
        cl = O.colors[c]
        out += scale(sc, getindex(ch, indexes...)) * cl
    end
    clamp01(out)
end
    end
end

setindex!(O::Overlay, val, I::Real...) = error("Overlays are read-only. Convert to Image{RGB} to adjust values.")


#### Other Overlay support functions ####
eltype{T}(o::Overlay{T}) = RGB{T}
length(o::Overlay) = isempty(o.channels) ? 0 : length(o.channels[1])
size(o::Overlay) = isempty(o.channels) ? (0,) : size(o.channels[1])
size(o::Overlay, d::Integer) = isempty(o.channels) ? 0 : size(o.channels[1],d)
nchannels(o::Overlay) = length(o.channels)

similar(o::Overlay) = Array(eltype(o), size(o))
similar(o::Overlay, ::NTuple{0}) = Array(eltype(o), size(o))
similar{T}(o::Overlay, ::Type{T}) = Array(T, size(o))
similar{T}(o::Overlay, ::Type{T}, sz::Int64) = Array(T, sz)
similar{T}(o::Overlay, ::Type{T}, sz::Int64...) = Array(T, sz)
similar{T}(o::Overlay, ::Type{T}, sz) = Array(T, sz)

showcompact(io::IO, o::Overlay) = print(io, summary(o), " with colors ", o.colors)
