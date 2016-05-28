# An array type for colorized overlays of grayscale images
"""
```
A = Overlay(channels, colors)
A = Overlay(channels, colors, clim)
A = Overlay(channels, colors, mapi)
```

Create an `Overlay` array from grayscale channels.  `channels = (channel1,
channel2, ...)`, `colors` is a vector or tuple of `Color`s, and `clim` is a
vector or tuple of min/max values, e.g., `clim = ((min1,max1),(min2,max2),...)`.
Alternatively, you can supply a list of `MapInfo` objects.

See also: `OverlayImage`.
"""
immutable Overlay{C<:RGB,N,NC,AT<:Tuple{Vararg{AbstractArray}},MITypes<:Tuple{Vararg{MapInfo}}} <: AbstractArray{C,N}
    channels::AT   # this holds the grayscale arrays
    colors::NTuple{NC,C}
    mapi::MITypes

    function Overlay(channels::Tuple{Vararg{AbstractArray}}, colors, mapi::Tuple{Vararg{MapInfo}})
        length(channels) == NC || error("Number of channels must match number of colors")
        length(mapi) == NC   || error("Number of ConvertInfo objects must match number of colors")
        for i = 2:NC
            size(channels[i]) == size(channels[1]) || error("All arrays must have the same size")
        end
        new(channels, colors, mapi)
    end
end
Overlay(channels::Tuple{Vararg{AbstractArray}}, colors::AbstractVector, mapi::Tuple{Vararg{MapInfo}}) =
    Overlay(channels,tuple(colors...),mapi)
Overlay{NC,C<:RGB}(channels::Tuple{Vararg{AbstractArray}}, colors::NTuple{NC,C}, mapi::Tuple{Vararg{MapInfo}}) =
    Overlay{C,ndims(channels[1]),NC,typeof(channels),typeof(mapi)}(channels,colors,mapi)

function Overlay(channels::Tuple{Vararg{AbstractArray}}, colors,
                 clim = ntuple(i->(zero(eltype(channels[i])), one(eltype(channels[i]))), length(channels)))
    n = length(channels)
    for i = 1:n
        if length(clim[i]) != 2
            error("clim must be a 2-vector")
        end
    end
    mapi = ntuple(i->ScaleMinMax(Float32, channels[i], clim[i][1], clim[i][2]), n)
    Overlay(channels, colors, mapi)
end

# Returns the overlay as an image, if possible
"`OverlayImage` is identical to `Overlay`, except that it returns an Image."
function OverlayImage(channels::Tuple{Vararg{AbstractArray}}, colors::Tuple{Vararg{Colorant}},
            arg = ntuple(i->(zero(eltype(channels[i])), one(eltype(channels[i]))), length(channels)))
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
    for K = 1:4
        indexargs = Symbol[Symbol("i_", d) for d = 1:K]
        sigargs = Expr[:($a::Integer) for a in indexargs]
        @eval begin
            function getindex{T,N,AT,MITypes}(O::Overlay{T,N,$NC,AT,MITypes}, $(sigargs...))
                # one of them needs a bounds-check
                out = map(O.mapi[$NC], getindex(O.channels[$NC], $(indexargs...))) * O.colors[$NC]
                @inbounds @nexprs $NCm c->begin
                    out += map(O.mapi[c], getindex(O.channels[c], $(indexargs...))) * O.colors[c]
                end
                clamp01(eltype(O), out)
            end
        end
    end
end

setindex!(O::Overlay, val, I::Real...) = error("Overlays are read-only. Convert to Image{RGB} to adjust values.")


#### Other Overlay support functions ####
length(o::Overlay) = isempty(o.channels) ? 0 : length(o.channels[1])
size(o::Overlay) = isempty(o.channels) ? (0,) : size(o.channels[1])
size(o::Overlay, d::Integer) = isempty(o.channels) ? 0 : size(o.channels[1],d)
nchannels(o::Overlay) = length(o.channels)

similar{T}(o::Overlay, ::Type{T}, sz::Dims) = Array(T, sz)

showcompact(io::IO, o::Overlay) = print(io, summary(o), " with colors ", o.colors)
