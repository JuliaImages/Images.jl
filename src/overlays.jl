# An array type for colorized overlays of grayscale images
immutable Overlay{T,N,NC,AT<:(@compat Tuple{Vararg{AbstractArray}}),MITypes<:(@compat Tuple{Vararg{MapInfo}})} <: AbstractArray{RGB{T},N}
    channels::AT   # this holds the grayscale arrays
    colors::NTuple{NC,RGB{T}}
    mapi::MITypes

    function Overlay(channels::(@compat Tuple{Vararg{AbstractArray}}), colors, mapi::(@compat Tuple{Vararg{MapInfo}}))
        length(channels) == NC || error("Number of channels must match number of colors")
        length(mapi) == NC   || error("Number of ConvertInfo objects must match number of colors")
        for i = 2:NC
            size(channels[i]) == size(channels[1]) || error("All arrays must have the same size")
        end
        new(channels, colors, mapi)
    end
end
Overlay(channels::(@compat Tuple{Vararg{AbstractArray}}), colors::AbstractVector, mapi::(@compat Tuple{Vararg{MapInfo}})) =
    Overlay(channels,tuple(colors...),mapi)
Overlay{NC,T}(channels::(@compat Tuple{Vararg{AbstractArray}}), colors::NTuple{NC,RGB{T}}, mapi::(@compat Tuple{Vararg{MapInfo}})) =
    Overlay{T,ndims(channels[1]),NC,typeof(channels),typeof(mapi)}(channels,colors,mapi)

function Overlay(channels::(@compat Tuple{Vararg{AbstractArray}}), colors,
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
function OverlayImage(channels::(@compat Tuple{Vararg{AbstractArray}}), colors::(@compat Tuple{Vararg{Paint}}), arg)
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
        indexargs = Symbol[symbol(string("i_",d)) for d = 1:K]
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
