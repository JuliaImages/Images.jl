function Overlay(channels::Tuple{Vararg{AbstractArray}}, colors, mapi::Tuple{Vararg{MapInfo}})
    _overlay(channels, colors, map(m->(x->immap(m, x)), mapi))
end

function Overlay(channels::Tuple{Vararg{AbstractArray}},
                 colors,
                 clim = ntuple(i->(zero(eltype(channels[i])), one(eltype(channels[i]))), length(channels)))
    n = length(channels)
    for i = 1:n
        if length(clim[i]) != 2
            error("clim must be a 2-vector")
        end
    end
    fs = ntuple(i->scaleminmax(Float32, clim[i][1], clim[i][2]), n)
    _overlay(channels, colors, fs)
end

function _overlay(channels, colors, fs)
    NC = length(colors)
    length(channels) == NC || error("Number of channels must match number of colors")
    length(fs) == NC   || error("Number of MapInfo objects must match number of colors")
    channelidx = channelassign(colors)
    syms = [:zeroarray, :A, :B, :C][channelidx+1]
    depwarn("Overlay(channels, colors, ...) is deprecated, please use colorview(RGB, StackedView($(syms...))), possibly in conjunction with mappedarray", :Overlay)
    mc = [mappedarray(fs[i], channels[i]) for i = 1:NC]
    zchannels = (zeroarray, mc...)
    T = promote_type((eltype(c) for c in colors)...)
    chan = zchannels[channelidx+1]
    colorview(RGB, StackedView{T}(chan...))
end

function channelassign(colors)
    n = length(colors)
    channelidx = zeros(Int, 3)
    for i = 1:n
        col = convert(RGB, colors[i])
        setchannelidx!(channelidx, i, red(col), 1)
        setchannelidx!(channelidx, i, green(col), 2)
        setchannelidx!(channelidx, i, blue(col), 3)
    end
    channelidx
end

function setchannelidx!(channelidx, i, val, colorindex)
    if val != 0
        if channelidx[colorindex] != 0
            error("in the deprecated version, mixing in color channels is not supported, please see ??")
        end
        channelidx[colorindex] = i
    end
    channelidx
end

nchannels{C,N,A<:StackedView}(ovr::ColorView{C,N,A}) = sum(!isa(A, ImageCore.ZeroArray) for A in ovr.parent.parents)

OverlayImage(args...) = ImageMeta(Overlay(args...))
