###
### writemime
###
# only mime writeable to PNG if 2D (used by IJulia for example)
mimewritable(::MIME"image/png", img::AbstractImage) = sdims(img) == 2 && timedim(img) == 0
mimewritable{C<:Colorant}(::MIME"image/png", img::AbstractArray{C}) = sdims(img) == 2 && timedim(img) == 0
# Colors.jl turns on SVG display of colors, which leads to poor
# performance and weird spacing if you're displaying images. We need
# to disable that here.
# See https://github.com/JuliaLang/IJulia.jl/issues/229 and Images #548
mimewritable(::MIME"image/svg+xml", img::AbstractImage)  = false
mimewritable{C<:Color}(::MIME"image/svg+xml", img::AbstractMatrix{C}) = false

# This is used for output by IJulia. Really large images can make
# display very slow, so we shrink big images.  Conversely, tiny images
# don't show up well, so in such cases we repeat pixels.
@compat function Base.show(io::IO, mime::MIME"image/png", img::AbstractImage; mapi=mapinfo_writemime(img), minpixels=10^4, maxpixels=10^6)
    assert2d(img)
    A = data(img)
    nc = ncolorelem(img)
    npix = length(A)/nc
    while npix > maxpixels
        # Big images
        A = restrict(A, coords_spatial(img))
        npix = length(A)/nc
    end
    if npix < minpixels
        # Tiny images
        fac = ceil(Int, sqrt(minpixels/npix))
        r = ones(Int, ndims(img))
        r[coords_spatial(img)] = fac
        A = repeat(A, inner=r)
    end
    imgcopy = shareproperties(img, A)
    save(Stream(format"PNG", io), imgcopy)
end

@compat function Base.show(stream::IO, mime::MIME"image/png", img::AbstractImageIndexed; kwargs...)
    @compat show(stream, mime, convert(Image, img); kwargs...)
end
@compat function Base.show{C<:Colorant}(stream::IO, mime::MIME"image/png", img::AbstractMatrix{C}; kwargs...)
    @compat show(stream, mime, Image(img, spatialorder=["y","x"]); kwargs...)
end

function mapinfo_writemime(img; maxpixels=10^6)
    if length(img) <= maxpixels
        return mapinfo_writemime_(img)
    end
    mapinfo_writemime_restricted(img)
end

to_native_color{T<:Colorant}(::Type{T}) = base_color_type(T){UFixed8}
to_native_color{T<:Color}(::Type{T}) = RGB{UFixed8}
to_native_color{T<:TransparentColor}(::Type{T}) = RGBA{UFixed8}

mapinfo_writemime_{T <:Colorant}(img::AbstractImage{T}) = Images.mapinfo(to_native_color(T), img)
mapinfo_writemime_(img::AbstractImage) = Images.mapinfo(UFixed8,img)

mapinfo_writemime_restricted{T<:Colorant}(img::AbstractImage{T}) = ClampMinMax(to_native_color(T), 0.0, 1.0)
mapinfo_writemime_restricted(img::AbstractImage) = Images.mapinfo(UFixed8, img)

function Base.show{T<:Image}(io::IOContext, m::MIME"text/html", imgs::AbstractMatrix{T})
    m,n=size(imgs)
    write(io, "<table>")
    for i = 1:m
        write(io, "<tr style='border:none; '>")
        for j = 1:n
            write(io, "<td style='text-align:center;vertical-align:middle; margin: 0.5em;border:1px #90999f solid;border-collapse:collapse'>")
            show_element(IOContext(io, thumbnail=true), imgs[i,j])
            write(io, "</td>")
        end
        write(io, "</tr>")
    end
    write(io, "</table>")
end

function downsize_for_thumbnail(img, w, h)
    a,b=size(img)
    a > 2w && b > 2h ?
        downsize_for_thumbnail(restrict(img), w, h) : img
end

function show_element(io::IOContext, img::Image)
    io2=IOBuffer()
    w,h=get(io, :thumbnailsize, (100,100))
    im_resized = downsize_for_thumbnail(img, w, h)
    thumbnail_style = get(io, :thumbnail, false) ? "max-width: $(w)px; max-height:$(h)px;" : ""
    b64pipe=Base64EncodePipe(io2)
    write(io,"<img style='$(thumbnail_style)display:inline' src=\"data:image/png;base64,")
    show(b64pipe, MIME"image/png"(), im_resized)
    write(io, read(seekstart(io2)))
    write(io,"\">")
end
