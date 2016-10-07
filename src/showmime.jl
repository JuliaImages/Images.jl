###
### show as MIME type
###

# This is used by IJulia (for example) to display images

# mimewriteable to PNG if 2D colorant array
mimewritable{C<:Colorant}(::MIME"image/png", img::AbstractMatrix{C}) = true

# Colors.jl turns on SVG display of colors, which leads to poor
# performance and weird spacing if you're displaying images. We need
# to disable that here.
# See https://github.com/JuliaLang/IJulia.jl/issues/229 and Images #548
mimewritable{C<:Color}(::MIME"image/svg+xml", img::AbstractMatrix{C}) = false

# Really large images can make display very slow, so we shrink big
# images.  Conversely, tiny images don't show up well, so in such
# cases we repeat pixels.
function Base.show{C<:Colorant}(io::IO, mime::MIME"image/png", img::AbstractMatrix{C}; mapi=clamp01nan, minpixels=10^4, maxpixels=10^6)
    while _length(img) > maxpixels
        img = restrict(img)  # big images
    end
    npix = _length(img)
    if npix < minpixels
        # Tiny images
        fac = ceil(Int, sqrt(minpixels/npix))
        r = ones(Int, ndims(img))
        r[[coords_spatial(img)...]] = fac
        img = repeat(img, inner=r)
    end
    save(Stream(format"PNG", io), img, mapi=mapi)
end
