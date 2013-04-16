# Display images so they fill a window

# TkRenderer adapted from Winston's tk.jl
import Tk
import Cairo

function TkRenderer(name, w, h, closecb=nothing)
    win = Tk.Window(name, w, h)
    c = Tk.Canvas(win)
    Tk.pack(c, {:expand => true, :fill => "both"})
    if !is(closecb,nothing)
        ccb = Tk.tcl_callback(closecb)
        Tk.tcl_eval("bind $(win.path) <Destroy> $ccb")
    end
    c.redraw = function (_)
#         r = Tk.getgc(c)
#         set_source_rgb(r, 0, 0, 0)
#         paint(r)
        Tk.reveal(c)
        Tk.tcl_doevent()
    end
    c
end

# Create a window that "views" an in-memory buffer
# This does no transposition, that should be handled by the caller (or see copyt! below)
type WindowImage
    c::Tk.Canvas
    surf::Cairo.CairoSurface
    buf::Array{Uint32,2}

    function WindowImage(buf::Array{Uint32,2}, format::Integer, title::String)
        w, h = size(buf)    # note it's in [x,y] order, not [row,col] order!
        c = TkRenderer(title, w, h)
        surf = Cairo.CairoImageSurface(buf, format, w, h)
        Cairo.image(Tk.getgc(c), surf, 0, 0, w, h)
        c.redraw(c)
        obj = new(c, surf, buf)
        # Set up the resize callback
        rcb = Tk.tcl_callback((path) -> resize(obj))
        Tk.tcl_eval("bind $(c.c.path) <Configure> {$rcb}")
        obj
    end
end
WindowImage(buf::Array{Uint32,2}, format::Integer) = WindowImage(buf, format, "Julia")
WindowImage(buf::Array{Uint32,2}) = WindowImage(buf, Cairo.CAIRO_FORMAT_RGB24, "Julia")

show(io::IO, wb::WindowImage) = print(io, "WindowImage with buffer size ", Base.dims2string(size(wb.buf)))

function resize(wb::WindowImage)
    f = wb.c.c
    win = f.parent
    w = Tk.width(win)
    h = Tk.height(win)
    Tk.configure(wb.c)
    Cairo.image(Tk.getgc(wb.c), wb.surf, 0, 0, w, h)
    wb.c.redraw(wb.c)
end

# If you write new data to the buffer (e.g., using copy!), refresh the display
function update(wb::WindowImage)
    Cairo.image(wb.r, wb.surf, 0, 0, Cairo.width(wb.surf), Cairo.height(wb.surf))
    wb.r.on_close()
end

copy!(wb::WindowImage, data::Array{Uint32,2}) = copy!(wb.buf, data)
fill!(wb::WindowImage, val::Uint32) = fill!(wb.buf, val)

# Copy-with-transpose
function copyt!(buf::Array{Uint32,2}, data::Array{Uint32,2})
    h, w = size(data)
    if size(buf,1) != w || size(buf,2) != h
        error("Size mismatch")
    end
    for j = 1:w, i = 1:h
        buf[j,i] = data[i,j]
    end
    buf
end

#### A demo  ####
# w = 400
# h = 200
# buf = zeros(Uint32,w,h)
# fill!(buf,0x00FF0000)  # red
# wb = WindowImage(buf)
#
# sleep(1)
#
# for val = 0x00000000:0x000000FF
#     fill!(wb, val)
#     update(wb)
# end



# Now add some code to display Images
# display(r::Cairo.CairoRenderer, img::AbstractArray, x, y, w, h)
#     buf = cairoRGB(img)
#     imw, imh = size(buf)
#     surf = Cairo.CairoImageSurface(buf, format, imw, imh)
#     Cairo.image(r, surf, x, y, w, h)
#     r.on_close()
# end
# display(r::Cairo.CairoRenderer, img::AbstractArray) = display(r, img, r.lowerleft[1], r.lowerleft[2], 

# display in a previous window
function display(wb::WindowImage, img::AbstractArray, scalei::ScaleInfo)
    cairoRGB(wb.buf, img, scalei)
    update(wb)
    wb
end
display(wb::WindowImage, img::AbstractArray) = display(wb, img, scaleinfo(Uint8, img))

# display in a new window
function display(img::AbstractArray, scalei::ScaleInfo)
    buf, format = cairoRGB(img, scalei)
    WindowImage(buf, format)
end
display(img::AbstractArray) = display(img, scaleinfo(Uint8, img))


# Efficient conversions to RGB24 or ARGB32
function cairoRGB(img::Union(StridedArray,AbstractImageDirect), scalei::ScaleInfo)
    w, h = widthheight(img)
    buf = Array(Uint32, w, h)
    format = cairoRGB(buf, img, scalei)
    buf, format
end

function cairoRGB(buf::Array{Uint32,2}, img::Union(StridedArray,AbstractImageDirect), scalei::ScaleInfo)
    assert2d(img)
    cs = colorspace(img)
    xfirst = isxfirst(img)
    firstindex, spsz, spstride, csz, cstride = iterate_spatial(img)
    isz, jsz = spsz
    istride, jstride = spstride
    A = parent(img)
    if xfirst
        w, h = isz, jsz
    else
        w, h = jsz, isz
    end
    if size(buf, 1) != w || size(buf, 2) != h
        error("Output buffer is of the wrong size")
    end
    # Check to see whether we can do a direct copy
    if eltype(img) <: Union(Uint32, Int32)
        if cs == "RGB24"
            if xfirst
                copy!(buf, img.data)
            else
                copyt!(buf, img.data)
            end
            return Cairo.CAIRO_FORMAT_RGB24
        elseif cs == "ARGB32"
            if xfirst
                copy!(buf, img.data)
            else
                copyt!(buf, img.data)
            end
            return Cairo.CAIRO_FORMAT_ARGB32
        end
    end
    local format
    if cstride == 0
        if cs == "Gray"
            if xfirst
                # Note: can't use a single linear index for RHS, because this might be a subarray
                l = 1
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 0:istride:(isz-1)*istride
                        gr = scale(scalei, A[k+i])
                        buf[l] = rgb24(gr, gr, gr)
                        l += 1
                    end
                end
            else
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 1:isz
                        gr = scale(scalei, A[k+(i-1)*istride])
                        buf[j,i] = rgb24(gr, gr, gr)
                    end
                end
            end
            format = Cairo.CAIRO_FORMAT_RGB24
        else
            error("colorspace ", cs, " not yet supported")
        end
    else
        if cs == "RGB"
            if xfirst
                l = 1
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 0:istride:(isz-1)*istride
                        ki = k+i
                        buf[l] = rgb24(scalei, A[ki], A[ki+cstride], A[ki+2cstride])
                        l += 1
                    end
                end
            else
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 1:isz
                        ki = k+(i-1)*istride
                        buf[j,i] = rgb24(scalei, A[ki], A[ki+cstride], A[ki+2cstride])
                    end
                end
            end
            format = Cairo.CAIRO_FORMAT_RGB24
        elseif cs == "ARGB"
            if xfirst
                l = 1
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 0:istride:(isz-1)*istride
                        ki = k+i
                        buf[l] = argb32(scalei,A[ki],A[ki+cstride],A[ki+2cstride],A[ki+3cstride])
                        l += 1
                    end
                end
            else
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 1:isz
                        ki = k+(i-1)*istride
                        buf[j,i] = argb32(scalei,A[ki],A[ki+cstride],A[ki+2cstride],A[ki+3cstride])
                    end
                end
            end
            format = Cairo.CAIRO_FORMAT_ARGB32
        elseif cs == "RGBA"
            if xfirst
                l = 1
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 0:istride:(isz-1)*istride
                        ki = k+i
                        buf[l] = argb32(scalei,A[ki+3cstride],A[ki],A[ki+cstride],A[ki+2cstride])
                        l += 1
                    end
                end
            else
                for j = 1:jsz
                    k = firstindex + (j-1)*jstride
                    for i = 1:isz
                        ki = k+(i-1)*istride
                        buf[j,i] = argb32(scalei,A[ki+3cstride],A[ki],A[ki+cstride],A[ki+2cstride])
                    end
                end
            end
            format = Cairo.CAIRO_FORMAT_ARGB32
        else
            error("colorspace ", cs, " not yet supported")
        end
    end
    format
end

rgb24(r::Uint8, g::Uint8, b::Uint8) = convert(Uint32,r)<<16 + convert(Uint32,g)<<8 + convert(Uint32,b)

argb32(a::Uint8, r::Uint8, g::Uint8, b::Uint8) = convert(Uint32,a)<<24 + convert(Uint32,r)<<16 + convert(Uint32,g)<<8 + convert(Uint32,b)

rgb24{T}(scalei::ScaleInfo{Uint8}, r::T, g::T, b::T) = convert(Uint32,scale(scalei,r))<<16 + convert(Uint32,scale(scalei,g))<<8 + convert(Uint32,scale(scalei,b))

argb32{T}(scalei::ScaleInfo{Uint8}, a::T, r::T, g::T, b::T) = convert(Uint32,scale(scalei,a))<<24 + convert(Uint32,scale(scalei,r))<<16 + convert(Uint32,scale(scalei,g))<<8 + convert(Uint32,scale(scalei,b))



# External-viewer interface
function imshow(img, range)
    if ndims(img) == 2 
        # only makes sense for gray scale images
        img = imadjustintensity(img, range)
    end
    tmp::String = "./tmp.ppm"
    imwrite(img, tmp)
    cmd = `feh $tmp`
    spawn(cmd)
end

imshow(img) = imshow(img, [])

# 'illustrates' fourier transform
ftshow{T}(A::Array{T,2}) = imshow(log(1+abs(fftshift(A))),[])

