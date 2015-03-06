### Edge and Gradient related tests ###

using Images, Base.Test, Color

EPS = 1e-14

## Checkerboard array, used to test image gradients

let
    white{T}(::Type{T}) = one(T)
    black{T}(::Type{T}) = zero(T)
    white{T<:Unsigned}(::Type{T}) = typemax(T)
    black{T<:Unsigned}(::Type{T}) = typemin(T)

    global checkerboard
    function checkerboard{T}(::Type{T}, sq_width::Integer, count::Integer)
        wh = fill(white(T), (sq_width,sq_width))
        bk = fill(black(T), (sq_width,sq_width))
        bw = [wh bk; bk wh]
        vert = repmat(bw, (count>>1), 1)
        isodd(count) && (vert = vcat(vert, [wh bk]))
        cb = repmat(vert, 1, (count>>1))
        isodd(count) && (cb = hcat(cb, vert[:,1:sq_width]))
        cb
    end

    checkerboard(sq_width::Integer, count::Integer) = checkerboard(Uint8, sq_width, count)
end

SZ=5

cb_array    = checkerboard(SZ,3)
cb_image_xy = grayim(cb_array)
cb_image_yx = grayim(cb_array)
cb_image_yx["spatialorder"] = ["y","x"]

cb_image_gray = convert(Image{Gray}, cb_image_xy)
cb_image_rgb = convert(Image{RGB}, cb_image_xy)
cb_image_rgb2 = convert(Image{RGB{Float64}}, cb_image_xy)

for method in ["sobel", "prewitt", "ando3", "ando4", "ando5", "ando4_sep", "ando5_sep"]
    ## Checkerboard array

    (agx, agy) = imgradients(cb_array, method)
    amag = magnitude(agx, agy)
    agphase = phase(agx, agy)
    @test (amag, agphase) == magnitude_phase(agx, agy)

    @test agx[1,SZ]   < 0.0   # white to black transition
    @test agx[1,2*SZ] > 0.0   # black to white transition
    @test agy[SZ,1]   < 0.0   # white to black transition
    @test agy[2*SZ,1] > 0.0   # black to white transition

    # Test direction of increasing gradient
    @test cos(agphase[1,SZ])   - (-1.0) < EPS   # increasing left  (=  pi   radians)
    @test cos(agphase[1,2*SZ]) -   1.0  < EPS   # increasing right (=   0   radians)
    @test sin(agphase[SZ,1])   -   1.0  < EPS   # increasing up    (=  pi/2 radians)
    @test sin(agphase[2*SZ,1]) - (-1.0) < EPS   # increasing down  (= -pi/2 radians)

    # Test that orientation is perpendicular to gradient
    aorient = orientation(agx, agy)
    @test all((cos(agphase).*cos(aorient) .+ sin(agphase).*sin(aorient) .< EPS) |
                ((agphase .== 0.0) & (aorient .== 0.0)))  # this part is where both are
                                                          # zero because there is no gradient

    ## Checkerboard Image with row major order

    (gx, gy) = imgradients(cb_image_xy, method)
    mag = magnitude(gx, gy)
    gphase = phase(gx, gy)
    @test (mag, gphase) == magnitude_phase(gx, gy)

    @test gx[SZ,1]   < 0.0   # white to black transition
    @test gx[2*SZ,1] > 0.0   # black to white transition
    @test gy[1,SZ]   < 0.0   # white to black transition
    @test gy[1,2*SZ] > 0.0   # black to white transition

    @test cos(gphase[SZ,1])   - (-1.0) < EPS   # increasing left  (=  pi   radians)
    @test cos(gphase[2*SZ,1]) -   1.0  < EPS   # increasing right (=   0   radians)
    @test sin(gphase[1,SZ])   -   1.0  < EPS   # increasing up    (=  pi/2 radians)
    @test sin(gphase[1,2*SZ]) - (-1.0) < EPS   # increasing down  (= -pi/2 radians)

    # Test that orientation is perpendicular to gradient
    orient = orientation(gx, gy)
    @test all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                ((gphase .== 0.0) & (orient .== 0.0)))  # this part is where both are
                                                        # zero because there is no gradient

    ## Checkerboard Image with column-major order

    (gx, gy) = imgradients(cb_image_yx, method)
    mag = magnitude(gx, gy)
    gphase = phase(gx, gy)
    @test (mag, gphase) == magnitude_phase(gx, gy)

    @test gx[1,SZ]   < 0.0   # white to black transition
    @test gx[1,2*SZ] > 0.0   # black to white transition
    @test gy[SZ,1]   < 0.0   # white to black transition
    @test gy[2*SZ,1] > 0.0   # black to white transition

    # Test direction of increasing gradient
    @test cos(gphase[1,SZ])   - (-1.0) < EPS   # increasing left  (=  pi   radians)
    @test cos(gphase[1,2*SZ]) -   1.0  < EPS   # increasing right (=   0   radians)
    @test sin(gphase[SZ,1])   -   1.0  < EPS   # increasing up    (=  pi/2 radians)
    @test sin(gphase[2*SZ,1]) - (-1.0) < EPS   # increasing down  (= -pi/2 radians)

    # Test that orientation is perpendicular to gradient
    orient = orientation(gx, gy)
    @test all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                ((gphase .== 0.0) & (orient .== 0.0)))  # this part is where both are
                                                        # zero because there is no gradient

    ## Checkerboard Image with Gray pixels

    (gxg, gyg) = imgradients(cb_image_gray, method)
    magg = magnitude(gxg, gyg)
    gphaseg = phase(gxg, gyg)
    @test (magg, gphaseg) == magnitude_phase(gxg, gyg)

    @test gxg[SZ,1]   < 0.0   # white to black transition
    @test gxg[2*SZ,1] > 0.0   # black to white transition
    @test gyg[1,SZ]   < 0.0   # white to black transition
    @test gyg[1,2*SZ] > 0.0   # black to white transition

    @test cos(gphaseg[SZ,1])   - (-1.0) < EPS   # increasing left  (=  pi   radians)
    @test cos(gphaseg[2*SZ,1]) -   1.0  < EPS   # increasing right (=   0   radians)
    @test sin(gphaseg[1,SZ])   -   1.0  < EPS   # increasing up    (=  pi/2 radians)
    @test sin(gphaseg[1,2*SZ]) - (-1.0) < EPS   # increasing down  (= -pi/2 radians)

    # Test that orientation is perpendicular to gradient
    orientg = orientation(gxg, gyg)
    @test all((cos(gphaseg).*cos(orientg) .+ sin(gphaseg).*sin(orientg) .< EPS) |
                ((gphaseg .== 0.0) & (orientg .== 0.0)))  # this part is where both are
                                                        # zero because there is no gradient

    ## Checkerboard Image with RBG pixels

    (gx_rgb, gy_rgb) = imgradients(cb_image_rgb, method)
    mag_rgb = magnitude(gx_rgb, gy_rgb)
    gphase_rgb = phase(gx_rgb, gy_rgb)
    @test (mag_rgb, gphase_rgb) == magnitude_phase(gx_rgb, gy_rgb)

    @test gx_rgb[1,SZ,1]   < 0.0   # white to black transition
    @test gx_rgb[1,2*SZ,1] > 0.0   # black to white transition
    @test gy_rgb[1,1,SZ]   < 0.0   # white to black transition
    @test gy_rgb[1,1,2*SZ] > 0.0   # black to white transition

    @test cos(gphase_rgb[1,SZ,1])   - (-1.0) < EPS   # increasing left  (=  pi   radians)
    @test cos(gphase_rgb[1,2*SZ,1]) -   1.0  < EPS   # increasing right (=   0   radians)
    @test sin(gphase_rgb[1,1,SZ])   -   1.0  < EPS   # increasing up    (=  pi/2 radians)
    @test sin(gphase_rgb[1,1,2*SZ]) - (-1.0) < EPS   # increasing down  (= -pi/2 radians)

    # Test that orientation is perpendicular to gradient
    orient_rgb = orientation(gx_rgb, gy_rgb)
    @test all((cos(gphase_rgb).*cos(orient_rgb) .+ sin(gphase_rgb).*sin(orient_rgb) .< EPS) |
                ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0)))  # this part is where both are
                                                                # zero because there is no gradient

    ## Checkerboard Image with RBG{Float64} pixels

    (gx_rgb, gy_rgb) = imgradients(cb_image_rgb2, method)
    mag_rgb = magnitude(gx_rgb, gy_rgb)
    gphase_rgb = phase(gx_rgb, gy_rgb)
    @test (mag_rgb, gphase_rgb) == magnitude_phase(gx_rgb, gy_rgb)

    @test gx_rgb[1,SZ,1]   < 0.0   # white to black transition
    @test gx_rgb[1,2*SZ,1] > 0.0   # black to white transition
    @test gy_rgb[1,1,SZ]   < 0.0   # white to black transition
    @test gy_rgb[1,1,2*SZ] > 0.0   # black to white transition

    @test cos(gphase_rgb[1,SZ,1])   - (-1.0) < EPS   # increasing left  (=  pi   radians)
    @test cos(gphase_rgb[1,2*SZ,1]) -   1.0  < EPS   # increasing right (=   0   radians)
    @test sin(gphase_rgb[1,1,SZ])   -   1.0  < EPS   # increasing up    (=  pi/2 radians)
    @test sin(gphase_rgb[1,1,2*SZ]) - (-1.0) < EPS   # increasing down  (= -pi/2 radians)

    # Test that orientation is perpendicular to gradient
    orient_rgb = orientation(gx_rgb, gy_rgb)
    @test all((cos(gphase_rgb).*cos(orient_rgb) .+ sin(gphase_rgb).*sin(orient_rgb) .< EPS) |
                ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0)))  # this part is where both are
                                                                # zero because there is no gradient
end

# Create an image with white along diagonals -2:2 and black elsewhere
m = zeros(Uint8, 20,20)
for i = -2:2; m[diagind(m,i)] = 0xff; end

m_xy = grayim(m')
m_yx = grayim(m)
m_yx["spatialorder"] = ["y","x"]

for method in ["sobel", "prewitt", "ando3", "ando4", "ando5", "ando4_sep", "ando5_sep"]
    ## Diagonal array

    (agx, agy) = imgradients(m, method)
    amag = magnitude(agx, agy)
    agphase = phase(agx, agy)
    @test (amag, agphase) == magnitude_phase(agx, agy)

    @test agx[7,9]  < 0.0   # white to black transition
    @test agx[10,8] > 0.0   # black to white transition
    @test agy[10,8] < 0.0   # white to black transition
    @test agy[7,9]  > 0.0   # black to white transition

    # Test direction of increasing gradient
    @test abs(agphase[10,8] -    pi/4 ) < EPS   # lower edge (increasing up-right  =   pi/4 radians)
    @test abs(agphase[7,9]  - (-3pi/4)) < EPS   # upper edge (increasing down-left = -3pi/4 radians)

    # Test that orientation is perpendicular to gradient
    aorient = orientation(agx, agy)
    @test all((cos(agphase).*cos(aorient) .+ sin(agphase).*sin(aorient) .< EPS) |
                ((agphase .== 0.0) & (aorient .== 0.0)))  # this part is where both are
                                                          # zero because there is no gradient

    ## Diagonal Image, row-major order

    (gx, gy) = imgradients(m_xy, method)
    mag = magnitude(gx, gy)
    gphase = phase(gx, gy)
    @test (mag, gphase) == magnitude_phase(gx, gy)

    @test gx[9,7]  < 0.0   # white to black transition
    @test gx[8,10] > 0.0   # black to white transition
    @test gy[8,10] < 0.0   # white to black transition
    @test gy[9,7]  > 0.0   # black to white transition

    # Test direction of increasing gradient
    @test abs(gphase[8,10] -    pi/4 ) < EPS   # lower edge (increasing up-right  =   pi/4 radians)
    @test abs(gphase[9,7]  - (-3pi/4)) < EPS   # upper edge (increasing down-left = -3pi/4 radians)

    # Test that orientation is perpendicular to gradient
    orient = orientation(gx, gy)
    @test all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                ((gphase .== 0.0) & (orient .== 0.0)))  # this part is where both are
                                                        # zero because there is no gradient

    ## Diagonal Image, column-major order

    (gx, gy) = imgradients(m_yx, method)
    mag = magnitude(gx, gy)
    gphase = phase(gx, gy)
    @test (mag, gphase) == magnitude_phase(gx, gy)

    @test gx[7,9]  < 0.0   # white to black transition
    @test gx[10,8] > 0.0   # black to white transition
    @test gy[10,8] < 0.0   # white to black transition
    @test gy[7,9]  > 0.0   # black to white transition

    # Test direction of increasing gradient
    @test abs(gphase[10,8] -    pi/4 ) < EPS   # lower edge (increasing up-right  =   pi/4 radians)
    @test abs(gphase[7,9]  - (-3pi/4)) < EPS   # upper edge (increasing down-left = -3pi/4 radians)

    # Test that orientation is perpendicular to gradient
    orient = orientation(gx, gy)
    @test all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                ((gphase .== 0.0) & (orient .== 0.0)))  # this part is where both are
                                                        # zero because there is no gradient
end

# Nonmaximal suppression

function thin_edges(img)
    # Get orientation
    gx,gy = imgradients(img)
    orient = phase(gx,gy)

    # Do NMS thinning
    thin_edges_nonmaxsup_subpix(img, orient, radius=1.35)
end


function nms_test_horiz_vert(img, which)
    ## which = :horizontal or :vertical

    # Do NMS thinning
    t,s = thin_edges(img)

    # Calc peak location by hand

    # Interpolate values 1.35 pixels left and right
    # Orientation is zero radians -> to the right
    v1 = 6 - 0.35   # slope on right is 1
    v2 = 5 - 0.35*2 # slope on left is 2
    c = 7.0         # peak value

    # solve v = a*r^2 + b*r + c
    a = (v1 + v2)/2 - c
    b = a + c - v2
    r = -b/2a

    @test abs(r - 1/6) < EPS

    # Location and value at peak
    peakloc = r*1.35 + 3
    peakval = a*r^2 + b*r + c

    transposed = spatialorder(img)[1] == "x"
    horizontal = which == :horizontal

    test_axis1 = transposed $ !horizontal

    @test test_axis1 ? all(t[:,[1,2,4,5]] .== 0) : all(t[[1,2,4,5],:] .== 0)
    @test test_axis1 ? all(t[:,3]   .== peakval) : all(t[3,:]   .== peakval)
    @test test_axis1 ? all(s[:,[1,2,4,5]] .== zero(Graphics.Point)) :
                         all(s[[1,2,4,5],:] .== zero(Graphics.Point))

    if transposed
        if which == :horizontal
            @test     [pt.x for pt in s[:,3]]  == [1:5;]
            @test all([pt.y for pt in s[:,3]] .== peakloc)
        else
            @test all([pt.x for pt in s[3,:]] .== peakloc)
            @test     [pt.y for pt in s[3,:]]  == [1:5;]
        end
    else
        if which == :horizontal
            @test     [pt.x for pt in s[3,:]]  == [1:5;]
            @test all([pt.y for pt in s[3,:]] .== peakloc)
        else
            @test all([pt.x for pt in s[:,3]] .== peakloc)
            @test     [pt.y for pt in s[:,3]]  == [1:5;]
        end
    end
end

# Test image: vertical edge
m = [3.0  5.0  7.0  6.0  5.0
     3.0  5.0  7.0  6.0  5.0
     3.0  5.0  7.0  6.0  5.0
     3.0  5.0  7.0  6.0  5.0
     3.0  5.0  7.0  6.0  5.0]

m_xy = grayim(m')
m_yx = grayim(m)
m_yx["spatialorder"] = ["y","x"]

nms_test_horiz_vert(m, :vertical)
nms_test_horiz_vert(m_xy, :vertical)
nms_test_horiz_vert(m_yx, :vertical)

# Test image: horizontal edge
m = m'
m_xy = grayim(m')
m_yx = grayim(m)
m_yx["spatialorder"] = ["y","x"]

nms_test_horiz_vert(m, :horizontal)
nms_test_horiz_vert(m_xy, :horizontal)
nms_test_horiz_vert(m_yx, :horizontal)


function nms_test_diagonal(img)
    # Do NMS thinning
    t,s = thin_edges(img)

    # Calc peak location by hand

    # Interpolate values 1.35 pixels up and left, down and right
    # using bilinear interpolation
    # Orientation is π/4 radians -> 45 degrees up
    fr = 1.35*cos(π/4)
    lower = (7 + fr*(6-7))
    upper = (6 + fr*(5-6))
    v1 = lower + fr*(upper-lower)

    lower = (7 + fr*(5-7))
    upper = (5 + fr*(3-5))
    v2 = lower + fr*(upper-lower)

    c = 7.0         # peak value

    # solve v = a*r^2 + b*r + c
    a = (v1 + v2)/2 - c
    b = a + c - v2
    r = -b/2a

    @test (r - 1/6) < EPS

    transposed = spatialorder(img)[1] == "x"

    # Location and value at peak

    x_peak_offset, y_peak_offset = r*fr, -r*fr
    peakval = a*r^2 + b*r + c

    @test all(diag(data(t))[2:4] .== peakval)  # Edge pixels aren't interpolated here
    @test all(t - diagm(diag(data(t))) .== 0)

    diag_s = copyproperties(s, diagm(diag(data(s))))
    @test s == diag_s

    @test all([pt.x for pt in diag(data(s))[2:4]] - ((2:4) + x_peak_offset) .< EPS)
    @test all([pt.y for pt in diag(data(s))[2:4]] - ((2:4) + y_peak_offset) .< EPS)

end


# Test image: diagonal edge
m = [7.0  6.0  5.0  0.0  0.0
     5.0  7.0  6.0  5.0  0.0
     3.0  5.0  7.0  6.0  5.0
     0.0  3.0  5.0  7.0  6.0
     0.0  0.0  3.0  5.0  7.0]

m_xy = grayim(m')
m_yx = grayim(m)
m_yx["spatialorder"] = ["y","x"]

nms_test_diagonal(m)
nms_test_diagonal(m_xy)
nms_test_diagonal(m_yx)
