using Images, FactCheck, Base.Test, Colors

facts("Edge") do
    EPS = 1e-14

    ## Checkerboard array, used to test image gradients
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

    SZ=5
    cb_array    = checkerboard(SZ,3)
    cb_image_xy = grayim(cb_array)
    cb_image_yx = grayim(cb_array)
    cb_image_yx["spatialorder"] = ["y","x"]

    cb_image_gray = convert(Image{Gray}, cb_image_xy)
    cb_image_rgb = convert(Image{RGB}, cb_image_xy)
    cb_image_rgb2 = convert(Image{RGB{Float64}}, cb_image_xy)

    context("Checkerboard") do
        for method in ["sobel", "prewitt", "ando3", "ando4", "ando5", "ando4_sep", "ando5_sep"]
            ## Checkerboard array

            (agx, agy) = imgradients(cb_array, method)
            amag = magnitude(agx, agy)
            agphase = phase(agx, agy)
            @fact (amag, agphase) --> magnitude_phase(agx, agy)

            @fact agx[1,SZ]   --> less_than(0.0)      # white to black transition
            @fact agx[1,2*SZ] --> greater_than(0.0)   # black to white transition
            @fact agy[SZ,1]   --> less_than(0.0)      # white to black transition
            @fact agy[2*SZ,1] --> greater_than(0.0)   # black to white transition

            # Test direction of increasing gradient
            @fact cos(agphase[1,SZ])   - (-1.0) --> less_than(EPS)   # increasing left  (=  pi   radians)
            @fact cos(agphase[1,2*SZ]) -   1.0  --> less_than(EPS)   # increasing right (=   0   radians)
            @fact sin(agphase[SZ,1])   -   1.0  --> less_than(EPS)   # increasing up    (=  pi/2 radians)
            @fact sin(agphase[2*SZ,1]) - (-1.0) --> less_than(EPS)   # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @fact all((cos(agphase).*cos(aorient) .+ sin(agphase).*sin(aorient) .< EPS) |
                      ((agphase .== 0.0) & (aorient .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with row major order

            (gx, gy) = imgradients(cb_image_xy, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @fact (mag, gphase) --> magnitude_phase(gx, gy)

            @fact gx[SZ,1]   --> less_than(0.0)      # white to black transition
            @fact gx[2*SZ,1] --> greater_than(0.0)   # black to white transition
            @fact gy[1,SZ]   --> less_than(0.0)      # white to black transition
            @fact gy[1,2*SZ] --> greater_than(0.0)   # black to white transition

            @fact cos(gphase[SZ,1])   - (-1.0) --> less_than(EPS)   # increasing left  (=  pi   radians)
            @fact cos(gphase[2*SZ,1]) -   1.0  --> less_than(EPS)   # increasing right (=   0   radians)
            @fact sin(gphase[1,SZ])   -   1.0  --> less_than(EPS)   # increasing up    (=  pi/2 radians)
            @fact sin(gphase[1,2*SZ]) - (-1.0) --> less_than(EPS)   # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            @fact all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0))) --> true
                       # this part is where both are zero because there is no gradient

            ## Checkerboard Image with column-major order

            (gx, gy) = imgradients(cb_image_yx, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @fact (mag, gphase) --> magnitude_phase(gx, gy)

            @fact gx[1,SZ]   --> less_than(0.0)      # white to black transition
            @fact gx[1,2*SZ] --> greater_than(0.0)   # black to white transition
            @fact gy[SZ,1]   --> less_than(0.0)      # white to black transition
            @fact gy[2*SZ,1] --> greater_than(0.0)   # black to white transition

            # Test direction of increasing gradient
            @fact cos(gphase[1,SZ])   - (-1.0) --> less_than(EPS)   # increasing left  (=  pi   radians)
            @fact cos(gphase[1,2*SZ]) -   1.0  --> less_than(EPS)   # increasing right (=   0   radians)
            @fact sin(gphase[SZ,1])   -   1.0  --> less_than(EPS)   # increasing up    (=  pi/2 radians)
            @fact sin(gphase[2*SZ,1]) - (-1.0) --> less_than(EPS)   # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            @fact all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with Gray pixels

            (gxg, gyg) = imgradients(cb_image_gray, method)
            magg = magnitude(gxg, gyg)
            gphaseg = phase(gxg, gyg)
            @fact (magg, gphaseg) --> magnitude_phase(gxg, gyg)

            @fact gxg[SZ,1]   --> less_than(0.0)      # white to black transition
            @fact gxg[2*SZ,1] --> greater_than(0.0)   # black to white transition
            @fact gyg[1,SZ]   --> less_than(0.0)      # white to black transition
            @fact gyg[1,2*SZ] --> greater_than(0.0)   # black to white transition

            @fact cos(gphaseg[SZ,1])   - (-1.0) --> less_than(EPS)   # increasing left  (=  pi   radians)
            @fact cos(gphaseg[2*SZ,1]) -   1.0  --> less_than(EPS)   # increasing right (=   0   radians)
            @fact sin(gphaseg[1,SZ])   -   1.0  --> less_than(EPS)   # increasing up    (=  pi/2 radians)
            @fact sin(gphaseg[1,2*SZ]) - (-1.0) --> less_than(EPS)   # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orientg = orientation(gxg, gyg)
            @fact all((cos(gphaseg).*cos(orientg) .+ sin(gphaseg).*sin(orientg) .< EPS) |
                      ((gphaseg .== 0.0) & (orientg .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with RBG pixels

            (gx_rgb, gy_rgb) = imgradients(cb_image_rgb, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @fact (mag_rgb, gphase_rgb) --> magnitude_phase(gx_rgb, gy_rgb)

            @fact gx_rgb[1,SZ,1]   --> less_than(0.0)      # white to black transition
            @fact gx_rgb[1,2*SZ,1] --> greater_than(0.0)   # black to white transition
            @fact gy_rgb[1,1,SZ]   --> less_than(0.0)      # white to black transition
            @fact gy_rgb[1,1,2*SZ] --> greater_than(0.0)   # black to white transition

            @fact cos(gphase_rgb[1,SZ,1])   - (-1.0) --> less_than(EPS)   # increasing left  (=  pi   radians)
            @fact cos(gphase_rgb[1,2*SZ,1]) -   1.0  --> less_than(EPS)   # increasing right (=   0   radians)
            @fact sin(gphase_rgb[1,1,SZ])   -   1.0  --> less_than(EPS)   # increasing up    (=  pi/2 radians)
            @fact sin(gphase_rgb[1,1,2*SZ]) - (-1.0) --> less_than(EPS)   # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @fact all((cos(gphase_rgb).*cos(orient_rgb) .+ sin(gphase_rgb).*sin(orient_rgb) .< EPS) |
                      ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with RBG{Float64} pixels

            (gx_rgb, gy_rgb) = imgradients(cb_image_rgb2, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @fact (mag_rgb, gphase_rgb) --> magnitude_phase(gx_rgb, gy_rgb)

            @fact gx_rgb[1,SZ,1]   --> less_than(0.0)      # white to black transition
            @fact gx_rgb[1,2*SZ,1] --> greater_than(0.0)   # black to white transition
            @fact gy_rgb[1,1,SZ]   --> less_than(0.0)      # white to black transition
            @fact gy_rgb[1,1,2*SZ] --> greater_than(0.0)   # black to white transition

            @fact cos(gphase_rgb[1,SZ,1])   - (-1.0) --> less_than(EPS)   # increasing left  (=  pi   radians)
            @fact cos(gphase_rgb[1,2*SZ,1]) -   1.0  --> less_than(EPS)   # increasing right (=   0   radians)
            @fact sin(gphase_rgb[1,1,SZ])   -   1.0  --> less_than(EPS)   # increasing up    (=  pi/2 radians)
            @fact sin(gphase_rgb[1,1,2*SZ]) - (-1.0) --> less_than(EPS)   # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @fact all((cos(gphase_rgb).*cos(orient_rgb) .+ sin(gphase_rgb).*sin(orient_rgb) .< EPS) |
                      ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient
        end
    end

    context("Diagonals") do
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
            @fact (amag, agphase) --> magnitude_phase(agx, agy)

            @fact agx[7,9]  --> less_than(0.0)      # white to black transition
            @fact agx[10,8] --> greater_than(0.0)   # black to white transition
            @fact agy[10,8] --> less_than(0.0)      # white to black transition
            @fact agy[7,9]  --> greater_than(0.0)   # black to white transition

            # Test direction of increasing gradient
            @fact abs(agphase[10,8] -    pi/4 ) --> less_than(EPS)   # lower edge (increasing up-right  =   pi/4 radians)
            @fact abs(agphase[7,9]  - (-3pi/4)) --> less_than(EPS)   # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @fact all((cos(agphase).*cos(aorient) .+ sin(agphase).*sin(aorient) .< EPS) |
                      ((agphase .== 0.0) & (aorient .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient

            ## Diagonal Image, row-major order

            (gx, gy) = imgradients(m_xy, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @fact (mag, gphase) --> magnitude_phase(gx, gy)

            @fact gx[9,7]  --> less_than(0.0)      # white to black transition
            @fact gx[8,10] --> greater_than(0.0)   # black to white transition
            @fact gy[8,10] --> less_than(0.0)      # white to black transition
            @fact gy[9,7]  --> greater_than(0.0)   # black to white transition

            # Test direction of increasing gradient
            @fact abs(gphase[8,10] -    pi/4 ) --> less_than(EPS)   # lower edge (increasing up-right  =   pi/4 radians)
            @fact abs(gphase[9,7]  - (-3pi/4)) --> less_than(EPS)   # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            @fact all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient

            ## Diagonal Image, column-major order

            (gx, gy) = imgradients(m_yx, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @fact (mag, gphase) --> magnitude_phase(gx, gy)

            @fact gx[7,9]  --> less_than(0.0)      # white to black transition
            @fact gx[10,8] --> greater_than(0.0)   # black to white transition
            @fact gy[10,8] --> less_than(0.0)      # white to black transition
            @fact gy[7,9]  --> greater_than(0.0)   # black to white transition

            # Test direction of increasing gradient
            @fact abs(gphase[10,8] -    pi/4 ) --> less_than(EPS)   # lower edge (increasing up-right  =   pi/4 radians)
            @fact abs(gphase[7,9]  - (-3pi/4)) --> less_than(EPS)   # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            @fact all((cos(gphase).*cos(orient) .+ sin(gphase).*sin(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0))) --> true
                      # this part is where both are zero because there is no gradient
        end
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

        @fact abs(r - 1/6) --> less_than(EPS)

        # Location and value at peak
        peakloc = r*1.35 + 3
        peakval = a*r^2 + b*r + c

        transposed = spatialorder(img)[1] == "x"
        horizontal = which == :horizontal

        test_axis1 = transposed $ !horizontal

        if test_axis1
            @fact all(t[:,[1,2,4,5]] .== 0) --> true
            @fact all(t[:,3]   .== peakval) --> true
            @fact all(s[:,[1,2,4,5]] .== zero(Graphics.Point)) --> true
        else
            @fact all(t[[1,2,4,5],:] .== 0) --> true
            @fact all(t[3,:]   .== peakval) --> true
            @fact all(s[[1,2,4,5],:] .== zero(Graphics.Point)) --> true
        end

        if transposed
            if which == :horizontal
                @fact     [pt.x for pt in s[:,3]]  --> [1:5;]
                @fact all([pt.y for pt in s[:,3]] .== peakloc) --> true
            else
                @fact all([pt.x for pt in s[3,:]] .== peakloc) --> true
                @fact     [pt.y for pt in s[3,:]]  --> [1:5;]
            end
        else
            if which == :horizontal
                @fact     [pt.x for pt in s[3,:]]  --> [1:5;]
                @fact all([pt.y for pt in s[3,:]] .== peakloc) --> true
            else
                @fact all([pt.x for pt in s[:,3]] .== peakloc) --> true
                @fact     [pt.y for pt in s[:,3]]  --> [1:5;]
            end
        end
    end

    context("Nonmax suppression vertical edge") do
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
    end

    context("Nonmax suppression horizontal edge") do
        # Test image: horizontal edge
        m = [3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0]
        m = m'
        m_xy = grayim(m')
        m_yx = grayim(m)
        m_yx["spatialorder"] = ["y","x"]

        nms_test_horiz_vert(m, :horizontal)
        nms_test_horiz_vert(m_xy, :horizontal)
        nms_test_horiz_vert(m_yx, :horizontal)
    end

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

    @fact (r - 1/6) --> less_than(EPS)

    transposed = spatialorder(img)[1] == "x"

    # Location and value at peak

    x_peak_offset, y_peak_offset = r*fr, -r*fr
    peakval = a*r^2 + b*r + c

    @fact all(diag(data(t))[2:4] .== peakval) --> true  # Edge pixels aren't interpolated here
    @fact all(t - diagm(diag(data(t))) .== 0) --> true

    diag_s = copyproperties(s, diagm(diag(data(s))))
    @fact s --> diag_s

    @fact all([pt.x for pt in diag(data(s))[2:4]] - ((2:4) + x_peak_offset) .< EPS) --> true
    @fact all([pt.y for pt in diag(data(s))[2:4]] - ((2:4) + y_peak_offset) .< EPS) --> true

end

    context("Nonmax suppression diagonal") do
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
    end

end
