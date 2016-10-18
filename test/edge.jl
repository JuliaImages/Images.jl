using Images, Colors

global checkboard

@testset "Edge" begin

    EPS = 1e-14

    ## Checkerboard array, used to test image gradients
    white{T}(::Type{T}) = one(T)
    black{T}(::Type{T}) = zero(T)
    white{T<:Unsigned}(::Type{T}) = typemax(T)
    black{T<:Unsigned}(::Type{T}) = typemin(T)

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

    checkerboard(sq_width::Integer, count::Integer) = checkerboard(UInt8, sq_width, count)

    #Canny Edge Detection
    @testset "Canny Edge Detection" begin
        #General Checks
        img = zeros(10, 10)
        edges = canny(img)
        @test eltype(edges.data) == Gray{U8}
        @test all(edges .== 0.0)

        #Box Edges

        img[2:end-1, 2:end-1] = 1
        edges = canny(img)
        @test all(edges[2:end - 1,2] .== 1.0)
        @test all(edges[2:end - 1,end - 1] .== 1.0)
        @test all(edges[2,2:end - 1] .== 1.0)
        @test all(edges[end - 1,2:end - 1] .== 1.0)
        @test all(edges[3:end - 2,3:end - 2] .== 0.0)

        edges = canny(img, 1.4, 0.9, 0.2, percentile = false)
        @test all(edges[2:end - 1,2] .== 1.0)
        @test all(edges[2:end - 1,end - 1] .== 1.0)
        @test all(edges[2,2:end - 1] .== 1.0)
        @test all(edges[end - 1,2:end - 1] .== 1.0)
        @test all(edges[3:end - 2,3:end - 2] .== 0.0)

        #Checkerboard - Corners are not detected as Edges!
        img = checkerboard(Gray, 5, 3)
        edges = canny(img, 1.4, 0.8, 0.2)
        id = [1,2,3,4,6,7,8,9,10,12,13,14,15]
        @test all(edges[id,id] .== zero(Gray))
        id = [5, 11]
        id2 = [1,2,3,4,7,8,9,12,13,14,15]
        id3 = [5,6,10,11]
        @test all(edges[id,id2] .== one(Gray))
        @test all(edges[id2,id] .== one(Gray))
        @test all(edges[id,id3] .== zero(Gray))
        @test all(edges[id3,id] .== zero(Gray))

        #Diagonal Edge
        img = zeros(10,10)
        img[diagind(img)] = 1
        img[diagind(img, 1)] = 1
        img[diagind(img, -1)] = 1
        edges = canny(img)
        @test all(edges[diagind(edges,2)] .== 1.0)
        @test all(edges[diagind(edges,-2)] .== 1.0)
        nondiags = setdiff(1:1:100, union(diagind(edges, 2), diagind(edges, -2)))
        @test all(edges[nondiags] .== 0.0)

        #Checks Hysteresis Thresholding
        img = ones(Gray{U8}, (10, 10))
        img[3:7, 3:7] = 0.0
        img[4:6, 4:6] = 0.7
        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.8)
        @test all(thresholded[3:7,3:7] .== 0.0)
        @test all(thresholded[1:2,:] .== 0.9)
        @test all(thresholded[:,1:2] .== 0.9)
        @test all(thresholded[8:10,:] .== 0.9)
        @test all(thresholded[:,8:10] .== 0.9)

        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.6)
        @test all(thresholded[4:6,4:6] .== 0.5)
        @test all(thresholded[3:7,3:7] .< 0.9)
        @test all(thresholded[1:2,:] .== 0.9)
        @test all(thresholded[:,1:2] .== 0.9)
        @test all(thresholded[8:10,:] .== 0.9)
        @test all(thresholded[:,8:10] .== 0.9)

        img[3, 5] = 0.7
        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.6)
        @test all(thresholded[4:6,4:6] .== 0.9)
        @test all(thresholded[3:7,3] .== 0.0)
        @test all(thresholded[3:7,7] .== 0.0)
        @test all(thresholded[7,3:7] .== 0.0)
        @test all(vec(thresholded[3,3:7]) .== [0.0,0.0,0.9,0.0,0.0])
        @test all(thresholded[1:2,:] .== 0.9)
        @test all(thresholded[:,1:2] .== 0.9)
        @test all(thresholded[8:10,:] .== 0.9)
        @test all(thresholded[:,8:10] .== 0.9)
    end

    SZ=5
    cb_array    = checkerboard(SZ,3)
    cb_image_xy = grayim(cb_array)
    cb_image_yx = grayim(cb_array)
    cb_image_yx["spatialorder"] = ["y","x"]

    cb_image_gray = convert(Image{Gray}, cb_image_xy)
    cb_image_rgb = convert(Image{RGB}, cb_image_xy)
    cb_image_rgb2 = convert(Image{RGB{Float64}}, cb_image_xy)

    @compat @testset "Checkerboard" begin
        for method in ["sobel", "prewitt", "ando3", "ando4", "ando5", "ando4_sep", "ando5_sep"]
            ## Checkerboard array

            (agx, agy) = imgradients(cb_array, method)
            amag = magnitude(agx, agy)
            agphase = phase(agx, agy)
            @test (amag,agphase) == magnitude_phase(agx,agy)

            @test agx[1,SZ] < 0.0 # white to black transition
            @test agx[1,2SZ] > 0.0 # black to white transition
            @test agy[SZ,1] < 0.0 # white to black transition
            @test agy[2SZ,1] > 0.0 # black to white transition

            # Test direction of increasing gradient
            @test cos(agphase[1,SZ]) - -1.0 < EPS # increasing left  (=  pi   radians)
            @test cos(agphase[1,2SZ]) - 1.0 < EPS # increasing right (=   0   radians)
            @test sin(agphase[SZ,1]) - 1.0 < EPS # increasing up    (=  pi/2 radians)
            @test sin(agphase[2SZ,1]) - -1.0 < EPS # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @test all((cos.(agphase) .* cos.(aorient) .+
                       sin.(agphase) .* sin.(aorient) .< EPS) |
                      ((agphase .== 0.0) & (aorient .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with row major order

            (gx, gy) = imgradients(cb_image_xy, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @test (mag,gphase) == magnitude_phase(gx,gy)

            @test gx[SZ,1] < 0.0 # white to black transition
            @test gx[2SZ,1] > 0.0 # black to white transition
            @test gy[1,SZ] < 0.0 # white to black transition
            @test gy[1,2SZ] > 0.0 # black to white transition

            @test cos(gphase[SZ,1]) - -1.0 < EPS # increasing left  (=  pi   radians)
            @test cos(gphase[2SZ,1]) - 1.0 < EPS # increasing right (=   0   radians)
            @test sin(gphase[1,SZ]) - 1.0 < EPS # increasing up    (=  pi/2 radians)
            @test sin(gphase[1,2SZ]) - -1.0 < EPS # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            @test all((cos.(gphase) .* cos.(orient) .+
                       sin.(gphase) .* sin.(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0)))
                       # this part is where both are zero because there is no gradient

            ## Checkerboard Image with column-major order

            (gx, gy) = imgradients(cb_image_yx, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @test (mag,gphase) == magnitude_phase(gx,gy)

            @test gx[1,SZ] < 0.0 # white to black transition
            @test gx[1,2SZ] > 0.0 # black to white transition
            @test gy[SZ,1] < 0.0 # white to black transition
            @test gy[2SZ,1] > 0.0 # black to white transition

            # Test direction of increasing gradient
            @test cos(gphase[1,SZ]) - -1.0 < EPS # increasing left  (=  pi   radians)
            @test cos(gphase[1,2SZ]) - 1.0 < EPS # increasing right (=   0   radians)
            @test sin(gphase[SZ,1]) - 1.0 < EPS # increasing up    (=  pi/2 radians)
            @test sin(gphase[2SZ,1]) - -1.0 < EPS # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            @test all((cos.(gphase) .* cos.(orient) .+
                       sin.(gphase) .* sin.(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with Gray pixels

            (gxg, gyg) = imgradients(cb_image_gray, method)
            magg = magnitude(gxg, gyg)
            gphaseg = phase(gxg, gyg)
            @test (magg,gphaseg) == magnitude_phase(gxg,gyg)

            @test gxg[SZ,1] < 0.0 # white to black transition
            @test gxg[2SZ,1] > 0.0 # black to white transition
            @test gyg[1,SZ] < 0.0 # white to black transition
            @test gyg[1,2SZ] > 0.0 # black to white transition

            @test cos(gphaseg[SZ,1]) - -1.0 < EPS # increasing left  (=  pi   radians)
            @test cos(gphaseg[2SZ,1]) - 1.0 < EPS # increasing right (=   0   radians)
            @test sin(gphaseg[1,SZ]) - 1.0 < EPS # increasing up    (=  pi/2 radians)
            @test sin(gphaseg[1,2SZ]) - -1.0 < EPS # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orientg = orientation(gxg, gyg)
            @test all((cos.(gphaseg) .* cos.(orientg) .+
                       sin.(gphaseg) .* sin.(orientg) .< EPS) |
                      ((gphaseg .== 0.0) & (orientg .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with RBG pixels

            (gx_rgb, gy_rgb) = imgradients(cb_image_rgb, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @test (mag_rgb,gphase_rgb) == magnitude_phase(gx_rgb,gy_rgb)

            @test gx_rgb[1,SZ,1] < 0.0 # white to black transition
            @test gx_rgb[1,2SZ,1] > 0.0 # black to white transition
            @test gy_rgb[1,1,SZ] < 0.0 # white to black transition
            @test gy_rgb[1,1,2SZ] > 0.0 # black to white transition

            @test cos(gphase_rgb[1,SZ,1]) - -1.0 < EPS # increasing left  (=  pi   radians)
            @test cos(gphase_rgb[1,2SZ,1]) - 1.0 < EPS # increasing right (=   0   radians)
            @test sin(gphase_rgb[1,1,SZ]) - 1.0 < EPS # increasing up    (=  pi/2 radians)
            @test sin(gphase_rgb[1,1,2SZ]) - -1.0 < EPS # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @test all((cos.(gphase_rgb) .* cos.(orient_rgb) .+
                       sin.(gphase_rgb) .* sin.(orient_rgb) .< EPS) |
                      ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with RBG{Float64} pixels

            (gx_rgb, gy_rgb) = imgradients(cb_image_rgb2, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @test (mag_rgb,gphase_rgb) == magnitude_phase(gx_rgb,gy_rgb)

            @test gx_rgb[1,SZ,1] < 0.0 # white to black transition
            @test gx_rgb[1,2SZ,1] > 0.0 # black to white transition
            @test gy_rgb[1,1,SZ] < 0.0 # white to black transition
            @test gy_rgb[1,1,2SZ] > 0.0 # black to white transition

            @test cos(gphase_rgb[1,SZ,1]) - -1.0 < EPS # increasing left  (=  pi   radians)
            @test cos(gphase_rgb[1,2SZ,1]) - 1.0 < EPS # increasing right (=   0   radians)
            @test sin(gphase_rgb[1,1,SZ]) - 1.0 < EPS # increasing up    (=  pi/2 radians)
            @test sin(gphase_rgb[1,1,2SZ]) - -1.0 < EPS # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @test all((cos.(gphase_rgb) .* cos.(orient_rgb) .+
                       sin.(gphase_rgb) .* sin.(orient_rgb) .< EPS) |
                      ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0)))
                      # this part is where both are zero because there is no gradient
        end
    end

    @compat @testset "Diagonals" begin
        # Create an image with white along diagonals -2:2 and black elsewhere
        m = zeros(UInt8, 20,20)
        for i = -2:2; m[diagind(m,i)] = 0xff; end

        m_xy = grayim(m')
        m_yx = grayim(m)
        m_yx["spatialorder"] = ["y","x"]

        for method in ["sobel", "prewitt", "ando3", "ando4", "ando5", "ando4_sep", "ando5_sep"]
            ## Diagonal array

            (agx, agy) = imgradients(m, method)
            amag = magnitude(agx, agy)
            agphase = phase(agx, agy)
            @test (amag,agphase) == magnitude_phase(agx,agy)

            @test agx[7,9] < 0.0 # white to black transition
            @test agx[10,8] > 0.0 # black to white transition
            @test agy[10,8] < 0.0 # white to black transition
            @test agy[7,9] > 0.0 # black to white transition

            # Test direction of increasing gradient
            @test abs(agphase[10,8] - pi / 4) < EPS # lower edge (increasing up-right  =   pi/4 radians)
            @test abs(agphase[7,9] - (-3pi) / 4) < EPS # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @test all((cos.(agphase) .* cos.(aorient) .+
                       sin.(agphase) .* sin.(aorient) .< EPS) |
                      ((agphase .== 0.0) & (aorient .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Diagonal Image, row-major order

            (gx, gy) = imgradients(m_xy, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @test (mag,gphase) == magnitude_phase(gx,gy)

            @test gx[9,7] < 0.0 # white to black transition
            @test gx[8,10] > 0.0 # black to white transition
            @test gy[8,10] < 0.0 # white to black transition
            @test gy[9,7] > 0.0 # black to white transition

            # Test direction of increasing gradient
            @test abs(gphase[8,10] - pi / 4) < EPS # lower edge (increasing up-right  =   pi/4 radians)
            @test abs(gphase[9,7] - (-3pi) / 4) < EPS # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            all((cos.(gphase) .* cos.(orient) .+
                       sin.(gphase) .* sin.(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Diagonal Image, column-major order

            (gx, gy) = imgradients(m_yx, method)
            mag = magnitude(gx, gy)
            gphase = phase(gx, gy)
            @test (mag,gphase) == magnitude_phase(gx,gy)

            @test gx[7,9] < 0.0 # white to black transition
            @test gx[10,8] > 0.0 # black to white transition
            @test gy[10,8] < 0.0 # white to black transition
            @test gy[7,9] > 0.0 # black to white transition

            # Test direction of increasing gradient
            @test abs(gphase[10,8] - pi / 4) < EPS # lower edge (increasing up-right  =   pi/4 radians)
            @test abs(gphase[7,9] - (-3pi) / 4) < EPS # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            orient = orientation(gx, gy)
            all((cos.(gphase) .* cos.(orient) .+
                       sin.(gphase) .* sin.(orient) .< EPS) |
                      ((gphase .== 0.0) & (orient .== 0.0)))
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

        @test abs(r - 1 / 6) < EPS

        # Location and value at peak
        peakloc = r*1.35 + 3
        peakval = a*r^2 + b*r + c

        transposed = spatialorder(img)[1] == "x"
        horizontal = which == :horizontal

        test_axis1 = transposed $ !horizontal

        if test_axis1
            @test all(t[:,[1,2,4,5]] .== 0)
            @test all(t[:,3] .== peakval)
            @test all(s[:,[1,2,4,5]] .== zero(Graphics.Point))
        else
            @test all(t[[1,2,4,5],:] .== 0)
            @test all(t[3,:] .== peakval)
            @test all(s[[1,2,4,5],:] .== zero(Graphics.Point))
        end

        if transposed
            if which == :horizontal
                @test [pt.x for pt = s[:,3]] == [1:5;]
                @test all([pt.y for pt = s[:,3]] .== peakloc)
            else
                @test all([pt.x for pt = s[3,:]] .== peakloc)
                @test [pt.y for pt = s[3,:]] == [1:5;]
            end
        else
            if which == :horizontal
                @test [pt.x for pt = s[3,:]] == [1:5;]
                @test all([pt.y for pt = s[3,:]] .== peakloc)
            else
                @test all([pt.x for pt = s[:,3]] .== peakloc)
                @test [pt.y for pt = s[:,3]] == [1:5;]
            end
        end
    end

    @testset "Nonmax suppression vertical edge" begin
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

    @testset "Nonmax suppression horizontal edge" begin
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

    @test r - 1 / 6 < EPS

    transposed = spatialorder(img)[1] == "x"

    # Location and value at peak

    x_peak_offset, y_peak_offset = r*fr, -r*fr
    peakval = a*r^2 + b*r + c

    @test all((diag(data(t)))[2:4] .== peakval) # Edge pixels aren't interpolated here
    @test all(t - diagm(diag(data(t))) .== 0)

    diag_s = copyproperties(s, diagm(diag(data(s))))
    @test s == diag_s

    @test all([pt.x for pt = (diag(data(s)))[2:4]] - ((2:4) + x_peak_offset) .< EPS)
    @test all([pt.y for pt = (diag(data(s)))[2:4]] - ((2:4) + y_peak_offset) .< EPS)

end

    @testset "Nonmax suppression diagonal" begin
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
