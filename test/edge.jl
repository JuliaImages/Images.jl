using Images, Base.Test, Colors, Compat

global checkboard

@testset "Edge" begin

    @testset "imedge" begin
        img = zeros(8, 10)
        img[:, 5] = 1
        grad_y, grad_x, mag, orient = imedge(img)
        @test all(x->x==0, grad_y)
        target_x = zeros(8, 10); target_x[:, 4] = 0.5; target_x[:, 6] = -0.5
        @test grad_x == target_x
        @test mag == abs.(grad_x)
        target_orient = zeros(8, 10); target_orient[:, 6] = pi
        @test orient ≈ target_orient
    end

    EPS = 1e-14

    kernelmethods = (KernelFactors.sobel, KernelFactors.prewitt, KernelFactors.ando3,
                     KernelFactors.ando4, KernelFactors.ando5)

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
        edges = canny(img, (0.8, 0.2))
        @test eltype(edges) == Bool
        @test all(.!edges)

        #Box Edges

        img[2:end-1, 2:end-1] = 1
        edges = canny(img, (Percentile(80), Percentile(20)))
        @test all(edges[2:end-1, 2])
        @test all(edges[2:end-1, end-1])
        @test all(edges[2, 2:end-1])
        @test all(edges[end-1, 2:end-1])
        @test all(.!edges[3:end-2, 3:end-2])

        edges = canny(img, (0.9/8, 0.2/8), 1.4)
        @test all(edges[2:end-1, 2])
        @test all(edges[2:end-1, end-1])
        @test all(edges[2, 2:end-1])
        @test all(edges[end-1, 2:end-1])
        @test all(.!edges[3:end-2, 3:end-2])

        #Checkerboard - Corners are not detected as Edges!
        img = checkerboard(Gray, 5, 3)
        edges = canny(img, (Percentile(80), Percentile(20)), 1.4)
        @test eltype(edges) == Bool
        id = [1,2,3,4,6,7,8,9,10,12,13,14,15]
        @test all(.!edges[id, id])
        id = [5, 11]
        id2 = [1,2,3,4,7,8,9,12,13,14,15]
        id3 = [5,6,10,11]
        @test all(edges[id, id2])
        @test all(edges[id2, id])
        @test all(.!edges[id, id3])
        @test all(.!edges[id3, id])

        #Diagonal Edge
        img = zeros(10,10)
        img[diagind(img)] = 1
        img[diagind(img, 1)] = 1
        img[diagind(img, -1)] = 1
        edges = canny(img, (Percentile(80),Percentile(20)))
        @test eltype(edges) == Bool
        @test all(edges[diagind(edges, 2)])
        @test all(edges[diagind(edges, -2)])
        nondiags = setdiff(1:1:100, union(diagind(edges, 2), diagind(edges, -2)))
        @test all(.!edges[nondiags])

        #Checks Hysteresis Thresholding
        img = ones(Gray{N0f8}, (10, 10))
        img[3:7, 3:7] = 0.0
        img[4:6, 4:6] = 0.7
        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.8)
        @test all(thresholded[3:7, 3:7] .== 0.0)
        @test all(thresholded[1:2, :] .== 0.9)
        @test all(thresholded[:, 1:2] .== 0.9)
        @test all(thresholded[8:10, :] .== 0.9)
        @test all(thresholded[:, 8:10] .== 0.9)

        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.6)
        @test all(thresholded[4:6, 4:6] .== 0.5)
        @test all(thresholded[3:7, 3:7] .< 0.9)
        @test all(thresholded[1:2, :] .== 0.9)
        @test all(thresholded[:, 1:2] .== 0.9)
        @test all(thresholded[8:10, :] .== 0.9)
        @test all(thresholded[:, 8:10] .== 0.9)

        img[3, 5] = 0.7
        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.6)
        @test all(thresholded[4:6, 4:6] .== 0.9)
        @test all(thresholded[3:7, 3] .== 0.0)
        @test all(thresholded[3:7, 7] .== 0.0)
        @test all(thresholded[7, 3:7] .== 0.0)
        @test all(vec(thresholded[3, 3:7]) .== [0.0, 0.0, 0.9, 0.0, 0.0])
        @test all(thresholded[1:2, :] .== 0.9)
        @test all(thresholded[:, 1:2] .== 0.9)
        @test all(thresholded[8:10, :] .== 0.9)
        @test all(thresholded[:, 8:10] .== 0.9)
    end

    SZ=5
    cb_array    = checkerboard(SZ,3)

    cb_gray = Gray.(normedview(cb_array))
    cb_rgb = convert(Array{RGB}, cb_gray)
    cb_rgbf64 = convert(Array{RGB{Float64}}, cb_gray)

    # TODO: now that all of these behave nearly the same, put in a
    # loop over the array type (only have to adjust tests for `red`)
    @testset "Checkerboard" begin
        for method in kernelmethods
            ## Checkerboard array

            (agy, agx) = imgradients(cb_array, method)
            amag = magnitude(agx, agy)
            agphase = phase(agx, agy)
            @test (amag, agphase) == magnitude_phase(agx, agy)

            @test agx[1,SZ] < 0
            @test agx[1,2*SZ] > 0
            @test agy[SZ,1] < 0
            @test agy[2*SZ,1] > 0

            # Test direction of increasing gradient
            @test cos(agphase[1,SZ])   - (-1.0) < EPS  # increasing left  (=  pi   radians)
            @test cos(agphase[1,2*SZ]) -   1.0  < EPS  # increasing right (=   0   radians)
            @test sin(agphase[SZ,1])   -   1.0  < EPS  # increasing up    (=  pi/2 radians)
            @test sin(agphase[2*SZ,1]) - (-1.0) < EPS  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @test all((cos.(agphase) .* cos.(aorient) .+
                       sin.(agphase) .* sin.(aorient) .< EPS) .|
                      ((agphase .== 0.0) .& (aorient .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Checkerboard with Gray pixels

            (gyg, gxg) = imgradients(cb_gray, method)
            magg = magnitude(gxg, gyg)
            gphaseg = phase(gxg, gyg)
            @test (magg, gphaseg) == magnitude_phase(gxg, gyg)
            @test gyg[SZ,1]   < 0     # white to black transition
            @test gyg[2*SZ,1] > 0  # black to white transition
            @test gxg[1,SZ]   < 0     # white to black transition
            @test gxg[1,2*SZ] > 0  # black to white transition

            @test cos(gphaseg[1,SZ])   - (-1.0) < EPS  # increasing left  (=  pi   radians)
            @test cos(gphaseg[1,2*SZ]) -   1.0  < EPS  # increasing right (=   0   radians)
            @test sin(gphaseg[SZ,1])   -   1.0  < EPS  # increasing up    (=  pi/2 radians)
            @test sin(gphaseg[2*SZ,1]) - (-1.0) < EPS  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orientg = orientation(gxg, gyg)
            @test all((cos.(gphaseg) .* cos.(orientg) .+
                       sin.(gphaseg) .* sin.(orientg) .< EPS) .|
                      ((gphaseg .== 0.0) .& (orientg .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Checkerboard with RBG pixels

            (gy_rgb, gx_rgb) = imgradients(cb_rgb, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @test (mag_rgb, gphase_rgb) == magnitude_phase(gx_rgb, gy_rgb)
            @test red(gy_rgb[SZ,1])   < 0     # white to black transition
            @test red(gy_rgb[2*SZ,1]) > 0  # black to white transition
            @test red(gx_rgb[1,SZ])   < 0     # white to black transition
            @test red(gx_rgb[1,2*SZ]) > 0  # black to white transition

            @test cos(gphase_rgb[1,SZ])   - (-1.0) < EPS  # increasing left  (=  pi   radians)
            @test cos(gphase_rgb[1,2*SZ]) -   1.0  < EPS  # increasing right (=   0   radians)
            @test sin(gphase_rgb[SZ,1])   -   1.0  < EPS  # increasing up    (=  pi/2 radians)
            @test sin(gphase_rgb[2*SZ,1]) - (-1.0) < EPS  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @test all((cos.(gphase_rgb) .* cos.(orient_rgb) .+
                       sin.(gphase_rgb) .* sin.(orient_rgb) .< EPS) .|
                      ((gphase_rgb .== 0.0) .& (orient_rgb .== 0.0)))
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with RBG{Float64} pixels

            (gy_rgb, gx_rgb) = imgradients(cb_rgbf64, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @test (mag_rgb, gphase_rgb) == magnitude_phase(gx_rgb, gy_rgb)
            @test red(gy_rgb[SZ,1])   < 0     # white to black transition
            @test red(gy_rgb[2*SZ,1]) > 0  # black to white transition
            @test red(gx_rgb[1,SZ])   < 0     # white to black transition
            @test red(gx_rgb[1,2*SZ]) > 0  # black to white transition

            @test cos(gphase_rgb[1,SZ])   - (-1.0) < EPS  # increasing left  (=  pi   radians)
            @test cos(gphase_rgb[1,2*SZ]) -   1.0  < EPS  # increasing right (=   0   radians)
            @test sin(gphase_rgb[SZ,1])   -   1.0  < EPS  # increasing up    (=  pi/2 radians)
            @test sin(gphase_rgb[2*SZ,1]) - (-1.0) < EPS  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @test all((cos.(gphase_rgb) .* cos.(orient_rgb) .+
                       sin.(gphase_rgb) .* sin.(orient_rgb) .< EPS) .|
                      ((gphase_rgb .== 0.0) .& (orient_rgb .== 0.0)))
                      # this part is where both are zero because there is no gradient
        end
    end

    @testset "Diagonals" begin
        # Create an image with white along diagonals -2:2 and black elsewhere
        m = zeros(UInt8, 20,20)
        for i = -2:2; m[diagind(m,i)] = 0xff; end

        for method in kernelmethods
            ## Diagonal array

            (agy, agx) = imgradients(m, method)
            amag = magnitude(agx, agy)
            agphase = phase(agx, agy)
            @test magnitude_phase(agx, agy) == (amag, agphase)
            @test agx[7,9]  < 0     # white to black transition
            @test agx[10,8] > 0  # black to white transition
            @test agy[10,8] < 0     # white to black transition
            @test agy[7,9]  > 0  # black to white transition

            # Test direction of increasing gradient
            @test abs(agphase[10,8] -    pi/4 ) < EPS  # lower edge (increasing up-right  =   pi/4 radians)
            @test abs(agphase[7,9]  - (-3pi/4)) < EPS  # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @test all((cos.(agphase) .* cos.(aorient) .+
                       sin.(agphase) .* sin.(aorient) .< EPS) .|
                      ((agphase .== 0.0) .& (aorient .== 0.0)))
                    # this part is where both are zero because there is no gradient
        end
    end

    # Nonmaximal suppression

    function thin_edges(img)
        # Get orientation
        gy,gx = imgradients(img, KernelFactors.ando3)
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

        horizontal = which == :horizontal

        if !horizontal
            @test all(t[:,[1,2,4,5]] .== 0)
            @test all(t[:,3]   .== peakval)
            @test all(s[:,[1,2,4,5]] .== zero(Graphics.Point))
        else
            @test all(t[[1,2,4,5],:] .== 0)
            @test all(t[3,:]   .== peakval)
            @test all(s[[1,2,4,5],:] .== zero(Graphics.Point))
        end

        if which == :horizontal
            @test     [pt.x for pt in s[3,:]]  == [1:5;]
            @test all([pt.y for pt in s[3,:]] .== peakloc)
        else
            @test all([pt.x for pt in s[:,3]] .== peakloc)
            @test     [pt.y for pt in s[:,3]]  == [1:5;]
        end
    end

    @testset "Nonmax suppression vertical edge" begin
        # Test image: vertical edge
        m = [3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0]

        nms_test_horiz_vert(m, :vertical)
    end

    @testset "Nonmax suppression horizontal edge" begin
        # Test image: horizontal edge
        m = [3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0
             3.0  5.0  7.0  6.0  5.0]
        m = m'

        nms_test_horiz_vert(m, :horizontal)
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

    @test (r - 1/6) < EPS

    # Location and value at peak

    x_peak_offset, y_peak_offset = r*fr, -r*fr
    peakval = a*r^2 + b*r + c

    @test all(diag(t)[2:4] .== peakval)
    @test all(t - diagm(diag(t)) .== 0)

    diag_s = diagm(diag(s))
    @test s == diag_s

    @test all([pt.x for pt in diag(s)[2:4]] - ((2:4) + x_peak_offset) .< EPS)
    @test all([pt.y for pt in diag(s)[2:4]] - ((2:4) + y_peak_offset) .< EPS)

end

    @testset "Nonmax suppression diagonal" begin
        # Test image: diagonal edge
        m = [7.0  6.0  5.0  0.0  0.0
             5.0  7.0  6.0  5.0  0.0
             3.0  5.0  7.0  6.0  5.0
             0.0  3.0  5.0  7.0  6.0
             0.0  0.0  3.0  5.0  7.0]

        nms_test_diagonal(m)
    end

end

nothing
