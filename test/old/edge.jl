using Images, FactCheck, Base.Test, Colors, Compat

global checkboard

if VERSION < v"0.5.0"
    # Overwrite `Base.all` to work around poor inference on 0.4
    function all(ary)
        Base.all(convert(Array{Bool}, ary))
    end
end

facts("Edge") do

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
    context("Canny Edge Detection") do
        #General Checks
        img = zeros(10, 10)
        edges = canny(img)
        @fact all(edges .== 0.0) --> true "test XPvVle"

        #Box Edges

        img[2:end-1, 2:end-1] = 1
        edges = canny(img)
        @fact all(edges[2:end-1, 2] .== 1.0) --> true "test XaR3zT"
        @fact all(edges[2:end-1, end-1] .== 1.0) --> true "test hYEggy"
        @fact all(edges[2, 2:end-1] .== 1.0) --> true "test SGpiuC"
        @fact all(edges[end-1, 2:end-1] .== 1.0) --> true "test O1jzIi"
        @fact all(edges[3:end-2, 3:end-2] .== 0.0) --> true "test mnlN9g"

        edges = canny(img, 1.4, 0.9/8, 0.2/8, percentile = false)
        @fact all(edges[2:end-1, 2] .== 1.0) --> true "test 8vqfWg"
        @fact all(edges[2:end-1, end-1] .== 1.0) --> true "test shxyUG"
        @fact all(edges[2, 2:end-1] .== 1.0) --> true "test HcRl6t"
        @fact all(edges[end-1, 2:end-1] .== 1.0) --> true "test gdzLcr"
        @fact all(edges[3:end-2, 3:end-2] .== 0.0) --> true "test U3xnBr"

        #Checkerboard - Corners are not detected as Edges!
        img = checkerboard(Gray, 5, 3)
        edges = canny(img, 1.4, 0.8, 0.2)
        id = [1,2,3,4,6,7,8,9,10,12,13,14,15]
        @fact all(edges[id, id] .== zero(Gray)) --> true "test iXLmrb"
        id = [5, 11]
        id2 = [1,2,3,4,7,8,9,12,13,14,15]
        id3 = [5,6,10,11]
        @fact all(edges[id, id2] .== one(Gray)) --> true "test sdlPvR"
        @fact all(edges[id2, id] .== one(Gray)) --> true "test sMVseP"
        @fact all(edges[id, id3] .== zero(Gray)) --> true "test jDMJux"
        @fact all(edges[id3, id] .== zero(Gray)) --> true "test sAPU9M"

        #Diagonal Edge
        img = zeros(10,10)
        img[diagind(img)] = 1
        img[diagind(img, 1)] = 1
        img[diagind(img, -1)] = 1
        edges = canny(img)
        @fact all(edges[diagind(edges, 2)] .== 1.0) --> true "test ZnFAXW"
        @fact all(edges[diagind(edges, -2)] .== 1.0) --> true "test S0bdlI"
        nondiags = setdiff(1:1:100, union(diagind(edges, 2), diagind(edges, -2)))
        @fact all(edges[nondiags] .== 0.0) --> true "test WDQfYZ"

        #Checks Hysteresis Thresholding
        img = ones(Gray{N0f8}, (10, 10))
        img[3:7, 3:7] = 0.0
        img[4:6, 4:6] = 0.7
        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.8)
        @fact all(thresholded[3:7, 3:7] .== 0.0) --> true "test 3XLFf1"
        @fact all(thresholded[1:2, :] .== 0.9) --> true "test x3Gdlv"
        @fact all(thresholded[:, 1:2] .== 0.9) --> true "test 15gDt3"
        @fact all(thresholded[8:10, :] .== 0.9) --> true "test YlZ9mL"
        @fact all(thresholded[:, 8:10] .== 0.9) --> true "test a8kckT"

        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.6)
        @fact all(thresholded[4:6, 4:6] .== 0.5) --> true "test mvXBYG"
        @fact all(thresholded[3:7, 3:7] .< 0.9) --> true "test C424vj"
        @fact all(thresholded[1:2, :] .== 0.9) --> true "test Ll7Ks2"
        @fact all(thresholded[:, 1:2] .== 0.9) --> true "test DL17A5"
        @fact all(thresholded[8:10, :] .== 0.9) --> true "test p6iPF2"
        @fact all(thresholded[:, 8:10] .== 0.9) --> true "test U1rPYX"

        img[3, 5] = 0.7
        thresholded = Images.hysteresis_thresholding(img, 0.9, 0.6)
        @fact all(thresholded[4:6, 4:6] .== 0.9) --> true "test sAo0s9"
        @fact all(thresholded[3:7, 3] .== 0.0) --> true "test FL5swa"
        @fact all(thresholded[3:7, 7] .== 0.0) --> true "test H1IfJw"
        @fact all(thresholded[7, 3:7] .== 0.0) --> true "test SC1yNK"
        @fact all(vec(thresholded[3, 3:7]) .== [0.0, 0.0, 0.9, 0.0, 0.0]) --> true "test 0YsjuY"
        @fact all(thresholded[1:2, :] .== 0.9) --> true "test 9nWiWC"
        @fact all(thresholded[:, 1:2] .== 0.9) --> true "test VhcOoO"
        @fact all(thresholded[8:10, :] .== 0.9) --> true "test qjmIqQ"
        @fact all(thresholded[:, 8:10] .== 0.9) --> true "test MWZ3sx"
    end

    SZ=5
    cb_array    = checkerboard(SZ,3)

    cb_gray = grayim(cb_array)
    cb_rgb = convert(Array{RGB}, cb_gray)
    cb_rgbf64 = convert(Array{RGB{Float64}}, cb_gray)

    # TODO: now that all of these behave nearly the same, put in a
    # loop over the array type (only have to adjust tests for `red`)
    @compat context("Checkerboard") do
        for method in ["sobel", "prewitt", "ando3", "ando4", "ando5"]
            ## Checkerboard array

            (agy, agx) = imgradients(cb_array, method)
            amag = magnitude(agx, agy)
            agphase = phase(agx, agy)
            @fact (amag, agphase) --> magnitude_phase(agx, agy) "test 80oF4m"

            @fact agx[1,SZ]   --> less_than(0.0) "test 4N2OkG" # white to black transition
            @fact agx[1,2*SZ] --> greater_than(0.0) "test QF4xEq" # black to white transition
            @fact agy[SZ,1]   --> less_than(0.0) "test Lcw3dQ"  # white to black transition
            @fact agy[2*SZ,1] --> greater_than(0.0) "test XIhUPC" # black to white transition

            # Test direction of increasing gradient
            @fact cos(agphase[1,SZ])   - (-1.0) --> less_than(EPS) "test MDZZJR"  # increasing left  (=  pi   radians)
            @fact cos(agphase[1,2*SZ]) -   1.0  --> less_than(EPS) "test UI49Lu"  # increasing right (=   0   radians)
            @fact sin(agphase[SZ,1])   -   1.0  --> less_than(EPS) "test 8L7FiR"  # increasing up    (=  pi/2 radians)
            @fact sin(agphase[2*SZ,1]) - (-1.0) --> less_than(EPS) "test o0CIl6"  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @fact all((cos.(agphase) .* cos.(aorient) .+
                       sin.(agphase) .* sin.(aorient) .< EPS) |
                      ((agphase .== 0.0) & (aorient .== 0.0))) --> true  "test zPGMub"
                      # this part is where both are zero because there is no gradient

            ## Checkerboard with Gray pixels

            (gyg, gxg) = imgradients(cb_gray, method)
            magg = magnitude(gxg, gyg)
            gphaseg = phase(gxg, gyg)
            @fact (magg, gphaseg) --> magnitude_phase(gxg, gyg) "test AM0uph"

            @fact gyg[SZ,1]   --> less_than(0.0) "test cQKPsb"     # white to black transition
            @fact gyg[2*SZ,1] --> greater_than(0.0) "test 5wgkII"  # black to white transition
            @fact gxg[1,SZ]   --> less_than(0.0) "test EeAjQ1"     # white to black transition
            @fact gxg[1,2*SZ] --> greater_than(0.0) "test dua7cD"  # black to white transition

            @fact cos(gphaseg[1,SZ])   - (-1.0) --> less_than(EPS) "test TQ0aJN"  # increasing left  (=  pi   radians)
            @fact cos(gphaseg[1,2*SZ]) -   1.0  --> less_than(EPS) "test Mf2h17"  # increasing right (=   0   radians)
            @fact sin(gphaseg[SZ,1])   -   1.0  --> less_than(EPS) "test fKv3Vl"  # increasing up    (=  pi/2 radians)
            @fact sin(gphaseg[2*SZ,1]) - (-1.0) --> less_than(EPS) "test CZQeGF"  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orientg = orientation(gxg, gyg)
            @fact all((cos.(gphaseg) .* cos.(orientg) .+
                       sin.(gphaseg) .* sin.(orientg) .< EPS) |
                      ((gphaseg .== 0.0) & (orientg .== 0.0))) --> true "test BsoAOy"
                      # this part is where both are zero because there is no gradient

            ## Checkerboard with RBG pixels

            (gy_rgb, gx_rgb) = imgradients(cb_rgb, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @fact (mag_rgb, gphase_rgb) --> magnitude_phase(gx_rgb, gy_rgb) "test YfK3X2"

            @fact red(gy_rgb[SZ,1])   --> less_than(0.0) "test 1WYzkc"     # white to black transition
            @fact red(gy_rgb[2*SZ,1]) --> greater_than(0.0) "test rZ10Ar"  # black to white transition
            @fact red(gx_rgb[1,SZ])   --> less_than(0.0) "test aoRWZJ"     # white to black transition
            @fact red(gx_rgb[1,2*SZ]) --> greater_than(0.0) "test kNDrlj"  # black to white transition

            @fact cos(gphase_rgb[1,SZ])   - (-1.0) --> less_than(EPS) "test qcAgh5"  # increasing left  (=  pi   radians)
            @fact cos(gphase_rgb[1,2*SZ]) -   1.0  --> less_than(EPS) "test Mb9flX"  # increasing right (=   0   radians)
            @fact sin(gphase_rgb[SZ,1])   -   1.0  --> less_than(EPS) "test Wxzwl1"  # increasing up    (=  pi/2 radians)
            @fact sin(gphase_rgb[2*SZ,1]) - (-1.0) --> less_than(EPS) "test iCilcm"  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @fact all((cos.(gphase_rgb) .* cos.(orient_rgb) .+
                       sin.(gphase_rgb) .* sin.(orient_rgb) .< EPS) |
                      ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0))) --> true "test vrd5mw"
                      # this part is where both are zero because there is no gradient

            ## Checkerboard Image with RBG{Float64} pixels

            (gy_rgb, gx_rgb) = imgradients(cb_rgbf64, method)
            mag_rgb = magnitude(gx_rgb, gy_rgb)
            gphase_rgb = phase(gx_rgb, gy_rgb)
            @fact (mag_rgb, gphase_rgb) --> magnitude_phase(gx_rgb, gy_rgb) "test Z7WRsV"

            @fact red(gy_rgb[SZ,1])   --> less_than(0.0) "test B5N6ez"     # white to black transition
            @fact red(gy_rgb[2*SZ,1]) --> greater_than(0.0) "test BevGJV"  # black to white transition
            @fact red(gx_rgb[1,SZ])   --> less_than(0.0) "test l8bc6e"     # white to black transition
            @fact red(gx_rgb[1,2*SZ]) --> greater_than(0.0) "test eZ6Plt"  # black to white transition

            @fact cos(gphase_rgb[1,SZ])   - (-1.0) --> less_than(EPS) "test M3VjU7"  # increasing left  (=  pi   radians)
            @fact cos(gphase_rgb[1,2*SZ]) -   1.0  --> less_than(EPS) "test AoPGDf"  # increasing right (=   0   radians)
            @fact sin(gphase_rgb[SZ,1])   -   1.0  --> less_than(EPS) "test ls8xr8"  # increasing up    (=  pi/2 radians)
            @fact sin(gphase_rgb[2*SZ,1]) - (-1.0) --> less_than(EPS) "test 49BiFi"  # increasing down  (= -pi/2 radians)

            # Test that orientation is perpendicular to gradient
            orient_rgb = orientation(gx_rgb, gy_rgb)
            @fact all((cos.(gphase_rgb) .* cos.(orient_rgb) .+
                       sin.(gphase_rgb) .* sin.(orient_rgb) .< EPS) |
                      ((gphase_rgb .== 0.0) & (orient_rgb .== 0.0))) --> true "test EW5CFV"
                      # this part is where both are zero because there is no gradient
        end
    end

    @compat context("Diagonals") do
        # Create an image with white along diagonals -2:2 and black elsewhere
        m = zeros(UInt8, 20,20)
        for i = -2:2; m[diagind(m,i)] = 0xff; end

        for method in ["sobel", "prewitt", "ando3", "ando4", "ando5"]
            ## Diagonal array

            (agy, agx) = imgradients(m, method)
            amag = magnitude(agx, agy)
            agphase = phase(agx, agy)
            @fact magnitude_phase(agx, agy) --> (amag, agphase) "test magphase"

            @fact agx[7,9]  --> less_than(0.0) "test 3cHL0U"     # white to black transition
            @fact agx[10,8] --> greater_than(0.0) "test ZnqMYL"  # black to white transition
            @fact agy[10,8] --> less_than(0.0) "test iIZNpr"     # white to black transition
            @fact agy[7,9]  --> greater_than(0.0) "test olF7SY"  # black to white transition

            # Test direction of increasing gradient
            @fact abs(agphase[10,8] -    pi/4 ) --> less_than(EPS) "test 2to4XR"  # lower edge (increasing up-right  =   pi/4 radians)
            @fact abs(agphase[7,9]  - (-3pi/4)) --> less_than(EPS) "test 4n7Fd5"  # upper edge (increasing down-left = -3pi/4 radians)

            # Test that orientation is perpendicular to gradient
            aorient = orientation(agx, agy)
            @fact all((cos.(agphase) .* cos.(aorient) .+
                       sin.(agphase) .* sin.(aorient) .< EPS) |
                      ((agphase .== 0.0) & (aorient .== 0.0))) --> true "test cYbfiH"
                    # this part is where both are zero because there is no gradient
        end
    end

    # Nonmaximal suppression

    function thin_edges(img)
        # Get orientation
        gy,gx = imgradients(img)
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

        @fact abs(r - 1/6) --> less_than(EPS) "test fmHvvQ"

        # Location and value at peak
        peakloc = r*1.35 + 3
        peakval = a*r^2 + b*r + c

        transposed = spatialorder(img)[1] == "x"
        horizontal = which == :horizontal

        test_axis1 = transposed ⊻ !horizontal

        if test_axis1
            @fact all(t[:,[1,2,4,5]] .== 0) --> true "test 4HIP6f"
            @fact all(t[:,3]   .== peakval) --> true "test ZITaq0"
            @fact all(s[:,[1,2,4,5]] .== zero(Graphics.Point)) --> true "test 0JGsoU"
        else
            @fact all(t[[1,2,4,5],:] .== 0) --> true "test OZUJOS"
            @fact all(t[3,:]   .== peakval) --> true "test IEOYk3"
            @fact all(s[[1,2,4,5],:] .== zero(Graphics.Point)) --> true "test pnnCRT"
        end

        if transposed
            if which == :horizontal
                @fact     [pt.x for pt in s[:,3]]  --> [1:5;] "test 7uIThT"
                @fact all([pt.y for pt in s[:,3]] .== peakloc) --> true "test OTE4OQ"
            else
                @fact all([pt.x for pt in s[3,:]] .== peakloc) --> true "test HZSUky"
                @fact     [pt.y for pt in s[3,:]]  --> [1:5;] "test 3PyK02"
            end
        else
            if which == :horizontal
                @fact     [pt.x for pt in s[3,:]]  --> [1:5;] "test oydYR7"
                @fact all([pt.y for pt in s[3,:]] .== peakloc) --> true "test pwrLbc"
            else
                @fact all([pt.x for pt in s[:,3]] .== peakloc) --> true "test 035uAN"
                @fact     [pt.y for pt in s[:,3]]  --> [1:5;] "test A7DGMj"
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

        nms_test_horiz_vert(m, :vertical)
    end

    context("Nonmax suppression horizontal edge") do
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

    @fact (r - 1/6) --> less_than(EPS) "test xuJ9PL"

    transposed = false

    # Location and value at peak

    x_peak_offset, y_peak_offset = r*fr, -r*fr
    peakval = a*r^2 + b*r + c

    @fact all(diag(data(t))[2:4] .== peakval) --> true "test UpmYpg" # Edge pixels aren't interpolated here
    @fact all(t - diagm(diag(data(t))) .== 0) --> true "test sK2s90"

    diag_s = copyproperties(s, diagm(diag(data(s))))
    @fact s --> diag_s "test 4xwjQB"

    @fact all([pt.x for pt in diag(data(s))[2:4]] - ((2:4) + x_peak_offset) .< EPS) --> true "test luz17v"
    @fact all([pt.y for pt in diag(data(s))[2:4]] - ((2:4) + y_peak_offset) .< EPS) --> true "test RYzBTz"

end

    context("Nonmax suppression diagonal") do
        # Test image: diagonal edge
        m = [7.0  6.0  5.0  0.0  0.0
             5.0  7.0  6.0  5.0  0.0
             3.0  5.0  7.0  6.0  5.0
             0.0  3.0  5.0  7.0  6.0
             0.0  0.0  3.0  5.0  7.0]

        nms_test_diagonal(m)
    end

end
