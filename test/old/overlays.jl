using FactCheck, Images, Colors, FixedPointNumbers
using Compat

facts("Overlay") do
    gray = linspace(0.0, 1.0, 5)
    context("One") do
        ovr = Images.Overlay((2gray, 2gray), (RGB(1, 0, 0), RGB(0, 0, 1)), (Clamp{Float64}(), Clamp{Float64}()))
        @fact ovr[1] --> RGB(0, 0, 0) "test XcqmPT"
        @fact ovr[2] --> RGB{N0f8}(0.5, 0, 0.5) "test yAFSVQ"
        @fact ovr[3] --> ovr[4] "test dyKCV9"
        @fact ovr[4] --> ovr[5] "test d4aUI1"
        @fact ovr[5] --> exactly(RGB(1, 0, 1)) "test UF1jSj"
        @fact eltype(ovr) --> RGB{N0f8} "test Es9OnV"
        @fact length(ovr) --> 5 "test 04ptpL"
        @fact size(ovr) --> (5,) "test iE5wc4"
        @fact size(ovr, 1) --> 5 "test OvtQ4m"
        @fact size(ovr, 2) --> 1 "test O1oIKi"
        @fact nchannels(ovr) --> 2 "test vhNvBL"
        @fact raw(ovr) --> [0x00 0x80 0xff 0xff 0xff;
                            0x00 0x00 0x00 0x00 0x00;
                            0x00 0x80 0xff 0xff 0xff] "test vLlExB"
        @fact separate(ovr) --> N0f8[0   0   0;
                                        0.5 0   0.5;
                                        1   0   1;
                                        1   0   1;
                                        1   0   1] "test XrmTTp"
        iob = IOBuffer()
        show(iob, ovr)  # exercise only
    end

    context("Two") do
        ovr = Images.Overlay((gray, 0*gray), (RGB{N0f8}(1, 0, 1), RGB{N0f8}(0, 1, 0)), ((0, 1), (0, 1)))
        @fact eltype(ovr) --> RGB{N0f8} "test u7gavU"
        ovr = collect(ovr)
        s = similar(ovr)
        @fact typeof(s) --> Vector{RGB{N0f8}} "test qOV0Iu"
        @fact length(s) --> 5 "test HxDDe4"
        s = similar(ovr, RGB{Float32})
        @fact isa(s, Vector{RGB{Float32}}) --> true "test tPX9bh"
        @fact length(s) --> 5 "test CEJl6T"
        s = similar(ovr, RGB{Float32}, (3, 2))
        @fact isa(s, Matrix{RGB{Float32}}) --> true "test qQvSYP"
        @fact size(s) --> (3, 2) "test uoSePw"
        buf = Images.uint32color(ovr)
        gray8 = round(UInt8, 255*gray)
        nogreen = [convert(UInt32, g)<<16 | convert(UInt32, g) for g in gray8]
        @fact buf --> nogreen "test XOvI8c"
    end

    context("Three") do
        ovr = Images.Overlay((gray, 0*gray), [RGB(1, 0, 1), RGB(0, 1, 0)], ([0, 1], [0, 1]))
        @fact_throws ErrorException Images.Overlay((gray, 0*gray), (RGB(1, 0, 1), RGB(0, 1, 0)), ((0,), (0, 1))) "test AKzBSc"
        @fact_throws ErrorException ovr[1] = RGB(0.2, 0.4, 0.6) "test uxe7aR"
    end

    context("Four") do
        img1 = Images.Image(gray)
        ovr = Images.OverlayImage((2gray, img1), (RGB{Float32}(1, 0, 1), RGB{Float32}(0, 1, 0)), ((0, 1),(0, 1)))
        @fact isa(ovr, Images.Image) --> true "test AcmCRK"
        @fact haskey(ovr, "colorspace") --> false "test W1xBNq"
        @fact Images.colorspace(ovr) --> "RGB" "test me6TWH"
        @fact ovr[2] --> RGB{Float32}(0.5, 0.25, 0.5) "test lznPxT"
        a = rand(Float32, 3, 2)
        b = rand(Float32, 3, 2)
        ovr = Images.OverlayImage((a, b), (RGB{Float32}(1, 0, 1), RGB{Float32}(0, 1, 0)), ((0, 1), (0, 1)))
        @fact isa(ovr, Images.Image) --> true "test 8nGPCi"
        @fact abs(ovr[1, 2] - RGB{Float32}(a[1, 2], b[1, 2], a[1, 2])) --> roughly(0, atol=1e-5) "test 5N74oJ"
    end

    context("permutation") do
        L1 = convert(Image{Gray}, rand(Float32, 10,10))
        L2 = convert(Image{Gray}, rand(Float32, 10,10))
        L3 = convert(Image{Gray}, rand(Float32, 10,10))
        overlay = OverlayImage((L1, L2, L3), (colorant"red", colorant"blue", colorant"green"), ((0,1),(0,1),(0,1)))
        permutedims(overlay, [2,1])
        permutedims(data(overlay), [2,1])
    end
end
