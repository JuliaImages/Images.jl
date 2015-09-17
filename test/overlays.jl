using FactCheck, Images, Colors, FixedPointNumbers

facts("Overlay") do
    gray = linspace(0.0, 1.0, 5)
    context("One") do
        ovr = Images.Overlay((2gray, 2gray), (RGB(1, 0, 0), RGB(0, 0, 1)), (Clamp{Float64}(), Clamp{Float64}()))
        @fact ovr[1] --> RGB(0, 0, 0)
        @fact ovr[2] --> RGB{U8}(0.5, 0, 0.5)
        @fact ovr[3] --> ovr[4]
        @fact ovr[4] --> ovr[5]
        @fact ovr[5] --> exactly(RGB(1, 0, 1))
        @fact eltype(ovr) --> RGB{U8}
        @fact length(ovr) --> 5
        @fact size(ovr) --> (5,)
        @fact size(ovr, 1) --> 5
        @fact size(ovr, 2) --> 1
        @fact nchannels(ovr) --> 2
        iob = IOBuffer()
        show(iob, ovr)  # exercise only
    end

    context("Two") do
        local ovr = Images.Overlay((gray, 0*gray), (RGB{Ufixed8}(1, 0, 1), RGB{Ufixed8}(0, 1, 0)), ((0, 1), (0, 1)))
        @fact eltype(ovr) --> RGB{Ufixed8}
        s = similar(ovr)
        @fact isa(s, Vector{RGB{Ufixed8}}) --> true
        @fact length(s) --> 5
        s = similar(ovr, RGB{Float32})
        @fact isa(s, Vector{RGB{Float32}}) --> true
        @fact length(s) --> 5
        s = similar(ovr, RGB{Float32}, (3, 2))
        @fact isa(s, Matrix{RGB{Float32}}) --> true
        @fact size(s) --> (3, 2)
        buf = Images.uint32color(ovr)
        gray8 = round(UInt8, 255*gray)
        nogreen = [convert(UInt32, g)<<16 | convert(UInt32, g) for g in gray8]
        @fact buf --> nogreen
    end

    context("Three") do
        ovr = Images.Overlay((gray, 0*gray), [RGB(1, 0, 1), RGB(0, 1, 0)], ([0, 1], [0, 1]))
        @fact_throws ErrorException Images.Overlay((gray, 0*gray), (RGB(1, 0, 1), RGB(0, 1, 0)), ((0,), (0, 1)))
        @fact_throws ErrorException ovr[1] = RGB(0.2, 0.4, 0.6)
    end

    context("Four") do
        img1 = Images.Image(gray, Dict{ASCIIString, Any}([("colorspace", "Gray"), ("spatialorder",["x"])]))
        ovr = Images.OverlayImage((2gray, img1), (RGB{Float32}(1, 0, 1), RGB{Float32}(0, 1, 0)), ((0, 1),(0, 1)))
        @fact isa(ovr, Images.Image) --> true
        @fact haskey(ovr, "colorspace") --> false
        @fact Images.colorspace(ovr) --> "RGB"
        @fact ovr[2] --> RGB{Float32}(0.5, 0.25, 0.5)
        a = rand(Float32, 3, 2)
        b = rand(Float32, 3, 2)
        ovr = Images.OverlayImage((a, b), (RGB{Float32}(1, 0, 1), RGB{Float32}(0, 1, 0)), ((0, 1), (0, 1)))
        @fact isa(ovr, Images.Image) --> true
        @fact abs(ovr[1, 2] - RGB{Float32}(a[1, 2], b[1, 2], a[1, 2])) --> roughly(0, atol=1e-5)
    end
end
