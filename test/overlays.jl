using Images, Colors, FixedPointNumbers
using Compat

@testset "Overlay" begin
    gray = linspace(0.0, 1.0, 5)
    @testset "One" begin
        ovr = Images.Overlay((2gray, 2gray), (RGB(1, 0, 0), RGB(0, 0, 1)), (Clamp{Float64}(), Clamp{Float64}()))
        @test ovr[1] == RGB(0,0,0)
        @test ovr[2] == RGB{U8}(0.5,0,0.5)
        @test ovr[3] == ovr[4]
        @test ovr[4] == ovr[5]
        @test ovr[5] === RGB(1,0,1)
        @test eltype(ovr) == RGB{U8}
        @test length(ovr) == 5
        @test size(ovr) == (5,)
        @test size(ovr,1) == 5
        @test size(ovr,2) == 1
        @test nchannels(ovr) == 2
        @test raw(ovr) == [0x00 0x80 0xff 0xff 0xff;
                            0x00 0x00 0x00 0x00 0x00;
                            0x00 0x80 0xff 0xff 0xff]
        @test separate(ovr) == UFixed8[0   0   0;
                                        0.5 0   0.5;
                                        1   0   1;
                                        1   0   1;
                                        1   0   1]
        iob = IOBuffer()
        show(iob, ovr)  # exercise only
    end

    @testset "Two" begin
        ovr = Images.Overlay((gray, 0*gray), (RGB{UFixed8}(1, 0, 1), RGB{UFixed8}(0, 1, 0)), ((0, 1), (0, 1)))
        @test eltype(ovr) == RGB{UFixed8}
        s = similar(ovr)
        @test typeof(s) == Vector{RGB{UFixed8}}
        @test length(s) == 5
        s = similar(ovr, RGB{Float32})
        @test isa(s,Vector{RGB{Float32}})
        @test length(s) == 5
        s = similar(ovr, RGB{Float32}, (3, 2))
        @test isa(s,Matrix{RGB{Float32}})
        @test size(s) == (3,2)
        buf = Images.uint32color(ovr)
        gray8 = round(UInt8, 255*gray)
        nogreen = reinterpret(RGB24, [convert(UInt32, g)<<16 | convert(UInt32, g) for g in gray8])
        @test buf == nogreen
    end

    @testset "Three" begin
        ovr = Images.Overlay((gray, 0*gray), [RGB(1, 0, 1), RGB(0, 1, 0)], ([0, 1], [0, 1]))
        @test_throws ErrorException Images.Overlay((gray,0gray),(RGB(1,0,1),RGB(0,1,0)),((0,),(0,1)))
        @test_throws ErrorException ovr[1] = RGB(0.2,0.4,0.6)
    end

    @testset "Four" begin
        img1 = Images.Image(gray, Dict{Compat.ASCIIString, Any}([("colorspace", "Gray"), ("spatialorder",["x"])]))
        ovr = Images.OverlayImage((2gray, img1), (RGB{Float32}(1, 0, 1), RGB{Float32}(0, 1, 0)), ((0, 1),(0, 1)))
        @test isa(ovr,Images.Image)
        @test !(haskey(ovr,"colorspace"))
        @test Images.colorspace(ovr) == "RGB"
        @test ovr[2] == RGB{Float32}(0.5,0.25,0.5)
        a = rand(Float32, 3, 2)
        b = rand(Float32, 3, 2)
        ovr = Images.OverlayImage((a, b), (RGB{Float32}(1, 0, 1), RGB{Float32}(0, 1, 0)), ((0, 1), (0, 1)))
        @test isa(ovr,Images.Image)
        @test isapprox(abs(ovr[1,2] - RGB{Float32}(a[1,2],b[1,2],a[1,2])),0,atol=1.0e-5)
    end

    @testset "permutation" begin
        L1 = convert(Image{Gray}, rand(Float32, 10,10))
        L2 = convert(Image{Gray}, rand(Float32, 10,10))
        L3 = convert(Image{Gray}, rand(Float32, 10,10))
        overlay = OverlayImage((L1, L2, L3), (colorant"red", colorant"blue", colorant"green"), ((0,1),(0,1),(0,1)))
        permutedims(overlay, [2,1])
        permutedims(data(overlay), [2,1])
    end
end
