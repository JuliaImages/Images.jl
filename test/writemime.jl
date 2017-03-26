using Images, Colors, FixedPointNumbers
using Base.Test

@testset "show (MIME)" begin
    # Test that we remembered to turn off Colors.jl's colorswatch display
    @test !mimewritable(MIME("image/svg+xml"), rand(Gray{N0f8}, 5, 5))
    @test !mimewritable(MIME("image/svg+xml"), rand(RGB{N0f8},  5, 5))
    @test mimewritable(MIME("image/png"), rand(Gray{N0f8}, 5, 5))
    @test mimewritable(MIME("image/png"), rand(RGB{N0f8},  5, 5))
    workdir = joinpath(tempdir(), "Images")
    if !isdir(workdir)
        mkdir(workdir)
    end
    @testset "no compression or expansion" begin
        A = N0f8[0.01 0.99; 0.25 0.75]
        fn = joinpath(workdir, "writemime.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), Gray.(A), minpixels=0, maxpixels=typemax(Int))
        end
        b = load(fn)
        @test b == A
    end
    @testset "small images (expansion)" begin
        A = N0f8[0.01 0.99; 0.25 0.75]
        fn = joinpath(workdir, "writemime.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), Gray.(A), minpixels=5, maxpixels=typemax(Int))
        end
        @test load(fn) == A[[1,1,2,2],[1,1,2,2]]
    end
    @testset "big images (use of restrict)" begin
        A = N0f8[0.01 0.4 0.99; 0.25 0.8 0.75; 0.6 0.2 0.0]
        Ar = restrict(A)
        fn = joinpath(workdir, "writemime.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), Gray.(A), minpixels=0, maxpixels=5)
        end
        @test load(fn) == N0f8.(Ar)
        # a genuinely big image (tests the defaults)
        abig = colorview(Gray, normedview(rand(UInt8, 1024, 1023)))
        fn = joinpath(workdir, "big.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), abig, maxpixels=10^6)
        end
        b = load(fn)
        @test b == N0f8.(restrict(abig, (1,2)))
    end
    @testset "display matrix of images" begin
        img() = colorview(Gray, rand([0.250 0.5; 0.75 1.0], rand(2:10), rand(2:10)))
        io = IOBuffer()
        # test that these methods don't fail
        show(io, MIME"text/html"(), [img() for i=1:2])
        show(io, MIME"text/html"(), [img() for i=1:2, j=1:2])
        show(io, MIME"text/html"(), [img() for i=1:2, j=1:2, k=1:2])
    end
end

nothing
