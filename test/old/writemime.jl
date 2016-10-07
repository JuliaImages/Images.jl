using Images, FactCheck, Colors, FixedPointNumbers

facts("show (MIME)") do
    # Test that we remembered to turn off Colors.jl's colorswatch display
    @fact mimewritable(MIME("image/svg+xml"), rand(Gray{U8}, 5, 5)) --> false
    @fact mimewritable(MIME("image/svg+xml"), rand(RGB{U8},  5, 5)) --> false
    @fact mimewritable(MIME("image/png"), rand(Gray{U8}, 5, 5)) --> true
    @fact mimewritable(MIME("image/png"), rand(RGB{U8},  5, 5)) --> true
    workdir = joinpath(tempdir(), "Images")
    if !isdir(workdir)
        mkdir(workdir)
    end
    context("no compression or expansion") do
        A = U8[0.01 0.99; 0.25 0.75]
        fn = joinpath(workdir, "writemime.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), grayim(A), minpixels=0, maxpixels=typemax(Int))
        end
        b = convert(Image{Gray{U8}}, load(fn))
        @fact data(b) --> A
    end
    context("small images (expansion)") do
        A = U8[0.01 0.99; 0.25 0.75]
        fn = joinpath(workdir, "writemime.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), grayim(A), minpixels=5, maxpixels=typemax(Int))
        end
        b = convert(Image{Gray{U8}}, load(fn))
        @fact data(b) --> A[[1,1,2,2],[1,1,2,2]]
    end
    context("big images (use of restrict)") do
        A = U8[0.01 0.4 0.99; 0.25 0.8 0.75; 0.6 0.2 0.0]
        Ar = restrict(A)
        fn = joinpath(workdir, "writemime.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), grayim(A), minpixels=0, maxpixels=5)
        end
        b = convert(Image{Gray{U8}}, load(fn))
        @fact data(b) --> convert(Array{U8}, Ar)
        # a genuinely big image (tests the defaults)
        abig = grayim(rand(UInt8, 1024, 1023))
        fn = joinpath(workdir, "big.png")
        open(fn, "w") do file
            show(file, MIME("image/png"), abig, maxpixels=10^6)
        end
        b = convert(Image{Gray{U8}}, load(fn))
        abigui = convert(Array{UFixed8,2}, data(restrict(abig, (1,2))))
        @fact data(b) --> abigui
    end
end
