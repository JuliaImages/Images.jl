using FactCheck, Images, Colors, FixedPointNumbers

facts("IO") do
    workdir = joinpath(tempdir(), "Images")
    if !isdir(workdir)
        mkdir(workdir)
    end

    context("Gray png") do
        a = rand(2,2)
        aa = convert(Array{Ufixed8}, a)
        fn = joinpath(workdir, "2by2.png")
        Images.imwrite(a, fn)
        b = Images.imread(fn)
        @fact convert(Array, b) --> aa
        Images.imwrite(aa, fn)
        b = Images.imread(fn)
        @fact convert(Array, b) --> aa
        aaimg = Images.grayim(aa)
        open(fn, "w") do file
            writemime(file, MIME("image/png"), aaimg, minpixels=0)
        end
        b = Images.imread(fn)
        @fact b --> aaimg
        aa = convert(Array{Ufixed16}, a)
        Images.imwrite(aa, fn)
        b = Images.imread(fn)
        @fact convert(Array, b) --> aa
        aa = Ufixed12[0.6 0.2;
                      1.4 0.8]
        open(fn, "w") do file
            writemime(file, MIME("image/png"), Images.grayim(aa), minpixels=0)
        end
        b = Images.imread(fn)
        @fact Images.data(b) --> Ufixed8[0.6 0.2;
                                        1.0 0.8]
    end

    context("Color") do
        fn = joinpath(workdir, "2by2.png")
        img = Images.colorim(rand(3,2,2))
        img24 = convert(Images.Image{RGB24}, img)
        Images.imwrite(img24, fn)
        b = Images.imread(fn)
        imgrgb8 = convert(Images.Image{RGB{Ufixed8}}, img)
        @fact Images.data(imgrgb8) --> Images.data(b)
    end

    context("Writemime's use of restrict") do
        abig = Images.grayim(rand(Uint8, 1024, 1023))
        fn = joinpath(workdir, "big.png")
        open(fn, "w") do file
            writemime(file, MIME("image/png"), abig, maxpixels=10^6)
        end
        b = Images.imread(fn)
        @fact Images.data(b) --> convert(Array{Ufixed8,2}, Images.data(Images.restrict(abig, (1,2))))
    end

    context("More writemime tests") do
        a = Images.colorim(rand(Uint8, 3, 2, 2))
        fn = joinpath(workdir, "2by2.png")
        open(fn, "w") do file
            writemime(file, MIME("image/png"), a, minpixels=0)
        end
        b = Images.imread(fn)
        @fact Images.data(b) --> Images.data(a)

        abig = Images.colorim(rand(Uint8, 3, 1021, 1026))
        fn = joinpath(workdir, "big.png")
        open(fn, "w") do file
            writemime(file, MIME("image/png"), abig, maxpixels=10^6)
        end
        b = Images.imread(fn)
        @fact Images.data(b) --> convert(Array{RGB{Ufixed8},2}, Images.data(Images.restrict(abig, (1,2))))

        # Issue #269
        abig = Images.colorim(rand(Uint16, 3, 1024, 1023))
        open(fn, "w") do file
            writemime(file, MIME("image/png"), abig, maxpixels=10^6)
        end
        b = Images.imread(fn)
        @fact Images.data(b) --> convert(Array{RGB{Ufixed8},2}, Images.data(Images.restrict(abig, (1,2))))
    end

    context("Colormap usage") do
        datafloat = reshape(linspace(0.5, 1.5, 6), 2, 3)
        dataint = round(Uint8, 254*(datafloat .- 0.5) .+ 1)  # ranges from 1 to 255
        # build our colormap
        b = RGB(0,0,1)
        w = RGB(1,1,1)
        r = RGB(1,0,0)
        cmaprgb = Array(RGB{Float64}, 255)
        f = linspace(0,1,128)
        cmaprgb[1:128] = [(1-x)*b + x*w for x in f]
        cmaprgb[129:end] = [(1-x)*w + x*r for x in f[2:end]]
        img = Images.ImageCmap(dataint, cmaprgb)
        Images.imwrite(img,joinpath(workdir,"cmap.jpg"))
        cmaprgb = Array(RGB, 255) # poorly-typed cmap, issue #336
        cmaprgb[1:128] = [(1-x)*b + x*w for x in f]
        cmaprgb[129:end] = [(1-x)*w + x*r for x in f[2:end]]
        img = Images.ImageCmap(dataint, cmaprgb)
        Images.imwrite(img,joinpath(workdir,"cmap.pbm"))
    end

    context("Alpha") do
        c = reinterpret(Images.BGRA{Ufixed8}, [0xf0884422]'')
        fn = joinpath(workdir, "alpha.png")
    Images.imwrite(c, fn)
        C = Images.imread(fn)
        # @test C[1] == c[1]  # disabled because Travis has a weird, old copy of ImageMagick for which this fails (see #261)
        Images.imwrite(reinterpret(ARGB32, [0xf0884422]''), fn)
        D = Images.imread(fn)
        # @test D[1] == c[1]
    end

    context("3D TIFF (issue #307)") do
        A = Images.grayim(rand(0x00:0xff, 2, 2, 4))
        fn = joinpath(workdir, "3d.tif")
        Images.imwrite(A, fn)
        B = Images.imread(fn)
        @fact A --> B
    end

    context("Clamping (issue #256)") do
        A = grayim(rand(2,2))
        A[1,1] = -0.4
        fn = joinpath(workdir, "2by2.png")
        @fact_throws InexactError Images.imwrite(A, fn)
        Images.imwrite(A, fn, mapi=Images.mapinfo(Images.Clamp, A))
        B = Images.imread(fn)
        A[1,1] = 0
        @fact B --> map(Ufixed8, A)
    end

    @unix_only context("Reading from a stream (issue #312)") do
        fn = joinpath(workdir, "2by2.png")
        io = open(fn)
        img = Images.imread(io, Images.ImageMagick)
        close(io)
        @fact isa(img, Images.Image) --> true
    end
end
