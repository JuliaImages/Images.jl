using FactCheck, Images, Colors, FixedPointNumbers

facts("Read remote") do
    urlbase = "http://www.imagemagick.org/Usage/images/"
    workdir = joinpath(tempdir(), "Images")
    writedir = joinpath(workdir, "write")
    if !isdir(workdir)
        mkdir(workdir)
    end
    if !isdir(writedir)
        mkdir(writedir)
    end

    function getfile(name)
        file = joinpath(workdir, name)
        if !isfile(file)
            file = download(urlbase*name, file)
        end
        file
    end

    context("Gray") do
        file = getfile("jigsaw_tmpl.png")
        img = imread(file)
        @fact colorspace(img) --> "Gray"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> Gray{Ufixed8}
        outname = joinpath(writedir, "jigsaw_tmpl.png")
        imwrite(img, outname)
        imgc = imread(outname)
        @fact img.data --> imgc.data
        @fact reinterpret(Uint32, data(map(mapinfo(RGB24, img), img))) -->
            map(x->x&0x00ffffff, reinterpret(Uint32, data(map(mapinfo(ARGB32, img), img))))
        @fact mapinfo(UInt32, img) --> mapinfo(RGB24, img)
        @fact data(convert(Image{Gray{Float32}}, img)) --> float32(data(img))
        mapi = mapinfo(RGB{Ufixed8}, img)
        imgrgb8 = map(mapi, img)
        @fact imgrgb8[1,1].r --> img[1].val
        open(outname, "w") do file
            writemime(file, "image/png", img)
        end
    end
    
    context("Gray with alpha channel") do
        file = getfile("wmark_image.png")
        img = imread(file)
        @fact colorspace(img) --> "GrayA"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @fact eltype(img) --> Images.ColorTypes.GrayA{Ufixed8}
        @linux_only begin
            outname = joinpath(writedir, "wmark_image.png")
            imwrite(img, outname)
            sleep(0.2)
            imgc = imread(outname)
            @fact img.data --> imgc.data
            open(outname, "w") do file
                writemime(file, "image/png", img)
            end
        end
        @fact reinterpret(Uint32, data(map(mapinfo(RGB24, img), img))) -->
            map(x->x&0x00ffffff, reinterpret(Uint32, data(map(mapinfo(ARGB32, img), img))))
        @fact mapinfo(Uint32, img) --> mapinfo(ARGB32, img)
    end

    context("RGB") do
        file = getfile("rose.png")
        img = imread(file)
        # Mac reader reports RGB4, imagemagick reports RGB
        @osx? begin
            @fact colorspace(img) --> "RGB4"
        end : @fact colorspace(img) --> "RGB"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @osx? begin
            @fact eltype(img) --> RGB4{Ufixed8}
        end : @fact eltype(img) --> RGB{Ufixed8}
        outname = joinpath(writedir, "rose.ppm")
        imwrite(img, outname)
        imgc = imread(outname)
        T = eltype(imgc)
        lim = limits(imgc)
        @fact typeof(lim[1]) --> typeof(lim[2])  # issue #62
        @fact typeof(lim[2]) --> T  # issue #62
        # Why does this one fail on OSX??
        @osx? nothing : @fact img.data --> imgc.data
        @fact reinterpret(Uint32, data(map(mapinfo(RGB24, img), img))) -->
            map(x->x&0x00ffffff, reinterpret(Uint32, data(map(mapinfo(ARGB32, img), img))))
        @fact mapinfo(Uint32, img) --> mapinfo(RGB24, img)
        mapi = mapinfo(RGB{Ufixed8}, img)
        imgrgb8 = map(mapi, img)
        @fact data(imgrgb8) --> data(img)
        convert(Array{Gray{Ufixed8}}, img)
        convert(Image{Gray{Ufixed8}}, img)
        convert(Array{Gray}, img)
        convert(Image{Gray}, img)
        imgs = separate(img)
        @fact permutedims(convert(Image{Gray}, imgs), [2,1]) --> convert(Image{Gray}, img)
        # Make sure that all the operations in README will work:
        buf = Array(Uint32, size(img))
        buft = Array(Uint32, reverse(size(img)))
        uint32color(img)
        uint32color!(buf, img)
        imA = convert(Array, img)
        uint32color(imA)
        uint32color!(buft, imA)
        uint32color(imgs)
        uint32color!(buft, imgs)
        imr = reinterpret(Ufixed8, img)
        uint32color(imr)
        uint32color!(buf, imr)
        @osx? nothing : begin
            imhsv = convert(Image{HSV}, float32(img))
            uint32color(imhsv)
            uint32color!(buf, imhsv)
            @fact pixelspacing(restrict(img)) --> [2.0,2.0]
        end
        outname = joinpath(writedir, "rose.png")
        open(outname, "w") do file
            writemime(file, "image/png", img)
        end
    end

    context("RGBA with 16 bit depth") do
        file = getfile("autumn_leaves.png")
        img = imread(file)
        @osx? begin
            @fact colorspace(img) --> "RGBA"
        end : @fact colorspace(img) --> "BGRA"
        @fact ndims(img) --> 2
        @fact colordim(img) --> 0
        @osx? begin
            @fact eltype(img) --> Images.ColorTypes.RGBA{Ufixed16}
        end : @fact eltype(img) --> Images.ColorTypes.BGRA{Ufixed16}
        outname = joinpath(writedir, "autumn_leaves.png")
        @osx? nothing : begin
            imwrite(img, outname)
            sleep(0.2)
            imgc = imread(outname)
            @fact img.data --> imgc.data
            @fact reinterpret(Uint32, data(map(mapinfo(RGB24, img), img))) -->
                map(x->x&0x00ffffff, reinterpret(Uint32, data(map(mapinfo(ARGB32, img), img))))
            @fact mapinfo(Uint32, img) --> mapinfo(ARGB32, img)
        end
        open(outname, "w") do file
            writemime(file, "image/png", img)
        end
    end
    
    context("Indexed") do
        file = getfile("present.gif")
        img = imread(file)
        @fact nimages(img) --> 1
        @fact reinterpret(Uint32, data(map(mapinfo(RGB24, img), img))) -->
            map(x->x&0x00ffffff, reinterpret(Uint32, data(map(mapinfo(ARGB32, img), img))))
        @fact mapinfo(Uint32, img) --> mapinfo(RGB24, img)
        outname = joinpath(writedir, "present.png")
        open(outname, "w") do file
            writemime(file, "image/png", img)
        end
    end

    context("Images with a temporal dimension") do
        fname = "swirl_video.gif"
        #fname = "bunny_anim.gif"  # this one has transparency but LibMagick gets confused about its size
        file = getfile(fname)  # this also has transparency
        img = imread(file)
        @fact timedim(img) --> 3
        @fact nimages(img) --> 26
        outname = joinpath(writedir, fname)
        imwrite(img, outname)
        imgc = imread(outname)
        # Something weird happens after the 2nd image (compression?), and one starts getting subtle differences.
        # So don't compare the values.
        # Also take the opportunity to test some things with temporal images
        @fact storageorder(img) --> ["x", "y", "t"]
        @fact haskey(img, "timedim") --> true
        @fact timedim(img) --> 3
        s = getindexim(img, 1:5, 1:5, 3)
        @fact timedim(s) --> 0
        s = sliceim(img, :, :, 5)
        @fact timedim(s) --> 0
        imgt = sliceim(img, "t", 1)
        @fact reinterpret(Uint32, data(map(mapinfo(RGB24, imgt), imgt))) -->
            map(x->x&0x00ffffff, reinterpret(Uint32, data(map(mapinfo(ARGB32, imgt), imgt))))
    end

    context("Extra properties") do
        @osx? nothing : begin
            file = getfile("autumn_leaves.png")
            # List properties
            extraProps = imread(file, extrapropertynames=true)

            img = imread(file,extraprop=extraProps)
            props = properties(img)
            for key in extraProps
                @fact haskey(props, key) --> true
                @fact props[key] --> anything
            end
            img = imread(file, extraprop=extraProps[1])
            props = properties(img)
            @fact haskey(props, extraProps[1]) --> true
            @fact props[extraProps[1]] --> anything
    
            println("The following \"Undefined property\" warning indicates normal operation")
            img = imread(file, extraprop="Non existing property")
            props = properties(img)
            @fact haskey(props, "Non existing property") --> true
            @fact props["Non existing property"] --> not(anything)
        end
    end
end
