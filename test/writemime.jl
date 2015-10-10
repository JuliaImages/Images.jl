writemime_(io::IO, ::MIME"image/png", img::AbstractImage) = serialize(io, map(Images.mapinfo_writemime(img), img))
facts("Writemime") do
	workdir = joinpath(tempdir(), "Images")
    if !isdir(workdir)
        mkdir(workdir)
    end
    context("use of restrict") do
        abig = grayim(rand(UInt8, 1024, 1023))
        fn = joinpath(workdir, "big.png")
        open(fn, "w") do file
            writemime(file, MIME("image/png"), abig, maxpixels=10^6)
        end
        b = open(fn, "r") do io 
        	deserialize(io)
        end
        abigui = convert(Array{Ufixed8,2}, data(restrict(abig, (1,2))))
        @fact data(b) --> convert(Array{Ufixed8,2}, data(restrict(abig, (1,2))))
    end
end
