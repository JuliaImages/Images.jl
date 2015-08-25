using FactCheck, Images, Colors, FixedPointNumbers
@osx_only using TestImages

facts("Read stream") do
    @osx_only begin
        # OS X only for the moment, because TestImages is only
        # required for OS X (see test/REQUIRE).  That seems to cause
        # a testing failure on Travis for Linux.
        # I assume that at some point, streaming will expand and other tests
        # will be needed, at which point we can open up this test
        workdir = joinpath(tempdir(), "ImagesStreamTest")
        testfile = joinpath(workdir, "jigsaw_tmpl.png")
        if !isfile(testfile)
            mkpath(workdir)
            download("http://www.imagemagick.org/Usage/images/jigsaw_tmpl.png", testfile)
        end
        open(testfile, "r") do io
            img = imread(io, Images.OSXNative)
        end
        @pending something --> true
        rm(workdir, recursive=true)
    end
end
