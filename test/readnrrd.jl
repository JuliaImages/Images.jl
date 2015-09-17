using FactCheck, Images, Colors, FixedPointNumbers

testing_units = Int == Int64
if testing_units
    using SIUnits, SIUnits.ShortUnits
end

facts("Read NRRD") do
    workdir = joinpath(tempdir(), "Images")
    writedir = joinpath(workdir, "write")
    if !isdir(workdir)
        mkdir(workdir)
    end
    if !isdir(writedir)
        mkdir(writedir)
    end

    context("Gray, raw") do
        img = imread(joinpath(dirname(@__FILE__), "io", "small.nrrd"))
        @fact colorspace(img) --> "Gray"
        @fact ndims(img) --> 3
        @fact colordim(img) --> 0
        @fact eltype(img) --> Float32
        outname = joinpath(writedir, "small.nrrd")
        imwrite(img, outname)
        imgc = imread(outname)
        @fact img.data --> imgc.data
    end
    
    context("Units") do
        img = imread(joinpath(dirname(@__FILE__), "io", "units.nhdr"))
        ps = pixelspacing(img)
        @fact ps[1]/(0.1*Milli*Meter) --> roughly(1)
        @fact ps[2]/(0.2*Milli*Meter)  --> roughly(1)
        @fact ps[3]/(1*Milli*Meter)  --> roughly(1)
    end
    
    context("Gray, compressed (gzip)") do
        img = imread(joinpath(dirname(@__FILE__), "io", "smallgz.nrrd"))
        @fact colorspace(img) --> "Gray"
        @fact ndims(img) --> 3
        @fact colordim(img) --> 0
        @fact eltype(img) --> Float32
        outname = joinpath(writedir, "smallgz.nrrd")
        imwrite(img, outname)
        imgc = imread(outname)
        @fact img.data --> imgc.data
    end
    
    context("Time is 4th dimension") do
        img = imread(joinpath(dirname(@__FILE__), "io", "small_time.nrrd"))
        @fact timedim(img) --> 4
        outname = joinpath(writedir, "small_time.nrrd")
        imwrite(img, outname)
        imgc = imread(outname)
        @fact img.data --> imgc.data    
    end
end
