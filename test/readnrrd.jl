using Images

const savedir = joinpath(tempdir(), "Images")
const writedir = joinpath(savedir, "write")

if !isdir(savedir)
    mkdir(savedir)
end
if !isdir(writedir)
    mkdir(writedir)
end


# Gray, raw
img = imread(joinpath(Pkg.dir(), "Images", "test", "io", "small.nrrd"))
@assert colorspace(img) == "Gray"
@assert ndims(img) == 3
@assert colordim(img) == 0
@assert eltype(img) == Float32
outname = joinpath(writedir, "small.nrrd")
imwrite(img, outname)
sleep(0.2)
imgc = imread(outname)
@assert img.data == imgc.data


# Gray, compressed (gzip)
img = imread(joinpath(Pkg.dir(), "Images", "test", "io", "smallgz.nrrd"))
@assert colorspace(img) == "Gray"
@assert ndims(img) == 3
@assert colordim(img) == 0
@assert eltype(img) == Float32
outname = joinpath(writedir, "smallgz.nrrd")
imwrite(img, outname)
sleep(0.2)
imgc = imread(outname)
@assert img.data == imgc.data
