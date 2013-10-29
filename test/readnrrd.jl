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
img = imread("../test/small.nrrd")
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
img = imread("../test/smallgz.nrrd")
@assert colorspace(img) == "Gray"
@assert ndims(img) == 3
@assert colordim(img) == 0
@assert eltype(img) == Float32
outname = joinpath(writedir, "smallgz.nrrd")
imwrite(img, outname)
sleep(0.2)
imgc = imread(outname)
@assert img.data == imgc.data


# RGB
#file = getfile("rose.png")
#img = imread(file)
#@assert colorspace(img) == "RGB"
#@assert ndims(img) == 3
#@assert colordim(img) == 1
#@assert size(img, 1) == 3
#@assert eltype(img) == Uint8
#outname = joinpath(writedir, "rose.ppm")
###imwrite(img, outname)
#imgc = imread(outname)
#@assert img.data == imgc.data

# RGBA with 16 bit depth
#file = getfile("autumn_leaves.png")
####img = imread(file)
#@assert colorspace(img) == "RGBA"
#@assert ndims(img) == 3
#@assert colordim(img) == 1
#@assert size(img, 1) == 4
#@assert eltype(img) == Uint16
# outname = joinpath(writedir, "autumn_leaves.png")
# imwrite(img, outname)
# imgc = imread(outname)
# @assert img.data == imgc.data

# Indexed
#file = getfile("present.gif")
#img = imread(file)
