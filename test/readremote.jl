using Images, FixedPointNumbers, Color
using Base.Test

urlbase = "http://www.imagemagick.org/Usage/images/"

const savedir = joinpath(tempdir(), "Images")
const writedir = joinpath(savedir, "write")

function getfile(name)
    file = joinpath(savedir, name)
    if !isfile(file)
        file = download(urlbase*name, file)
    end
    file
end

if !isdir(savedir)
    mkdir(savedir)
end
if !isdir(writedir)
    mkdir(writedir)
end

# Gray
file = getfile("jigsaw_tmpl.png")
img = imread(file)
@test colorspace(img) == "Gray"
@test ndims(img) == 2
@test colordim(img) == 0
@test eltype(img) == Ufixed8
outname = joinpath(writedir, "jigsaw_tmpl.png")
imwrite(img, outname)
imgc = imread(outname)
@test img.data == imgc.data


# Gray with alpha channel
file = getfile("wmark_image.png")
img = imread(file)
@test colorspace(img) == "GrayAlpha"
@test ndims(img) == 2
@test colordim(img) == 0
@test eltype(img) == Images.ColorTypes.GrayAlpha{Ufixed8}
outname = joinpath(writedir, "wmark_image.png")
imwrite(img, outname)
sleep(0.2)
imgc = imread(outname)
# @test img.data == imgc.data   # libmagick bug: doesn't write GrayAlpha properly?

# RGB
file = getfile("rose.png")
img = imread(file)
@show colorspace(img)
@test colorspace(img) == "RGB"
@test ndims(img) == 2
@test colordim(img) == 0
@test eltype(img) == RGB{Ufixed8}
outname = joinpath(writedir, "rose.ppm")
imwrite(img, outname)
imgc = imread(outname)
T = eltype(imgc)
lim = limits(imgc)
@test (typeof(lim[1]) == typeof(lim[2]) == T)  # issue #62
@test img.data == imgc.data

# RGBA with 16 bit depth
file = getfile("autumn_leaves.png")
img = imread(file)
@test colorspace(img) == "BGRA"
@test ndims(img) == 2
@test colordim(img) == 0
@test eltype(img) == Images.ColorTypes.BGRA{Ufixed16}
outname = joinpath(writedir, "autumn_leaves.png")
imwrite(img, outname)
sleep(0.2)
imgc = imread(outname)
@test img.data == imgc.data

# Indexed
file = getfile("present.gif")
img = imread(file)
