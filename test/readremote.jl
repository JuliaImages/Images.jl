urlbase = "http://www.imagemagick.org/Usage/images/"

const savedir = joinpath(tempdir(), "Images")
if !isdir(savedir)
    mkdir(savedir)
end
function getfile(name)
    file = joinpath(savedir, name)
    if !isfile(file)
        @show file
        @show urlbase*name
        file = download(urlbase*name, file)
    end
    file
end

# RGB
file = getfile("rose.png")
img = Images.imread(file)
@assert Images.colorspace(img) == "RGB"
@assert ndims(img) == 3
@assert Images.colordim(img) == 1
@assert size(img, 1) == 3
@assert eltype(img) == Uint8

# RGBA with 16 bit depth
file = getfile("autumn_leaves.png")
img = Images.imread(file)
@assert Images.colorspace(img) == "RGBA"
@assert ndims(img) == 3
@assert Images.colordim(img) == 1
@assert size(img, 1) == 4
@assert eltype(img) == Uint16

