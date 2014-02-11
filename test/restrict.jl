using Images

url = "https://atmire.com/dspace-labs3/bitstream/handle/123456789/7618/earth-map-huge.jpg?sequence=1"

const savedir = joinpath(tempdir(), "Images")
if !isdir(savedir)
    mkdir(savedir)
end

const savename = joinpath(savedir, "earth-map-huge.jpg")
if !isfile(savename)
    download(url, savename)
end

img = imread(savename)
