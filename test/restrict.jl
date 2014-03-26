# This is for testing display in IJulia. A huge image should
# be automatically restricted before sending to IJulia for display.

using Images

url = "http://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg"

const savedir = joinpath(tempdir(), "Images")
if !isdir(savedir)
    mkdir(savedir)
end

const savename = joinpath(savedir, "The_Earth_seen_from_Apollo_17.jpg")
if !isfile(savename)
    download(url, savename)
end

img = imread(savename)
