# This is for testing display in IJulia. A huge image should
# be automatically restricted before sending to IJulia for display.

using Images

url = "http://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg"

if !isdefined(:workdir)
    const workdir = joinpath(tempdir(), "Images")
end
if !isdefined(:writedir)
    const writedir = joinpath(workdir, "write")
end

if !isdir(workdir)
    mkdir(workdir)
end
if !isdir(writedir)
    mkdir(writedir)
end

const savename = joinpath(writedir, "The_Earth_seen_from_Apollo_17.jpg")
if !isfile(savename)
    download(url, savename)
end

img = imread(savename)
