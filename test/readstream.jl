using Images, TestImages
using Base.Test

workdir = joinpath(tempdir(), "ImagesStreamTest")
testfile = joinpath(workdir,"jigsaw_tmpl.png")
if !isfile(testfile)
  mkpath(workdir)
  download("http://www.imagemagick.org/Usage/images/jigsaw_tmpl.png",testfile)
end

@osx_only open(testfile,"r") do io
  img = imread(io,Images.OSXNative)
end

## cleanup
rm(workdir,recursive=true)
