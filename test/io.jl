import Images
using Color, FixedPointNumbers, Base.Test

if !isdefined(:workdir)
    const workdir = joinpath(tempdir(), "Images")
end

if !isdir(workdir)
    mkdir(workdir)
end

a = rand(2,2)
aa = convert(Array{Ufixed8}, a)
fn = joinpath(workdir, "2by2.png")
Images.imwrite(a, fn)
b = Images.imread(fn)
@test convert(Array, b) == aa
Images.imwrite(aa, fn)
b = Images.imread(fn)
@test convert(Array, b) == aa
aaimg = Images.grayim(aa)
open(fn, "w") do file
    writemime(file, MIME("image/png"), aaimg, minpixels=0)
end
b = Images.imread(fn)
@test b == aaimg
aa = convert(Array{Ufixed16}, a)
Images.imwrite(aa, fn)
b = Images.imread(fn)
@test convert(Array, b) == aa
aa = Ufixed12[0.6 0.2;
              1.4 0.8]
open(fn, "w") do file
    writemime(file, MIME("image/png"), Images.grayim(aa), minpixels=0)
end
b = Images.imread(fn)
@test Images.data(b) == Ufixed8[0.6 0.2;
                                1.0 0.8]

img = Images.colorim(rand(3,2,2))
img24 = convert(Images.Image{RGB24}, img)
Images.imwrite(img24, fn)
b = Images.imread(fn)
imgrgb8 = convert(Images.Image{RGB{Ufixed8}}, img)
@test Images.data(imgrgb8) == Images.data(b)

# test writemime's use of restrict
abig = Images.grayim(rand(Uint8, 1024, 1023))
fn = joinpath(workdir, "big.png")
open(fn, "w") do file
    writemime(file, MIME("image/png"), abig, maxpixels=10^6)
end
b = Images.imread(fn)
@test Images.data(b) == convert(Array{Ufixed8,2}, Images.data(Images.restrict(abig, (1,2))))

# More writemime tests
a = Images.colorim(rand(Uint8, 3, 2, 2))
fn = joinpath(workdir, "2by2.png")
open(fn, "w") do file
    writemime(file, MIME("image/png"), a, minpixels=0)
end
b = Images.imread(fn)
@test Images.data(b) == Images.data(a)

abig = Images.colorim(rand(Uint8, 3, 1021, 1026))
fn = joinpath(workdir, "big.png")
open(fn, "w") do file
    writemime(file, MIME("image/png"), abig, maxpixels=10^6)
end
b = Images.imread(fn)
@test Images.data(b) == convert(Array{RGB{Ufixed8},2}, Images.data(Images.restrict(abig, (1,2))))

# Issue #269
abig = Images.colorim(rand(Uint16, 3, 1024, 1023))
open(fn, "w") do file
    writemime(file, MIME("image/png"), abig, maxpixels=10^6)
end
b = Images.imread(fn)
@test Images.data(b) == convert(Array{RGB{Ufixed8},2}, Images.data(Images.restrict(abig, (1,2))))

using Color
datafloat = reshape(linspace(0.5, 1.5, 6), 2, 3)
dataint = round(Uint8, 254*(datafloat .- 0.5) .+ 1)  # ranges from 1 to 255
# build our colormap
b = RGB(0,0,1)
w = RGB(1,1,1)
r = RGB(1,0,0)
cmaprgb = Array(RGB{Float64}, 255)
f = linspace(0,1,128)
cmaprgb[1:128] = [(1-x)*b + x*w for x in f]
cmaprgb[129:end] = [(1-x)*w + x*r for x in f[2:end]]
img = Images.ImageCmap(dataint, cmaprgb)
Images.imwrite(img,joinpath(workdir,"cmap.jpg"))

c = reinterpret(Images.BGRA{Ufixed8}, [0xf0884422]'')
fn = joinpath(workdir, "alpha.png")
Images.imwrite(c, fn)
C = Images.imread(fn)
# @test C[1] == c[1]  # disabled because Travis has a weird, old copy of ImageMagick for which this fails (see #261)
Images.imwrite(reinterpret(ARGB32, [0xf0884422]''), fn)
D = Images.imread(fn)
# @test D[1] == c[1]
