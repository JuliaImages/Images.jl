import Images

const writedir = joinpath(tempdir(), "Images")

if !isdir(writedir)
    mkdir(writedir)
end

a = rand(2,2)
fn = joinpath(writedir, "2by2.png")
Images.imwrite(a, fn)
b = Images.imread(fn)
@assert convert(Array, b) == scale(Images.scaleinfo(a), a)
aa = int8(127*a)
Images.imwrite(aa, fn)
b = Images.imread(fn)
@assert convert(Array{Int8}, b) == aa
aa = uint8(255*a)
Images.imwrite(aa, fn)
b = Images.imread(fn)
@assert convert(Array, b) == aa
aa = uint16(65535*a)
Images.imwrite(aa, fn)
b = Images.imread(fn)
@assert convert(Array, b) == aa

using Color
datafloat = reshape(linspace(0.5, 1.5, 6), 2, 3)
dataint = iround(Uint8, 254*(datafloat .- 0.5) .+ 1)  # ranges from 1 to 255
# build our colormap
b = RGB(0,0,1)
w = RGB(1,1,1)
r = RGB(1,0,0)
cmaprgb = Array(RGB, 255)
f = linspace(0,1,128)
cmaprgb[1:128] = [(1-x)*b + x*w for x in f]
cmaprgb[129:end] = [(1-x)*w + x*r for x in f[2:end]]
img = Images.ImageCmap(dataint, cmaprgb)
Images.imwrite(img,joinpath(writedir,"cmap.jpg"))
