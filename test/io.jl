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
