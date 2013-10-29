import Images

const writedir = joinpath(tempdir(), "Images")

if !isdir(writedir)
    mkdir(writedir)
end

a = rand(2,2)
fn = joinpath(writedir, "2by2.png")
Images.imwrite(a, fn)
b = Images.imread(fn)
@assert convert(Array, b) == scale(scaleinfo(a), a)
