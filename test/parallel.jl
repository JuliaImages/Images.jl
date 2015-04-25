addprocs(2)
@everywhere require("Images.jl")

# Issue #287
using Images
import Images.RGB
@everywhere function test287(img)
    return 0;
end
Imgs = Array(Image{RGB{Float64}}, 2);
Imgs[1] = convert(Image{RGB}, rand(100, 100, 3));
Imgs[2] = convert(Image{RGB}, rand(100, 100, 3));

let Imgs = Imgs;
    ret = pmap(i -> test287(Imgs[i]), 1:2);
    @test ret == Any[0,0]
end
