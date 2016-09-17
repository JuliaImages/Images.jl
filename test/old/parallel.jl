addprocs(2)
using Images, Base.Test, Colors
# Normally "using Images" would suffice, but since Images has
# already been loaded by the time this file runs, that yields
# a no-op. So we have to force the workers to load it manually.
# See https://github.com/JuliaLang/julia/issues/3674
@sync for p in workers()
    @spawnat p eval(Expr(:using, :Images))   # FIXME: on 0.4 you can say :(using Images) directly
end

# Issue #287
@everywhere function test287(img)
    return 0;
end
Imgs = Array(Images.Image{RGB{Float64}}, 2);
Imgs[1] = convert(Images.Image{RGB}, rand(100, 100, 3));
Imgs[2] = convert(Images.Image{RGB}, rand(100, 100, 3));

let Imgs = Imgs;
    ret = pmap(i -> test287(Imgs[i]), 1:2);
    @test ret == Any[0,0]
end
