using Images

all_close(ar::AbstractImage, v) = all(abs(data(ar)-v) .< sqrt(eps(v)))
all_close(ar, v) = all(abs(ar-v) .< sqrt(eps(v)))

# arithmetic
img = convert(Image, zeros(3,3))
img2 = (img + 3)/2
@assert all(img2 .== 1.5)
img3 = 2img2
@assert all(img3 .== 3)
img3 = copy(img2)
img3[img2 .< 4] = -1
@assert all(img3 .== -1)

# scaling, ssd
img = convert(Image, fill(typemax(Uint16), 3, 3))
scalei = scaleinfo(Uint8, img)
img8 = scale(scalei, img)
@assert all(img8 .== typemax(Uint8))
A = randn(3,3)
mxA = max(A)
offset = 30.0
img = convert(Image, A + offset)
scalei = Images.ScaleMinMax{Uint8, Float64}(offset, offset+mxA, 100/mxA)
imgs = scale(scalei, img)
@assert min(imgs) == 0
@assert max(imgs) == 100
@assert eltype(imgs) == Uint8
imgs = imadjustintensity(img, [])
mnA = min(A)
@assert ssd(imgs, (A-mnA)/(mxA-mnA)) < eps()


# filtering
@assert all_close(imfilter(ones(4,4), ones(3,3)), 9.0)
@assert all_close(imfilter(ones(3,3), ones(3,3)), 9.0)
@assert all_close(imfilter(ones(3,3), [1 1 1;1 0.0 1;1 1 1]), 8.0)

img = convert(Image, ones(4,4))
@assert all_close(imfilter(img, ones(3,3)), 9.0)
