import Images
using Color, Base.Test

# Comparison of each element in arrays with a scalar
approx_equal(ar, v) = all(abs(ar.-v) .< sqrt(eps(v)))
approx_equal(ar::Images.AbstractImage, v) = approx_equal(Images.data(ar), v)

# arithmetic
img = convert(Images.Image, zeros(3,3))
@assert Images.limits(img) == (0,1)
img2 = (img .+ 3)/2
@assert all(img2 .== 1.5)
@assert Images.limits(img2) == (1.5,2.0)
img3 = 2img2
@assert all(img3 .== 3)
img3 = copy(img2)
img3[img2 .< 4] = -1
@assert all(img3 .== -1)
img = convert(Images.Image, rand(3,4))
A = rand(3,4)
img2 = img .* A
@assert all(Images.data(img2) == Images.data(img).*A)
@assert Images.limits(img2) == (0,1)
img2 = convert(Images.Image, A)
img2 = img2 .- 0.5
img3 = 2img .* img2
@assert Images.limits(img3) == (-1, 1)
img2 = img ./ A
@assert Images.limits(img2) == (0, Inf)
img2 = (2img).^2
@assert Images.limits(img2) == (0, 4)
imgu = convert(Images.Image, Uint8[1 240; 10 128])  # from #101
@assert limits(2imgu) == (0x00,0xff)

# scaling, ssd
img = convert(Images.Image, fill(typemax(Uint16), 3, 3))
scalei = Images.scaleinfo(Uint8, img)
img8 = scale(scalei, img)
@assert all(img8 .== typemax(Uint8))
A = randn(3,3)
mxA = maximum(A)
offset = 30.0
img = convert(Images.Image, A .+ offset)
scalei = Images.ScaleMinMax{Uint8, Float64}(offset, offset+mxA, 100/mxA)
imgs = scale(scalei, img)
@assert minimum(imgs) == 0
@assert maximum(imgs) == 100
@assert eltype(imgs) == Uint8
imgs = Images.imadjustintensity(img, [])
mnA = minimum(A)
@assert Images.ssd(imgs, (A.-mnA)/(mxA-mnA)) < eps()
A = reshape(1:9, 3, 3)
B = scale(Images.ClipMin(Float32, 3), A)
@assert eltype(B) == Float32 && B == [3 4 7; 3 5 8; 3 6 9]
B = scale(Images.ClipMax(Uint8, 7), A)
@assert eltype(B) == Uint8 && B == [1 4 7; 2 5 7; 3 6 7]

# filtering
EPS = 1e-14
for T in (Float64, Int)
    A = zeros(T,3,3); A[2,2] = 1
    kern = rand(3,3)
    @test maximum(abs(Images.imfilter(A, kern) - rot180(kern))) < EPS
    kern = rand(2,3)
    @test maximum(abs(Images.imfilter(A, kern)[1:2,:] - rot180(kern))) < EPS
    kern = rand(3,2)
    @test maximum(abs(Images.imfilter(A, kern)[:,1:2] - rot180(kern))) < EPS
end
for T in (Float64, Int)
    # Separable kernels
    A = zeros(T,3,3); A[2,2] = 1
    kern = rand(3).*rand(3)'
    @test maximum(abs(Images.imfilter(A, kern) - rot180(kern))) < EPS
    kern = rand(2).*rand(3)'
    @test maximum(abs(Images.imfilter(A, kern)[1:2,:] - rot180(kern))) < EPS
    kern = rand(3).*rand(2)'
    @test maximum(abs(Images.imfilter(A, kern)[:,1:2] - rot180(kern))) < EPS
end
A = zeros(3,3); A[2,2] = 1
kern = rand(3,3)
@test maximum(abs(Images.imfilter_fft(A, kern) - rot180(kern))) < EPS
kern = rand(2,3)
@test maximum(abs(Images.imfilter_fft(A, kern)[1:2,:] - rot180(kern))) < EPS
kern = rand(3,2)
@test maximum(abs(Images.imfilter_fft(A, kern)[:,1:2] - rot180(kern))) < EPS

@assert approx_equal(Images.imfilter(ones(4,4), ones(3,3)), 9.0)
@assert approx_equal(Images.imfilter(ones(3,3), ones(3,3)), 9.0)
@assert approx_equal(Images.imfilter(ones(3,3), [1 1 1;1 0.0 1;1 1 1]), 8.0)
img = convert(Images.Image, ones(4,4))
@assert approx_equal(Images.imfilter(img, ones(3,3)), 9.0)
A = zeros(5,5,3); A[3,3,[1,3]] = 1
@assert Images.colordim(A) == 3
kern = rand(3,3)
kernpad = zeros(5,5); kernpad[2:4,2:4] = kern
Af = Images.imfilter(A, kern)

@test_approx_eq Af cat(3, rot180(kernpad), zeros(5,5), rot180(kernpad))
Aimg = permutedims(convert(Images.Image, A), [3,1,2])
@test_approx_eq Images.imfilter(Aimg, kern) permutedims(Af, [3,1,2])
@assert approx_equal(Images.imfilter(ones(4,4),ones(1,3),"replicate"), 3.0)

@assert approx_equal(Images.imfilter_gaussian(ones(4,4), [5,5]), 1.0)

# restriction
A = reshape(uint16(1:60), 4, 5, 3)
B = Images.restrict(A, (1,2))
@test_approx_eq B cat(3, [ 0.96875  4.625   5.96875;
                           2.875   10.5    12.875;
                           1.90625  5.875   6.90625],
                         [ 8.46875  14.625 13.46875;
                          17.875    30.5   27.875;
                           9.40625  15.875 14.40625],
                         [15.96875  24.625 20.96875;
                          32.875    50.5   42.875;
                          16.90625  25.875 21.90625])
A = reshape(1:60, 5, 4, 3)
B = Images.restrict(A, (1,2,3))
@test_approx_eq B cat(3, [ 2.6015625  8.71875 6.1171875;
                           4.09375   12.875   8.78125;
                           3.5390625 10.59375 7.0546875],
                         [10.1015625 23.71875 13.6171875;
                          14.09375   32.875   18.78125;
                          11.0390625 25.59375 14.5546875])

# color conversion
gray = linspace(0.0,1.0,5) # a 1-dimensional image
gray8 = iround(Uint8, 255*gray)
gray32 = [uint32(g)<<16 | uint32(g)<<8 | uint32(g) for g in gray8]
imgray = Images.Image(gray, ["colordim"=>0, "colorspace"=>"Gray"])
buf = Images.uint32color(imgray)
@assert buf == gray32
rgb = [RGB(g, g, g) for g in gray]
buf = Images.uint32color(rgb)
@assert buf == gray32
img = Images.Image(gray32, ["colordim"=>0, "colorspace"=>"RGB24"])
buf = Images.uint32color(img)
@assert buf == gray32
rgb = repeat(gray, outer=[1,3])
img = Images.Image(rgb, ["colordim"=>2, "colorspace"=>"RGB"])
buf = Images.uint32color(img)
@assert buf == gray32
rgb = repeat(gray', outer=[3,1])
img = Images.Image(rgb, ["colordim"=>1, "colorspace"=>"RGB"])
buf = Images.uint32color(img)
@assert buf == gray32
ovr = Images.Overlay((gray, 0*gray), (RGB(1,0,1), RGB(0,1,0)), ((0,1),(0,1)))
buf = Images.uint32color(ovr)
nogreen = [uint32(g)<<16 | uint32(g) for g in gray8]
@assert buf == nogreen
ovr = Images.Overlay((gray, gray), (RGB(1,0,1), RGB(0,1,0)), ((0,1),(0,1)))
ovr.visible[2] = false
buf = Images.uint32color(ovr)
@assert buf == nogreen

# erode/dilate
A = zeros(4,4,3)
A[2,2,1] = 0.8
A[4,4,2] = 0.6
Ae = Images.erode(A)
@assert Ae == zeros(size(A))
Ad = Images.dilate(A)
Ar = [0.8 0.8 0.8 0;
      0.8 0.8 0.8 0;
      0.8 0.8 0.8 0;
      0 0 0 0]
Ag = [0 0 0 0;
      0 0 0 0;
      0 0 0.6 0.6;
      0 0 0.6 0.6]
@assert Ad == cat(3, Ar, Ag, zeros(4,4))
Ae = Images.erode(Ad)
Ar = [0.8 0.8 0 0;
      0.8 0.8 0 0;
      0 0 0 0;
      0 0 0 0]
Ag = [0 0 0 0;
      0 0 0 0;
      0 0 0 0;
      0 0 0 0.6]
@assert Ae == cat(3, Ar, Ag, zeros(4,4))

# opening/closing
A = zeros(4,4,3)
A[2,2,1] = 0.8
A[4,4,2] = 0.6
Ao = Images.opening(A)
@assert Ao == zeros(size(A))
A = zeros(10,10)
A[4:7,4:7] = 1
B = copy(A)
A[5,5] = 0
Ac = Images.closing(A)
@assert Ac == B

# label_components
A = [true  true  false true;
     true  false true  true]
lbltarget = [1 1 0 2;
             1 0 2 2]
lbltarget1 = [1 2 0 4;
              1 0 3 4]
@assert Images.label_components(A) == lbltarget
@assert Images.label_components(A, [1]) == lbltarget1
connectivity = [false true  false;
                true  false true;
                false true  false]
@assert Images.label_components(A, connectivity) == lbltarget
connectivity = trues(3,3)
lbltarget2 = [1 1 0 1;
              1 0 1 1]
@assert Images.label_components(A, connectivity) == lbltarget2

# phantoms

P = [ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0;
      0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
      0.0  0.0  0.2  0.3  0.3  0.2  0.0  0.0;
      0.0  0.0  0.2  0.0  0.2  0.2  0.0  0.0;
      0.0  0.0  0.2  0.0  0.0  0.2  0.0  0.0;
      0.0  0.0  0.2  0.2  0.2  0.2  0.0  0.0;
      0.0  0.0  1.0  0.2  0.2  1.0  0.0  0.0;
      0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ]

Q = Images.shepp_logan(8)
@assert norm((P-Q)[:]) < 1e-10

P = [ 0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0;
      0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
      0.0  0.0  1.02  1.03  1.03  1.02  0.0  0.0;
      0.0  0.0  1.02  1.0   1.02  1.02  0.0  0.0;
      0.0  0.0  1.02  1.0   1.0   1.02  0.0  0.0;
      0.0  0.0  1.02  1.02  1.02  1.02  0.0  0.0;
      0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
      0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0 ]

Q = Images.shepp_logan(8,highContrast=false)
@assert norm((P-Q)[:]) < 1e-10

