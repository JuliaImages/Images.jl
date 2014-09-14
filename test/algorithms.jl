import Images
using Color, Base.Test, FixedPointNumbers

# Comparison of each element in arrays with a scalar
approx_equal(ar, v) = all(abs(ar.-v) .< sqrt(eps(v)))
approx_equal(ar::Images.AbstractImage, v) = approx_equal(Images.data(ar), v)

# arithmetic
img = convert(Images.Image, zeros(3,3))
img2 = (img .+ 3)/2
@test all(img2 .== 1.5)
img3 = 2img2
@test all(img3 .== 3)
img3 = copy(img2)
img3[img2 .< 4] = -1
@test all(img3 .== -1)
img = convert(Images.Image, rand(3,4))
A = rand(3,4)
img2 = img .* A
@test all(Images.data(img2) == Images.data(img).*A)
img2 = convert(Images.Image, A)
img2 = img2 .- 0.5
img3 = 2img .* img2
img2 = img ./ A
img2 = (2img).^2
# Same operations with ColorValue images
img = Images.colorim(zeros(Float32,3,4,5))
img2 = (img .+ RGB{Float32}(1,1,1))/2
@test all(img2 .== RGB{Float32}(1,1,1)/2)
img3 = 2img2
@test all(img3 .== RGB{Float32}(1,1,1))
A = fill(2, 4, 5)
@test all(A.*img2 .== fill(RGB{Float32}(1,1,1), 4, 5))
img2 = img2 .- RGB{Float32}(1,1,1)/2

# Reductions
let
    A = rand(5,5,3)
    img = Images.colorim(A, "RGB")
    s12 = sum(img, (1,2))
    @test Images.colorspace(s12) == "RGB"
    s3 = sum(img, (3,))
    @test Images.colorspace(s3) == "Unknown"
    A = [NaN, 1, 2, 3]
    @test_approx_eq Images.meanfinite(A, 1) [2]
    A = [NaN 1 2 3;
         NaN 6 5 4]
    @test_approx_eq Images.meanfinite(A, 1) [NaN 3.5 3.5 3.5]
    @test_approx_eq Images.meanfinite(A, 2) [2, 5]'
    @test_approx_eq Images.meanfinite(A, (1,2)) [3.5]
    @test Images.minfinite(A) == 1
    @test Images.maxfinite(A) == 6
    @test Images.maxabsfinite(A) == 6
    A = float32(rand(3,5,5))
    img = Images.colorim(A, "RGB")
    dc = Images.data(Images.meanfinite(img, 1))-reinterpret(RGB{Float32}, mean(A, 2), (1,5))
    @test maximum(map(abs, dc)) < 1e-6
end


# Array padding
let A = [1 2; 3 4]
    @test Images.padindexes(A, 1, 0, 0, "replicate") == [1,2]
    @test Images.padindexes(A, 1, 1, 0, "replicate") == [1,1,2]
    @test Images.padindexes(A, 1, 0, 1, "replicate") == [1,2,2]
    @test Images.padarray(A, (0,0), (0,0), "replicate") == A
    @test Images.padarray(A, (1,2), (2,0), "replicate") == [1 1 1 2; 1 1 1 2; 3 3 3 4; 3 3 3 4; 3 3 3 4]
    @test Images.padindexes(A, 1, 1, 0, "circular") == [2,1,2]
    @test Images.padindexes(A, 1, 0, 1, "circular") == [1,2,1]
    @test Images.padarray(A, [2,1], [0,2], "circular") == [2 1 2 1 2; 4 3 4 3 4; 2 1 2 1 2; 4 3 4 3 4]
    @test Images.padindexes(A, 1, 1, 0, "symmetric") == [1,1,2]
    @test Images.padindexes(A, 1, 0, 1, "symmetric") == [1,2,2]
    @test Images.padarray(A, (1,2), (2,0), "symmetric") == [2 1 1 2; 2 1 1 2; 4 3 3 4; 4 3 3 4; 2 1 1 2]
    @test Images.padarray(A, (1,2), (2,0), "value", -1) == [-1 -1 -1 -1; -1 -1 1 2; -1 -1 3 4; -1 -1 -1 -1; -1 -1 -1 -1]
    A = [1 2 3; 4 5 6]
    @test Images.padindexes(A, 2, 1, 0, "reflect") == [2,1,2,3]
    @test Images.padindexes(A, 2, 0, 1, "reflect") == [1,2,3,2]
    @test Images.padarray(A, (1,2), (2,0), "reflect") == [6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6; 3 2 1 2 3; 6 5 4 5 6]
    A = [1 2; 3 4]
    @test Images.padarray(A, (1,1)) == [1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4]
    @test Images.padarray(A, (1,1), "replicate", "both") == [1 1 2 2; 1 1 2 2; 3 3 4 4; 3 3 4 4]
    @test Images.padarray(A, (1,1), "circular", "pre") == [4 3 4; 2 1 2; 4 3 4]
    @test Images.padarray(A, (1,1), "symmetric", "post") == [1 2 2; 3 4 4; 3 4 4]
    A = ["a" "b"; "c" "d"]
    @test Images.padarray(A, (1,1)) == ["a" "a" "b" "b"; "a" "a" "b" "b"; "c" "c" "d" "d"; "c" "c" "d" "d"]
    @test_throws ErrorException Images.padindexes(A, 1, 1, 1, "unknown")
    @test_throws ErrorException Images.padarray(A, (1,1), "unknown")
end

# filtering
EPS = 1e-14
imgcol = Images.colorim(rand(3,5,6))
imgcolf = convert(Image{RGB{Ufixed8}}, imgcol)
for T in (Float64, Int)
    A = zeros(T,3,3); A[2,2] = 1
    kern = rand(3,3)
    @test maximum(abs(Images.imfilter(A, kern) - rot180(kern))) < EPS
    kern = rand(2,3)
    @test maximum(abs(Images.imfilter(A, kern)[1:2,:] - rot180(kern))) < EPS
    kern = rand(3,2)
    @test maximum(abs(Images.imfilter(A, kern)[:,1:2] - rot180(kern))) < EPS
end
kern = zeros(3,3); kern[2,2] = 1
@test maximum(map(abs, imgcol - Images.imfilter(imgcol, kern))) < EPS
@test maximum(map(abs, imgcolf - Images.imfilter(imgcolf, kern))) < EPS
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
kern = zeros(3,3); kern[2,2] = 1
@test maximum(map(abs, imgcol - Images.imfilter_fft(imgcol, kern))) < EPS
@test maximum(map(abs, imgcolf - Images.imfilter_fft(imgcolf, kern))) < EPS

@test approx_equal(Images.imfilter(ones(4,4), ones(3,3)), 9.0)
@test approx_equal(Images.imfilter(ones(3,3), ones(3,3)), 9.0)
@test approx_equal(Images.imfilter(ones(3,3), [1 1 1;1 0.0 1;1 1 1]), 8.0)
img = convert(Images.Image, ones(4,4))
@test approx_equal(Images.imfilter(img, ones(3,3)), 9.0)
A = zeros(5,5,3); A[3,3,[1,3]] = 1
@test Images.colordim(A) == 3
kern = rand(3,3)
kernpad = zeros(5,5); kernpad[2:4,2:4] = kern
Af = Images.imfilter(A, kern)

@test_approx_eq Af cat(3, rot180(kernpad), zeros(5,5), rot180(kernpad))
Aimg = permutedims(convert(Images.Image, A), [3,1,2])
@test_approx_eq Images.imfilter(Aimg, kern) permutedims(Af, [3,1,2])
@test approx_equal(Images.imfilter(ones(4,4),ones(1,3),"replicate"), 3.0)

A = zeros(5,5); A[3,3] = 1
kern = rand(3,3)
Af = Images.imfilter(A, kern, "inner")
@test Af == rot180(kern)
Afft = Images.imfilter_fft(A, kern, "inner")
@test_approx_eq Af Afft

@test approx_equal(Images.imfilter_gaussian(ones(4,4), [5,5]), 1.0)
A = fill(nan(Float32), 3, 3)
A[1,1] = 1
A[2,1] = 2
A[3,1] = 3
@test Images.imfilter_gaussian(A, [0,0]) === A
@test_approx_eq Images.imfilter_gaussian(A, [0,3]) A
B = copy(A)
B[isfinite(B)] = 2
@test_approx_eq Images.imfilter_gaussian(A, [10^3,0]) B
@test maximum(map(abs, Images.imfilter_gaussian(imgcol, [10^3,10^3]) - mean(imgcol))) < 1e-4
@test maximum(map(abs, Images.imfilter_gaussian(imgcolf, [10^3,10^3]) - mean(imgcolf))) < 1e-4

A = zeros(Int, 9, 9); A[5, 5] = 1
@test maximum(abs(Images.imfilter_LoG(A, [1,1]) - Images.imlog(1.0))) < EPS

@test Images.imaverage() == fill(1/9, 3, 3)
@test Images.imaverage([3,3]) == fill(1/9, 3, 3)
@test_throws ErrorException Images.imaverage([5])

# restriction
A = reshape(uint16(1:60), 4, 5, 3)
B = Images.restrict(A, (1,2))
Btarget = cat(3, [ 0.96875  4.625   5.96875;
                   2.875   10.5    12.875;
                   1.90625  5.875   6.90625],
                 [ 8.46875  14.625 13.46875;
                  17.875    30.5   27.875;
                   9.40625  15.875 14.40625],
                 [15.96875  24.625 20.96875;
                  32.875    50.5   42.875;
                  16.90625  25.875 21.90625])
@test_approx_eq B Btarget
Argb = reinterpret(RGB, reinterpret(Ufixed16, permutedims(A, (3,1,2))))
B = Images.restrict(Argb)
Bf = permutedims(reinterpret(Float64, B), (2,3,1))
@test_approx_eq_eps Bf Btarget/reinterpret(one(Ufixed16)) 1e-12
Argba = reinterpret(RGBA{Ufixed16}, reinterpret(Ufixed16, A))
B = Images.restrict(Argba)
@test_approx_eq_eps reinterpret(Float64, B) restrict(A, (2,3))/reinterpret(one(Ufixed16)) 1e-12
A = reshape(1:60, 5, 4, 3)
B = Images.restrict(A, (1,2,3))
@test_approx_eq B cat(3, [ 2.6015625  8.71875 6.1171875;
                           4.09375   12.875   8.78125;
                           3.5390625 10.59375 7.0546875],
                         [10.1015625 23.71875 13.6171875;
                          14.09375   32.875   18.78125;
                          11.0390625 25.59375 14.5546875])
Images.restrict(imgcol, (1,2))

# erode/dilate
A = zeros(4,4,3)
A[2,2,1] = 0.8
A[4,4,2] = 0.6
Ae = Images.erode(A)
@test Ae == zeros(size(A))
Ad = Images.dilate(A)
Ar = [0.8 0.8 0.8 0;
      0.8 0.8 0.8 0;
      0.8 0.8 0.8 0;
      0 0 0 0]
Ag = [0 0 0 0;
      0 0 0 0;
      0 0 0.6 0.6;
      0 0 0.6 0.6]
@test Ad == cat(3, Ar, Ag, zeros(4,4))
Ae = Images.erode(Ad)
Ar = [0.8 0.8 0 0;
      0.8 0.8 0 0;
      0 0 0 0;
      0 0 0 0]
Ag = [0 0 0 0;
      0 0 0 0;
      0 0 0 0;
      0 0 0 0.6]
@test Ae == cat(3, Ar, Ag, zeros(4,4))

# opening/closing
A = zeros(4,4,3)
A[2,2,1] = 0.8
A[4,4,2] = 0.6
Ao = Images.opening(A)
@test Ao == zeros(size(A))
A = zeros(10,10)
A[4:7,4:7] = 1
B = copy(A)
A[5,5] = 0
Ac = Images.closing(A)
@test Ac == B

# label_components
A = [true  true  false true;
     true  false true  true]
lbltarget = [1 1 0 2;
             1 0 2 2]
lbltarget1 = [1 2 0 4;
              1 0 3 4]
@test Images.label_components(A) == lbltarget
@test Images.label_components(A, [1]) == lbltarget1
connectivity = [false true  false;
                true  false true;
                false true  false]
@test Images.label_components(A, connectivity) == lbltarget
connectivity = trues(3,3)
lbltarget2 = [1 1 0 1;
              1 0 1 1]
@test Images.label_components(A, connectivity) == lbltarget2

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
@test norm((P-Q)[:]) < 1e-10

P = [ 0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0;
      0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
      0.0  0.0  1.02  1.03  1.03  1.02  0.0  0.0;
      0.0  0.0  1.02  1.0   1.02  1.02  0.0  0.0;
      0.0  0.0  1.02  1.0   1.0   1.02  0.0  0.0;
      0.0  0.0  1.02  1.02  1.02  1.02  0.0  0.0;
      0.0  0.0  2.0   1.02  1.02  2.0   0.0  0.0;
      0.0  0.0  0.0   0.0   0.0   0.0   0.0  0.0 ]

Q = Images.shepp_logan(8,highContrast=false)
@test norm((P-Q)[:]) < 1e-10

