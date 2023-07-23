# When we deprecate old APIs, we copy the old tests here to make sure they still work

@testset "deprecations" begin
        # deprecated
    @suppress_err @testset "Thresholding" begin

        #otsu_threshold
        img = testimage("cameraman")
        thres = otsu_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 87/256), atol=eps(N0f8))
        thres = otsu_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 87/256), atol=eps(N0f8))

        img = map(x->convert(Gray{Float64}, x), img)
        thres = otsu_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)
        thres = otsu_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)

        img = map(x->convert(Float64, x), img)
        thres = otsu_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)
        thres = otsu_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 87/256, atol=0.01)

        #test for multidimension arrays
        img = rand(Float64, 10, 10, 3)
        @test otsu_threshold(img) == otsu_threshold(cat(img[:,:,1], img[:,:,2], img[:,:,3], dims=1))

        #yen_threshold
        img = testimage("cameraman")
        thres = yen_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 199/256), atol=eps(N0f8))
        thres = yen_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), convert(N0f8, 199/256), atol=eps(N0f8))

        img = map(x->convert(Gray{Float64}, x), img)
        thres = yen_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)
        thres = yen_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)

        img = map(x->convert(Float64, x), img)
        thres = yen_threshold(img)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)
        thres = yen_threshold(img, 512)
        @test typeof(thres) == eltype(img)
        @test ≈(gray(thres), 199/256, atol=0.01)

        img = rand(Float64, 10, 10, 3)
        @test yen_threshold(img) == yen_threshold(cat(img[:,:,1], img[:,:,2], img[:,:,3], dims=1))

        img = zeros(Gray{Float64},10,10,3)
        @test yen_threshold(img) == 0
    end
end
