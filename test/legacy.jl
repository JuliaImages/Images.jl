# These tests once lived in other files, but related functions are declared legacy or moved to
# other packages. For regression test purpose, we still keep them here.

@testset "legacy" begin

    # Moved to ColorVectorSpace
    @testset "Complement" begin
        img = Gray{N0f16}.([0.01164 0.01118; 0.01036 0.01187])
        @test complement(Gray(0.5)) == Gray(0.5)
        @test complement(Gray(0.2)) == Gray(0.8)
        @test all(complement.(img) .== 1 .- img)

        @test complement.([Gray(0.2)]) == [Gray(0.8)]
        @test complement.([Gray{N0f8}(0.2)]) == [Gray{N0f8}(0.8)]
        @test complement.([RGB(0,0.3,1)]) == [RGB(1,0.7,0)]
        @test complement.([RGBA(0,0.3,1,0.7)]) == [RGBA(1.0,0.7,0.0,0.7)]
        @test complement.([RGBA{N0f8}(0,0.6,1,0.7)]) == [RGBA{N0f8}(1.0,0.4,0.0,0.7)]
    end

    # Moved to ImageBase
    @testset "Reductions" begin
        _abs(x::Colorant) = mapreducec(abs, +, 0, x)

        A = rand(5,5,3)
        img = colorview(RGB, PermutedDimsArray(A, (3,1,2)))
        s12 = sum(img, dims=(1,2))
        @test eltype(s12) <: RGB
        A = [NaN, 1, 2, 3]
        @test meanfinite(A, 1) ≈ [2]
        A = [NaN 1 2 3;
            NaN 6 5 4]
        mf = meanfinite(A, 1)
        @test isnan(mf[1])
        @test mf[1,2:end] ≈ [3.5,3.5,3.5]
        @test meanfinite(A, 2) ≈ reshape([2, 5], 2, 1)
        @test meanfinite(A, (1,2)) ≈ [3.5]
        @test minfinite(A) == 1
        @test maxfinite(A) == 6
        @test maxabsfinite(A) == 6
        A = rand(10:20, 5, 5)
        @test minfinite(A) == minimum(A)
        @test maxfinite(A) == maximum(A)
        A = reinterpret(N0f8, rand(0x00:0xff, 5, 5))
        @test minfinite(A) == minimum(A)
        @test maxfinite(A) == maximum(A)
        A = rand(Float32,3,5,5)
        img = colorview(RGB, A)
        dc = meanfinite(img, 1)-reshape(reinterpretc(RGB{Float32}, mean(A, dims=2)), (1,5))
        @test maximum(map(_abs, dc)) < 1e-6
        dc = minfinite(img)-RGB{Float32}(minimum(A, dims=(2,3))...)
        @test _abs(dc) < 1e-6
        dc = maxfinite(img)-RGB{Float32}(maximum(A, dims=(2,3))...)
        @test _abs(dc) < 1e-6
    end

    # Once moved to ImageTransformations, and then moved to ImageBase
    @testset "Restriction" begin
        imgcol = colorview(RGB, rand(3,5,6))
        A = reshape([convert(UInt16, i) for i = 1:60], 4, 5, 3)
        B = restrict(A, (1,2))
        Btarget = cat([ 0.96875  4.625   5.96875;
                           2.875   10.5    12.875;
                           1.90625  5.875   6.90625],
                      [ 8.46875  14.625 13.46875;
                        17.875    30.5   27.875;
                        9.40625  15.875 14.40625],
                      [15.96875  24.625 20.96875;
                       32.875    50.5   42.875;
                       16.90625  25.875 21.90625], dims=3)
        @test B ≈ Btarget
        Argb = reinterpretc(RGB, reinterpret(N0f16, permutedims(A, (3,1,2))))
        B = restrict(Argb)
        Bf = permutedims(reinterpretc(eltype(eltype(B)), B), (2,3,1))
        @test isapprox(Bf, Btarget/reinterpret(one(N0f16)), atol=1e-10)
        Argba = reinterpretc(RGBA{N0f16}, reinterpret(N0f16, A))
        B = restrict(Argba)
        @test isapprox(reinterpretc(eltype(eltype(B)), B), restrict(A, (2,3))/reinterpret(one(N0f16)), atol=1e-10)
        A = reshape(1:60, 5, 4, 3)
        B = restrict(A, (1,2,3))
        @test cat([ 2.6015625  8.71875 6.1171875;
                       4.09375   12.875   8.78125;
                       3.5390625 10.59375 7.0546875],
                     [10.1015625 23.71875 13.6171875;
                      14.09375   32.875   18.78125;
                      11.0390625 25.59375 14.5546875], dims=3) ≈ B
        imgcolax = AxisArray(imgcol, :y, :x)
        imgr = restrict(imgcolax, (1,2))
        @test pixelspacing(imgr) == (2,2)
        @test pixelspacing(imgcolax) == (1,1)  # issue #347
        @inferred(restrict(imgcolax, Axis{:y}))
        @inferred(restrict(imgcolax, Axis{:x}))
        restrict(imgcolax, Axis{:y})  # FIXME #628
        restrict(imgcolax, Axis{:x})
        imgmeta = ImageMeta(imgcol, myprop=1)
        @test isa(restrict(imgmeta, (1, 2)), ImageMeta)
        # Issue #395
        img1 = colorview(RGB, fill(0.9, 3, 5, 5))
        img2 = colorview(RGB, fill(N0f8(0.9), 3, 5, 5))
        @test isapprox(channelview(restrict(img1)), channelview(restrict(img2)), rtol=0.01)
        # Issue #655
        tmp = AxisArray(rand(1080,1120,5,10), (:x, :y, :z, :t), (0.577, 0.5770, 5, 2));
        @test size(restrict(tmp, 2), 2) == 561

        # restricting OffsetArrays
        A = ones(4, 5)
        Ao = OffsetArray(A, 0:3, -2:2)
        Aor = restrict(Ao)
        @test axes(Aor) == axes(OffsetArray(restrict(A), 0:2, -1:1))

        # restricting AxisArrays with offset axes
        AA = AxisArray(Ao, Axis{:y}(0:3), Axis{:x}(-2:2))
        @test axes(AA) === axes(Ao)
        AAr = restrict(AA)
        axs = axisvalues(AAr)
        @test axes(axs[1])[1] == axes(Aor)[1]
        @test axes(axs[2])[1] == axes(Aor)[2]
        AAA = AxisArray(Aor, axs)  # just test that it's constructable (another way of enforcing agreement)
    end

    # moved to ImageBase.FiniteDiff.fdiv
    @testset "div" begin
        # Test that -div is the adjoint of forwarddiff
        p = rand(3,3,2)
        u = rand(3,3)
        gu = cat(
            ImageBase.FiniteDiff.fdiff(u, dims=1, boundary=:periodic),
            ImageBase.FiniteDiff.fdiff(u, dims=2, boundary=:periodic),
            dims=3)
        @test sum(-Images.div(p) .* u) ≈ sum(p .* gu)
    end

    @testset "Features" begin
        A = zeros(Int, 9, 9); A[5, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5,0,1])
        @test length(blobs) == 1
        blob = blobs[1]
        @test blob.amplitude ≈ 0.3183098861837907
        @test blob.σ ∈ (1.0, (1.0, 1.0))
        @test blob.location == CartesianIndex((5,5))
        @test blob_LoG(A, [1.0]) == blobs
        @test blob_LoG(A, [1.0], (true, false, false)) == blobs
        @test isempty(blob_LoG(A, [1.0], false))
        A = zeros(Int, 9, 9); A[1, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0,0.5,1])
        A = zeros(Int, 9, 9); A[1,5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5,0,1])
        @test all(b.amplitude < 1e-16 for b in blobs)
        blobs = filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], true))
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((1,5))
        @test filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], (true, true, false))) == blobs
        @test isempty(blob_LoG(A, 2.0.^[0,1], (false, true, false)))
        blobs = blob_LoG(A, 2.0.^[0,0.5,1], (true, false, true))
        @test all(b.amplitude < 1e-16 for b in blobs)
        # stub test for N-dimensional blob_LoG:
        A = zeros(Int, 9, 9, 9); A[5, 5, 5] = 1
        blobs = blob_LoG(A, 2.0.^[0.5, 0, 1])
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((5,5,5))
        # kinda anisotropic image
        A = zeros(Int,9,9,9); A[5,4:6,5] .= 1;
        blobs = blob_LoG(A,2 .^ [1.,0,0.5], [1.,3.,1.])
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((5,5,5))
        A = zeros(Int,9,9,9); A[1,1,4:6] .= 1;
        blobs = filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], true, [1.,1.,3.]))
        @test length(blobs) == 1
        @test blobs[1].location == CartesianIndex((1,1,5))
        @test filter(b->b.amplitude > 0.1, blob_LoG(A, 2.0.^[0.5,0,1], (true, true, true, false), [1.,1.,3.])) == blobs
        @test isempty(blob_LoG(A, 2.0.^[0,1], (false, true, false, false), [1.,1.,3.]))
        A = zeros(Int, 9, 9); A[[1:2;5],5].=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5))]
        @test findlocalmaxima(A,2) == [CartesianIndex((1,5)),CartesianIndex((2,5)),CartesianIndex((5,5))]
        @test findlocalmaxima(A,2,false) == [CartesianIndex((2,5)),CartesianIndex((5,5))]
        A = zeros(Int, 9, 9, 9); A[[1:2;5],5,5].=1
        @test findlocalmaxima(A) == [CartesianIndex((5,5,5))]
        @test findlocalmaxima(A,2) == [CartesianIndex((1,5,5)),CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
        @test findlocalmaxima(A,2,false) == [CartesianIndex((2,5,5)),CartesianIndex((5,5,5))]
    end

    @testset "Erode/ dilate" begin
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ae = erode(A)
        @test Ae == zeros(size(A))
        Ad = dilate(A, 1:2)
        Ar = [0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0.8 0.8 0.8 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0.6 0.6;
              0 0 0.6 0.6]
        @test Ad == cat(Ar, Ag, zeros(4,4), dims=3)
        Ae = erode(Ad, 1:2)
        Ar = [0.8 0.8 0 0;
              0.8 0.8 0 0;
              0 0 0 0;
              0 0 0 0]
        Ag = [0 0 0 0;
              0 0 0 0;
              0 0 0 0;
              0 0 0 0.6]
        @test Ae == cat(Ar, Ag, zeros(4,4), dims=3)
        # issue #311
        @test dilate(trues(3)) == trues(3)
        # ImageMeta
        @test arraydata(dilate(ImageMeta(A))) == dilate(A)
        @test arraydata(dilate(ImageMeta(A), 1:2)) == dilate(A, 1:2)
        @test arraydata(erode(ImageMeta(A))) == erode(A)
        @test arraydata(erode(ImageMeta(A), 1:2)) == erode(A, 1:2)
    end

    @testset "Opening / closing" begin
        A = zeros(4,4,3)
        A[2,2,1] = 0.8
        A[4,4,2] = 0.6
        Ao = opening(A)
        @test Ao == zeros(size(A))
        A = zeros(10,10)
        A[4:7,4:7] .= 1
        B = copy(A)
        A[5,5] = 0
        Ac = closing(A)
        @test Ac == B
    end

    @testset "Morphological Top-hat" begin
        A = zeros(13, 13)
        A[2:3, 2:3] .= 1
        Ae = copy(A)
        A[5:9, 5:9] .= 1
        Ao = tophat(A)
        @test Ao == Ae
        Aoo = tophat(Ae)
        @test Aoo == Ae
    end

    @testset "Morphological Bottom-hat" begin
        A = ones(13, 13)
        A[2:3, 2:3] .= 0
        Ae = 1 .- copy(A)
        A[5:9, 5:9] .= 0
        Ao = bothat(A)
        @test Ao == Ae
    end

    @testset "Morphological Gradient" begin
        A = zeros(13, 13)
        A[5:9, 5:9] .= 1
        Ao = morphogradient(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] .= 1
        Ae[6:8, 6:8] .= 0
        @test Ao == Ae
        Aee = dilate(A) - erode(A)
        @test Aee == Ae
    end

    @testset "Morphological Laplacian" begin
        A = zeros(13, 13)
        A[5:9, 5:9] .= 1
        Ao = morpholaplace(A)
        Ae = zeros(13, 13)
        Ae[4:10, 4:10] .= 1
        Ae[5:9, 5:9] .= -1
        Ae[6:8, 6:8] .= 0
        @test Ao == Ae
        Aee = dilate(A) + erode(A) - 2A
        @test Aee == Ae
    end

    @testset "Label components" begin
        A = [true  true  false true;
             true  false true  true]
        lbltarget = [1 1 0 2;
                     1 0 2 2]
        lbltarget1 = [1 2 0 4;
                      1 0 3 4]
        @test label_components(A) == lbltarget
        @test label_components(A, [1]) == lbltarget1
        connectivity = [false true  false;
                        true  false true;
                        false true  false]
        @test label_components(A, connectivity) == lbltarget
        connectivity = trues(3,3)
        lbltarget2 = [1 1 0 1;
                      1 0 1 1]
        @test label_components(A, connectivity) == lbltarget2
        @test component_boxes(lbltarget) == Vector{Tuple}[[(1,2),(2,3)],[(1,1),(2,2)],[(1,3),(2,4)]]
        @test component_lengths(lbltarget) == [2,3,3]
        @test component_indices(lbltarget) == Array{Int64}[[4,5],[1,2,3],[6,7,8]]
        @test component_subscripts(lbltarget) == Array{Tuple}[[(2,2),(1,3)],[(1,1),(2,1),(1,2)],[(2,3),(1,4),(2,4)]]
        @test @inferred(component_centroids(lbltarget)) == Tuple[(1.5,2.5),(4/3,4/3),(5/3,11/3)]
    end

end
