using Base.Test, Images, Colors, FixedPointNumbers

@testset "Corner" begin
    A = zeros(41,41)
    A[16:26,16:26] = 1

    @testset "Corners API" begin
    	img = zeros(20, 20)
    	img[4:17, 4:17] = 1
    	img[8:13, 8:13] = 0

    	expected_corners = zeros(20, 20)
    	ids = map(CartesianIndex{2}, [(4, 4), (4, 17), (17, 4), (17, 17), (8, 8), (8, 13), (13, 8), (13, 13)])
    	for id in ids expected_corners[id] = 1 end
    	corners = imcorner(img, method = harris)
    	for id in ids @test corners[id]  end
    	@test sum(corners .!= expected_corners) < 3
    	corners = imcorner(img, method = shi_tomasi)
    	for id in ids @test corners[id]  end
    	@test sum(corners .!= expected_corners) < 3

    	ids = map(CartesianIndex{2}, [(4, 4), (4, 17), (17, 4), (17, 17)])
    	corners = imcorner(img, method = kitchen_rosenfeld)
    	for id in ids @test corners[id]  end
    	ids = map(CartesianIndex{2}, [(8, 8), (8, 13), (13, 8), (13, 13)])
    	for id in ids expected_corners[id] = 0 end
    	@test sum(corners .!= expected_corners) < 3
    end

    @testset "Harris" begin
	Ac = imcorner(A, 0.99, true, method = harris)
	# check corners
	@test Ac[16,16]
	@test Ac[16,26]
	@test Ac[26,16]
	@test Ac[26,26]

	@test !all(Ac[1:14, :])
	@test !all(Ac[:, 1:14])
	@test !all(Ac[28:end, :])
	@test !all(Ac[:, 28:end])
	@test !all(Ac[18:24, 18:24])

	Ac = harris(A)

	@test Ac[16,16] ≈ maximum(Ac)
	@test Ac[16,26] ≈ Ac[16,16]
	@test Ac[26,16] ≈ Ac[16,16]
	@test Ac[26,26] ≈ Ac[16,16]

	# check edge intensity
	@test Ac[16,21] ≈ minimum(Ac)
	@test Ac[21,16] ≈ Ac[16,21]
	@test Ac[21,26] ≈ Ac[16,21]
	@test Ac[26,21] ≈ Ac[16,21]

	# check background intensity
	@test Ac[5,5] ≈ 0
	@test Ac[5,36] ≈ 0
	@test Ac[36,5] ≈ 0
	@test Ac[36,36] ≈ 0

    end

    @testset "Shi-Tomasi" begin
	Ac = imcorner(A, 0.99, true, method = shi_tomasi)
	# check corners
	@test Ac[16,16]
	@test Ac[16,26]
	@test Ac[26,16]
	@test Ac[26,26]

	@test !all(Ac[1:14, :])
	@test !all(Ac[:, 1:14])
	@test !all(Ac[28:end, :])
	@test !all(Ac[:, 28:end])
	@test !all(Ac[18:24, 18:24])

	Ac = shi_tomasi(A)
	# check corner intensity
	@test Ac[16,16] ≈ maximum(Ac)
	@test Ac[16,26] ≈ Ac[16,16]
	@test Ac[26,16] ≈ Ac[16,16]
	@test Ac[26,26] ≈ Ac[16,16]

	# check edge intensity
	@test Ac[16,21] ≈ 0
	@test Ac[21,16] ≈ 0
	@test Ac[21,26] ≈ 0
	@test Ac[26,21] ≈ 0

	# check background intensity
	@test Ac[5,5] ≈ 0
	@test Ac[5,36] ≈ 0
	@test Ac[36,5] ≈ 0
	@test Ac[36,36] ≈ 0

    end

    @testset "Kitchen-Rosenfeld" begin
	A[10:30, 10:30] = 1
	A[15:25, 15:25] = 0
	Ac = imcorner(A, 0.99, true, method = kitchen_rosenfeld)
	@test Ac[10, 10]
	@test Ac[10, 30]
	@test Ac[30, 10]
	@test Ac[30, 30]

	@test !all(Ac[1:9, :])
	@test !all(Ac[:, 1:9])
	@test !all(Ac[31:end, :])
	@test !all(Ac[:, 31:end])
	@test !all(Ac[12:28, 12:28])
    end

    @testset "Fast Corners" begin
	img = reshape(1:1:49, 7, 7)

	for i in 1:8
	    corners = fastcorners(img, i)
	    @test all(corners)
	end

	corners = fastcorners(img, 9)
	@test !all(corners[4, 1:end - 1])
	corners[4, 1:end - 1] = true
	@test all(corners)

	corners = fastcorners(img, 10)
	@test !all(corners[3:4, 1:end - 2])
	@test !corners[4, end - 1]
	corners[3:4, 1:end - 2] = true
	corners[4, end - 1] = true
	@test all(corners)

	corners = fastcorners(img, 11)
	@test !all(corners[2:5, 1:end - 3])
	@test !all(corners[3:5, end - 2])
	@test !corners[4, end - 1]
	corners[2:5, 1:end - 3] = true
	corners[3:5, 1:end - 2] = true
	corners[4, end - 1] = true
	@test all(corners)

	corners = fastcorners(img, 12)
	@test !all(corners[1:6, 1:end - 3])
	@test !all(corners[3:5, end - 2])
	@test !corners[4, end - 1]
	corners[1:6, 1:end - 3] = true
	corners[3:5, 1:end - 2] = true
	corners[4, end - 1] = true
	@test all(corners)

	img = parent(Kernel.gaussian(1.4))
	img = vcat(img, img)
	img = hcat(img, img)
	corners = fastcorners(img, 12, 0.05)

	@test corners[5, 5]
	@test corners[5, 14]
	@test corners[14, 5]
	@test corners[14, 14]

	@test all(corners[:, 1:3] .== false)
	@test all(corners[1:3, :] .== false)
	@test all(corners[:, 16:end] .== false)
	@test all(corners[16:end, :] .== false)
	@test all(corners[6:12, 6:12] .== false)

	img = 1 - img

	corners = fastcorners(img, 12, 0.05)

	@test corners[5, 5]
	@test corners[5, 14]
	@test corners[14, 5]
	@test corners[14, 14]

	@test all(corners[:, 1:3] .== false)
	@test all(corners[1:3, :] .== false)
	@test all(corners[:, 16:end] .== false)
	@test all(corners[16:end, :] .== false)
	@test all(corners[6:12, 6:12] .== false)
    end

end

nothing
