using Test, Images

@testset "Corner" begin
    A = zeros(41,41)
    A[16:26,16:26] .= 1

    @testset "Corners API" begin
        img = zeros(20, 20)
        img[4:17, 4:17] .= 1
        img[8:13, 8:13] .= 0

        corners = imcorner(img, method = harris)
        expected_corners = falses(20, 20)
        ids = map(CartesianIndex{2}, [(4, 4), (4, 17), (17, 4), (17, 17), (8, 8), (8, 13),
                                      (13, 8), (13, 13)])
        for id in ids expected_corners[id] = true end
        expected_harris = copy(expected_corners)

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

    @testset "Corners Sub-pixel API" begin
        # Validate Base.to_indices implementation
        # for a  HomogeneousPoint which facilitates
        # indexing into a multi-dimensional array.

        # Simulate a grid of voxels
        V = fill(false,41,41,41)
        V[16,16,20]= true
        V[16,26,20]= true
        V[26,16,20]= true
        V[26,26,20]= true

        # Homogeneous coordinates that are in standard form
        pt = HomogeneousPoint((16.3,16.4,20.3,1.0))
        @test V[pt] == true
        pt = HomogeneousPoint((16.6,16.7,20.7,1.0))
        @test V[pt] == false
        pt = HomogeneousPoint((26.3,16.4,20.2,1.0))
        @test V[pt] == true
        pt = HomogeneousPoint((26.6,16.7,20.8,1.0))
        @test V[pt] == false

        # Homogeneous coordinates that are not in standard form
        pt = HomogeneousPoint(5 .* (16.3,16.4,20.3,1.0))
        @test V[pt] == true
        pt = HomogeneousPoint(5 .* (16.6,16.7,20.7,1.0))
        @test V[pt] == false
        pt = HomogeneousPoint(5 .* (26.3,16.4,20.2,1.0))
        @test V[pt] == true
        pt = HomogeneousPoint(5 .* (26.6,16.7,20.8,1.0))
        @test V[pt] == false

        # Simulate corner responses.
        Ac = zeros(41,41)
        Ac[16,15:17] .=  [0.00173804, 0.00607446, 0.00458574]
        Ac[15:17,16] .=  [0.00173804, 0.00607446, 0.00458574]
        Ac[16,25:27] .=  [0.00458574, 0.00607446, 0.00173804]
        Ac[15:17,26] .=  [0.00173804, 0.00607446, 0.00458574]
        Ac[25:27,16] .=  [0.00458574, 0.00607446, 0.00173804]
        Ac[26,15:17] .=  [0.00173804, 0.00607446, 0.00458574]
        Ac[26,25:27] .=  [0.00458574, 0.00607446, 0.00173804]
        Ac[25:27,26] .=  [0.00458574, 0.00607446, 0.00173804]

        # Simulate corner detections.
        I = fill(false,41,41)
        I[16,16]= true
        I[16,26]= true
        I[26,16]= true
        I[26,26]= true

        # Validate Base.to_indices implementation
        # for a planar HomogeneousPoint which facilitates
        # indexing into a matrix.

        # Homogeneous coordinates that are in standard form
        pt = HomogeneousPoint((16.3,16.4,1.0))
        @test I[pt] == true
        pt = HomogeneousPoint((16.6,16.7,1.0))
        @test I[pt] == false
        pt = HomogeneousPoint((26.3,16.4,1.0))
        @test I[pt] == true
        pt = HomogeneousPoint((26.6,16.7,1.0))
        @test I[pt] == false

        # Homogeneous coordinates that are not in standard form
        pt = HomogeneousPoint(5 .* (16.3,16.4,1.0))
        @test I[pt] == true
        pt = HomogeneousPoint(5 .* (16.6,16.7,1.0))
        @test I[pt] == false
        pt = HomogeneousPoint(5 .* (26.3,16.4,1.0))
        @test I[pt] == true
        pt = HomogeneousPoint(5 .* (26.6,16.7,1.0))
        @test I[pt] == false

        # Check sub-pixel refinement on Harris corner response.
        pts = corner2subpixel(Ac, I)
        @test pts[1].coords[1] ≈ 16.24443189348239
        @test pts[1].coords[2] ≈ 16.24443189348239
        @test pts[1].coords[3] ≈ 1
        @test pts[2].coords[1] ≈ 16.24443189348239
        @test pts[2].coords[2] ≈ 25.75556810651761
        @test pts[2].coords[3] ≈ 1
        @test pts[3].coords[1] ≈ 25.75556810651761
        @test pts[3].coords[2] ≈ 16.24443189348239
        @test pts[3].coords[3] ≈ 1
        @test pts[4].coords[1] ≈ 25.75556810651761
        @test pts[4].coords[2] ≈ 25.75556810651761
        @test pts[4].coords[3] ≈ 1

        # Check imcorners_subpixel API.
        img = zeros(20, 20)
        img[4:17, 4:17] .= 1
        img[8:13, 8:13] .= 0

        ids = map(HomogeneousPoint,
            [(4.244432486132958, 4.244432486132958, 1.0),
             (4.244432486132958, 16.755567513867042, 1.0),
             (8.244432486132958, 8.244432486132958, 1.0),
             (8.244432486132958, 12.755567513867042, 1.0),
             (12.755567513867042, 8.244432486132958, 1.0),
             (12.755567513867042, 12.755567513867042, 1.0),
             (16.755567513867042, 4.244432486132958, 1.0),
             (16.755567513867042, 16.755567513867042, 1.0)])

        # Default.
        corner_pts = imcorner_subpixel(img, method = harris)
        @test length(corner_pts) == length(ids)
        for i = 1:length(ids)
            @test all(ids[i].coords .≈ corner_pts[i].coords)
        end

        # User specifies threshold.
        corner_pts = imcorner_subpixel(img, 0.005, method = harris)
        @test length(corner_pts) == length(ids)
        for i = 1:length(ids)
            @test all(ids[i].coords .≈ corner_pts[i].coords)
        end

        # User specifies percentile.
        corner_pts = imcorner_subpixel(img, Percentile(98), method = harris)
        @test length(corner_pts) == length(ids)
        for i = 1:length(ids)
            @test all(ids[i].coords .≈ corner_pts[i].coords)
        end

        # Check  border cases for corner2subpixel.
        responses = zeros(5,5)
        corner_indicator = fill(false,5,5)
        corner_indicator[1,3] = true
        corner_indicator[3,1] = true
        corner_indicator[3,5] = true
        corner_indicator[5,3] = true
        responses[1,3] = 0.5
        responses[1,2] = 0.2
        responses[5,3] = 0.5
        responses[5,2] = 0.2
        responses[3,1] = 0.5
        responses[2,1] = 0.2
        responses[3,5] = 0.5
        responses[2,5] = 0.2
        pts = corner2subpixel(responses, corner_indicator)
        @test pts[1].coords[1] ≈ 1
        @test pts[1].coords[2] ≈ 3
        @test pts[1].coords[3] ≈ 1
        @test pts[2].coords[1] ≈ 3
        @test pts[2].coords[2] ≈ 1
        @test pts[2].coords[3] ≈ 1
        @test pts[3].coords[1] ≈ 3
        @test pts[3].coords[2] ≈ 5
        @test pts[3].coords[3] ≈ 1
        @test pts[4].coords[1] ≈ 5
        @test pts[4].coords[2] ≈ 3
        @test pts[4].coords[3] ≈ 1

        # Test functionality using OffsetArray
        img = zeros(20, 20)
        img[4:17, 4:17] .= 1
        img[8:13, 8:13] .= 0

        for s = -100:50:100
            for t = -100:50:100
                imgo = OffsetArray(img, (s, t))
                ids = map(HomogeneousPoint,
                [(4.244432486132958 + t, 4.244432486132958 + s, 1.0),
                (4.244432486132958 + t, 16.755567513867042 + s, 1.0),
                (8.244432486132958 + t, 8.244432486132958 + s, 1.0),
                (8.244432486132958 + t, 12.755567513867042 + s, 1.0),
                (12.755567513867042 + t, 8.244432486132958 + s, 1.0),
                (12.755567513867042 + t, 12.755567513867042 + s, 1.0),
                (16.755567513867042 + t, 4.244432486132958 + s, 1.0),
                (16.755567513867042 + t, 16.755567513867042 + s, 1.0)])

                # Default.
                corner_pts_offset = imcorner_subpixel(imgo, method = harris)
                @test length(corner_pts_offset) == length(ids)
                for i = 1:length(ids)
                    @test all(ids[i].coords .≈ corner_pts_offset[i].coords)
                end

                # User specifies threshold.
                corner_pts_offset = imcorner_subpixel(imgo, 0.005, method = harris)
                @test length(corner_pts_offset) == length(ids)
                for i = 1:length(ids)
                    @test all(ids[i].coords .≈ corner_pts_offset[i].coords)
                end

                # User specifies percentile.
                corner_pts_offset = imcorner_subpixel(imgo, Percentile(98), method = harris)
                @test length(corner_pts_offset) == length(ids)
                for i = 1:length(ids)
                    @test all(ids[i].coords .≈ corner_pts_offset[i].coords)
                end
            end
        end
    end

    @testset "Harris" begin
    Ac = imcorner(A, Percentile(99), method = harris)
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
    Ac = imcorner(A, Percentile(99), method = shi_tomasi)
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
    A[10:30, 10:30] .= 1
    A[15:25, 15:25] .= 0
    Ac = imcorner(A, Percentile(99), method = kitchen_rosenfeld)
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
    corners[4, 1:end - 1] .= true
    @test all(corners)

    corners = fastcorners(img, 10)
    @test !all(corners[3:4, 1:end - 2])
    @test !corners[4, end - 1]
    corners[3:4, 1:end - 2] .= true
    corners[4, end - 1] = true
    @test all(corners)

    corners = fastcorners(img, 11)
    @test !all(corners[2:5, 1:end - 3])
    @test !all(corners[3:5, end - 2])
    @test !corners[4, end - 1]
    corners[2:5, 1:end - 3] .= true
    corners[3:5, 1:end - 2] .= true
    corners[4, end - 1] = true
    @test all(corners)

    corners = fastcorners(img, 12)
    @test !all(corners[1:6, 1:end - 3])
    @test !all(corners[3:5, end - 2])
    @test !corners[4, end - 1]
    corners[1:6, 1:end - 3] .= true
    corners[3:5, 1:end - 2] .= true
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

    img = 1 .- img

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
