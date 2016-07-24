using FactCheck, Base.Test, Images, Colors, FixedPointNumbers

facts("Corner") do
    A = zeros(41,41)
    A[16:26,16:26] = 1

	context("Harris") do
		Ac = Images.imcorner(A, method = harris)
		# check corners
		@fact Ac[16,16] --> 1.0
		@fact Ac[16,26] --> 1.0
		@fact Ac[26,16] --> 1.0
		@fact Ac[26,26] --> 1.0

		@fact all(Ac[1:14, :] .== 0.0) --> true
		@fact all(Ac[:, 1:14] .== 0.0) --> true
		@fact all(Ac[28:end, :] .== 0.0) --> true
		@fact all(Ac[:, 28:end] .== 0.0) --> true
		@fact all(Ac[18:24, 18:24] .== 0.0) --> true

		Ac = Images.harris(A)

		@fact Ac[16,16] --> roughly(maximum(Ac))
		@fact Ac[16,26] --> roughly(Ac[16,16])
		@fact Ac[26,16] --> roughly(Ac[16,16])
		@fact Ac[26,26] --> roughly(Ac[16,16])

		# check edge intensity
		@fact Ac[16,21] --> roughly(minimum(Ac))
		@fact Ac[21,16] --> roughly(Ac[16,21])
		@fact Ac[21,26] --> roughly(Ac[16,21])
		@fact Ac[26,21] --> roughly(Ac[16,21])

		# check background intensity
		@fact Ac[5,5] --> roughly(0)
		@fact Ac[5,36] --> roughly(0)
		@fact Ac[36,5] --> roughly(0)
		@fact Ac[36,36] --> roughly(0)

	end

	context("Shi-Tomasi") do
		Ac = Images.imcorner(A, method = shi_tomasi)
		# check corners
		@fact Ac[16,16] --> 1.0
		@fact Ac[16,26] --> 1.0
		@fact Ac[26,16] --> 1.0
		@fact Ac[26,26] --> 1.0

		@fact all(Ac[1:14, :] .== 0.0) --> true
		@fact all(Ac[:, 1:14] .== 0.0) --> true
		@fact all(Ac[28:end, :] .== 0.0) --> true
		@fact all(Ac[:, 28:end] .== 0.0) --> true
		@fact all(Ac[18:24, 18:24] .== 0.0) --> true

		Ac = Images.shi_tomasi(A)
		# check corner intensity
		@fact Ac[16,16] --> roughly(maximum(Ac))
		@fact Ac[16,26] --> roughly(Ac[16,16])
		@fact Ac[26,16] --> roughly(Ac[16,16])
		@fact Ac[26,26] --> roughly(Ac[16,16])

		# check edge intensity
		@fact Ac[16,21] --> roughly(0)
		@fact Ac[21,16] --> roughly(0)
		@fact Ac[21,26] --> roughly(0)
		@fact Ac[26,21] --> roughly(0)

		# check background intensity
		@fact Ac[5,5] --> roughly(0)
		@fact Ac[5,36] --> roughly(0)
		@fact Ac[36,5] --> roughly(0)
		@fact Ac[36,36] --> roughly(0)

	end

	context("Kitchen-Rosenfeld") do
		A[10:30, 10:30] = 1
		A[15:25, 15:25] = 0
		Ac = Images.imcorner(A, method = kitchen_rosenfeld)
		@fact Ac[15, 15] --> 1.0
		@fact Ac[15, 25] --> 1.0
		@fact Ac[25, 15] --> 1.0
		@fact Ac[25, 25] --> 1.0

		@fact all(Ac[1:13, :] .== 0.0) --> true
		@fact all(Ac[:, 1:13] .== 0.0) --> true
		@fact all(Ac[27:end, :] .== 0.0) --> true
		@fact all(Ac[:, 27:end] .== 0.0) --> true
	end

	context("Fast Corners") do 

		img = reshape(1:1:49, 7, 7)

		for i in 1:8
			corners = Images.fastcorners(img, i)
			@fact all(corners) --> true
		end

		corners = Images.fastcorners(img, 9)
		@fact all(corners[4, 1:end - 1]) --> false
		corners[4, 1:end - 1] = true
		@fact all(corners) --> true

		corners = Images.fastcorners(img, 10)
		@fact all(corners[3:4, 1:end - 2]) --> false
		@fact corners[4, end - 1] --> false
		corners[3:4, 1:end - 2] = true
		corners[4, end - 1] = true
		@fact all(corners) --> true

		corners = Images.fastcorners(img, 11)
		@fact all(corners[2:5, 1:end - 3]) --> false
		@fact all(corners[3:5, end - 2]) --> false
		@fact corners[4, end - 1] --> false
		corners[2:5, 1:end - 3] = true
		corners[3:5, 1:end - 2] = true
		corners[4, end - 1] = true
		@fact all(corners) --> true

		corners = Images.fastcorners(img, 12)
		@fact all(corners[1:6, 1:end - 3]) --> false
		@fact all(corners[3:5, end - 2]) --> false
		@fact corners[4, end - 1] --> false
		corners[1:6, 1:end - 3] = true
		corners[3:5, 1:end - 2] = true
		corners[4, end - 1] = true
		@fact all(corners) --> true

		img = gaussian2d(1.4)
		img = vcat(img, img)
		img = hcat(img, img)
		corners = Images.fastcorners(img, 12, 0.05)

		@fact corners[5, 5] --> true
		@fact corners[5, 14] --> true
		@fact corners[14, 5] --> true
		@fact corners[14, 14] --> true
		
		@fact all(corners[:, 1:3] .== false) --> true
		@fact all(corners[1:3, :] .== false) --> true
		@fact all(corners[:, 16:end] .== false) --> true
		@fact all(corners[16:end, :] .== false) --> true
		@fact all(corners[6:12, 6:12] .== false) --> true

		img = 1 - img

		corners = Images.fastcorners(img, 12, 0.05)

		@fact corners[5, 5] --> true
		@fact corners[5, 14] --> true
		@fact corners[14, 5] --> true
		@fact corners[14, 14] --> true

		@fact all(corners[:, 1:3] .== false) --> true
		@fact all(corners[1:3, :] .== false) --> true
		@fact all(corners[:, 16:end] .== false) --> true
		@fact all(corners[16:end, :] .== false) --> true
		@fact all(corners[6:12, 6:12] .== false) --> true

	end

end
