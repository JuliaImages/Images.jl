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

end
