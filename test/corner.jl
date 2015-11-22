using FactCheck, Base.Test, Images, Colors, FixedPointNumbers

facts("Corner") do
    A = zeros(41,41)
    A[16:26,16:26] = 1

	context("Harris") do
		Ac = Images.imcorner(A,method="harris")
		# check corner intensity
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
		Ac = Images.imcorner(A,method="shi-tomasi")
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

end
