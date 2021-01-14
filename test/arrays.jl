using Images, IndirectArrays, Test

@testset "ColorizedArray" begin
    intensity = [0.1 0.3; 0.2 0.4]
    labels = IndirectArray([1 2; 2 1], [RGB(1,0,0), RGB(0,0,1)])
    A = ColorizedArray(intensity, labels)
    target = intensity .* labels
    @test eltype(A) == RGB{Float64}
    @test size(A) == (2,2)
    @test axes(A) == (Base.OneTo(2), Base.OneTo(2))
    for i = 1:4
        @test A[i] === target[i]
    end
    for j = 1:2, i = 1:2
        @test A[i,j] === target[i,j]
    end
    for (a,t) in zip(A, target)
        @test a === t
    end

    intensity1 = [0.1 0.3 0.6; 0.2 0.4 0.1]
   labels1 = IndirectArray([1 2 3; 2 1 3], [RGB(1,0,0), RGB(0,0,1),RGB(0,1,0)])
   A1 = ColorizedArray(intensity1, labels1)
   target1 = intensity1 .* labels1
   @test eltype(A1) == RGB{Float64}
   @test size(A1) == (2,3)
   @test axes(A1) == (1:2, 1:3)
   for i = 1:6
       @test A1[i] === target1[i]
   end
   for j = 1:3, i = 1:2
       @test A1[i,j] === target1[i,j]
   end
   for (a,t) in zip(A1, target1)
       @test a === t
   end
end

nothing
