using Images, IndirectArrays, Test, Suppressor
using Images.MappedArrays: mappedarray

@testset "ColorizedArray" begin
    intensity = [0.1 0.3; 0.2 0.4]
    labels = IndirectArray([1 2; 2 1], [RGB(1,0,0), RGB(0,0,1)])
    colorized_A = @suppress_err ColorizedArray(intensity, labels)
    mapped_A = mappedarray(*, intensity, labels)
    
    target = intensity .* labels

    for A in (colorized_A, mapped_A) 
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
    end
end

nothing
