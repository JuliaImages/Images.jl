macro forcartesian(sym, sz, ex)
    idim = gensym()
    N = gensym()
    sz1 = gensym()
    isdone = gensym()
    quote
        if !(isempty($(esc(sz))) || prod($(esc(sz))) == 0)
            $N = length($(esc(sz)))
            $sz1 = $(esc(sz))[1]
            $isdone = false
            $(esc(sym)) = ones(Int, $N)
            while !$isdone
                $(esc(ex))
                if ($(esc(sym))[1] += 1) > $sz1
                    $idim = 1
                    while $(esc(sym))[$idim] > $(esc(sz))[$idim] && $idim < $N
                        $(esc(sym))[$idim] = 1
                        $idim += 1
                        $(esc(sym))[$idim] += 1
                    end
                    $isdone = $(esc(sym))[$N] > $(esc(sz))[$N]
                end
            end
        end
    end
end

function cartesian_linear(A::AbstractArray, I::Vector{Int})
    k = 0
    for j = length(I):-1:1
        k = size(A, j)*k + I[j]-1
    end
    k += 1
end
