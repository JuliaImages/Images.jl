if VERSION <= v"1.0.5"
    # https://github.com/JuliaLang/julia/pull/29442
    _oneunit(::CartesianIndex{N}) where {N} = _oneunit(CartesianIndex{N})
    _oneunit(::Type{CartesianIndex{N}}) where {N} = CartesianIndex(ntuple(x -> 1, Val(N)))
else
    _oneunit = Base.oneunit
end

if VERSION >= v"1.5.2"
    forced_depwarn(msg, funcsym) = Base.depwarn(msg, funcsym; force=true)
else
    forced_depwarn(msg, funcsym) = Base.depwarn(msg, funcsym)
end
