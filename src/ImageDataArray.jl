import DataArrays

import Base: ==, .==, +, -, *, /, .+, .-, .*, ./, .^, .<, .>

function remove_typevars(types::Vector, args, typevars, unique_names=[])
	for elem in types
		if isa(elem, Symbol) || (isa(elem, Expr) && elem.head == :(.))
			push!(args, elem)
		elseif elem.head == :curly
			T = first(elem.args)
			new_args = []
			remove_typevars(elem.args[2:end], new_args, typevars, unique_names)
			push!(args, Expr(:curly, T, new_args...))
		elseif elem.head == :(<:)
			name, supert = elem.args
			if name in unique_names
				nname = gensym(name)
				push!(unique_names, nname)
				typevar = Expr(:(<:), nname, supert)
			else
				push!(unique_names, name)
				typevar = elem
			end
			push!(typevars, typevar)
			push!(args, typevar.args[1])
		else
			error(elem)
		end
	end
	nothing
end

macro disambiguation_func(func)
	f = func.args[1]
	args = []
	typevars = []
	remove_typevars(func.args[2:end], args, typevars)


	arg_expressions = [Expr(:(::), typ) for typ in args]
	if isempty(typevars)
		funcname = f
	else
		funcname = Expr(:curly, f, typevars...)
	end
	fun = Expr(:call, funcname, arg_expressions...)
	expr = esc(Expr(:function, fun, quote error("lol") end))
	expr
end

#fixes:
#-(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.DataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(ImagesCore.AbstractImageDirect, DataArrays.DataArray))

#fixes:
#-(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.AbstractDataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(ImagesCore.AbstractImageDirect, DataArrays.AbstractDataArray))

#fixes:
#-(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(DataArrays.DataArray, ImagesCore.AbstractImageDirect))

#fixes:
#-(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(DataArrays.AbstractDataArray, ImagesCore.AbstractImageDirect))

#fixes:
#./(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:55
#./(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(./(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:181
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182
#.==(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.==(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.>(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.>(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.>(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.>(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:51
#.*(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:52
#.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, ImagesCore.AbstractImageDirect))

#fixes:
#.+(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:22
#.+(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:297.

@disambiguation_func(.+(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.-(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:40
#.-(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.-(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:177
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.<(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#+(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(+(DataArrays.DataArray, ImagesCore.AbstractImageDirect))

#fixes:
#+(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(+(DataArrays.AbstractDataArray, ImagesCore.AbstractImageDirect))

#fixes:
#-(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.DataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(ImagesCore.AbstractImageDirect, DataArrays.DataArray))

#fixes:
#-(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.AbstractDataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(ImagesCore.AbstractImageDirect, DataArrays.AbstractDataArray))

#fixes:
#-(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(DataArrays.DataArray, ImagesCore.AbstractImageDirect))

#fixes:
#-(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(DataArrays.AbstractDataArray, ImagesCore.AbstractImageDirect))

#fixes:
#./(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:55
#./(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(./(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:181
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182

#.==(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.==(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.>(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.>(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.>(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.>(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:51
#.*(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:52
#.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, ImagesCore.AbstractImageDirect))

#fixes:
#.+(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:22
#.+(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:297.

@disambiguation_func(.+(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.-(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:40
#.-(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.-(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:177
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(ImagesCore.AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.<(ImagesCore.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#+(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(+(DataArrays.DataArray, ImagesCore.AbstractImageDirect))

#fixes:
#+(AbstractArray, ImagesCore.AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(+(DataArrays.AbstractDataArray, ImagesCore.AbstractImageDirect))

#fixes:
#.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:177
#.<(ImagesCore.AbstractImageDirect, Union{DataArrays.DataArray{T<:Any, N<:Any}, DataArrays.PooledDataArray{T<:Any, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/ImagesCore/src/ImageDataArray.jl:46.
@disambiguation_func(.<(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}, DataArrays.DataArray{Bool, N<:Any}}))
#fixes:
#.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:181
#.==(ImagesCore.AbstractImageDirect, Union{DataArrays.DataArray{T<:Any, N<:Any}, DataArrays.PooledDataArray{T<:Any, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/ImagesCore/src/ImageDataArray.jl:46.
@disambiguation_func(.==(ImagesCore.AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}, DataArrays.DataArray{Bool, N<:Any}}))
