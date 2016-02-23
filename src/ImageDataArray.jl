import DataArrays

import Base: ==, .==, +, -, *, /, .+, .-, .*, ./, .^, .<, .>

"""
We use the messages directly printed by the warning, so we need to remove 
things like Array{T<:Float32, N<:Any} and put the T and N into the typevarlist
and Array{T, N} in the args list
"""
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
			# remove non unique typevars. This is not entirely correct, I guess, but 
			# removes the ambiguities way better
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
	expr = esc(Expr(:function, fun, quote error("not implemented") end))
	expr
end


#Needed to add these manually:

#.<(Images.AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.4/Images/src/algorithms.jl:177
#.<(Images.AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.4/Images/src/ImageDataArray.jl:47.
@disambiguation_func(.<(AbstractImageDirect{Bool}, Union{DataArrays.DataArray{Bool}, DataArrays.PooledDataArray{Bool}}))
@disambiguation_func(.<(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N}, DataArrays.PooledDataArray{Bool, R<:Integer, N}}))

@disambiguation_func(.==(AbstractImageDirect{Bool}, Union{DataArrays.DataArray{Bool}, DataArrays.PooledDataArray{Bool}}))
@disambiguation_func(.==(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N}, DataArrays.PooledDataArray{Bool, R<:Integer, N}}))


# More ore less the output from the warnings:

#fixes:
#-(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.DataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(AbstractImageDirect, DataArrays.DataArray))

#fixes:
#-(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.AbstractDataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(AbstractImageDirect, DataArrays.AbstractDataArray))

#fixes:
#-(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(DataArrays.DataArray, AbstractImageDirect))

#fixes:
#-(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(DataArrays.AbstractDataArray, AbstractImageDirect))

#fixes:
#./(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:55
#./(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(./(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.==(AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:181
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182
#.==(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.==(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.>(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.>(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.>(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.>(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:51
#.*(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:52
#.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractImageDirect))

#fixes:
#.+(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:22
#.+(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:297.

@disambiguation_func(.+(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.-(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:40
#.-(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.-(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.<(AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:177
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.<(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#+(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(+(DataArrays.DataArray, AbstractImageDirect))

#fixes:
#+(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(+(DataArrays.AbstractDataArray, AbstractImageDirect))

#fixes:
#-(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.DataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(AbstractImageDirect, DataArrays.DataArray))

#fixes:
#-(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:35
#-(AbstractArray, DataArrays.AbstractDataArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(AbstractImageDirect, DataArrays.AbstractDataArray))

#fixes:
#-(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(-(DataArrays.DataArray, AbstractImageDirect))

#fixes:
#-(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:37
#-(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(-(DataArrays.AbstractDataArray, AbstractImageDirect))

#fixes:
#./(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:55
#./(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(./(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.==(AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:181
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182
#.==(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.==(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.==(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:182

#.==(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.==(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.>(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.>(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.>(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:179
#.>(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.>(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:51
#.*(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.*(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:52
#.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:295.

@disambiguation_func(.*(Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractImageDirect))

#fixes:
#.+(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:22
#.+(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}, AbstractArray...) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:297.

@disambiguation_func(.+(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.-(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:40
#.-(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.-(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#.<(AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:177
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:330.

@disambiguation_func(.<(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.DataArray{Bool, N<:Any}, DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}}))

#fixes:
#.<(AbstractImageDirect, AbstractArray) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:178
#.<(AbstractArray, Union{DataArrays.PooledDataArray, DataArrays.DataArray}) at /home/s/.julia/v0.5/DataArrays/src/broadcast.jl:285.

@disambiguation_func(.<(AbstractImageDirect, Union{DataArrays.PooledDataArray, DataArrays.DataArray}))

#fixes:
#+(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.DataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:326.

@disambiguation_func(+(DataArrays.DataArray, AbstractImageDirect))

#fixes:
#+(AbstractArray, AbstractImageDirect) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:20
#+(DataArrays.AbstractDataArray, AbstractArray) at /home/s/.julia/v0.5/DataArrays/src/operators.jl:349.

@disambiguation_func(+(DataArrays.AbstractDataArray, AbstractImageDirect))

#fixes:
#.<(AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:177
#.<(AbstractImageDirect, Union{DataArrays.DataArray{T<:Any, N<:Any}, DataArrays.PooledDataArray{T<:Any, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/ImagesCore/src/ImageDataArray.jl:46.
@disambiguation_func(.<(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}, DataArrays.DataArray{Bool, N<:Any}}))
#fixes:
#.==(AbstractImageDirect{Bool, N<:Any}, AbstractArray{Bool, N<:Any}) at /home/s/.julia/v0.5/ImagesCore/src/algorithms.jl:181
#.==(AbstractImageDirect, Union{DataArrays.DataArray{T<:Any, N<:Any}, DataArrays.PooledDataArray{T<:Any, R<:Integer, N<:Any}}) at /home/s/.julia/v0.5/ImagesCore/src/ImageDataArray.jl:46.
@disambiguation_func(.==(AbstractImageDirect{Bool, N<:Any}, Union{DataArrays.PooledDataArray{Bool, R<:Integer, N<:Any}, DataArrays.DataArray{Bool, N<:Any}}))
