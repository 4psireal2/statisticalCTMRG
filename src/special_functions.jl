using TensorKit

mutable struct envs
    ul::TensorMap
    dl::TensorMap
    dr::TensorMap
    ur::TensorMap

    l::TensorMap
    d::TensorMap
    r::TensorMap
    u::TensorMap
end

function get_corner_sv(env_arr)
    sv_arr = []
    for i in eachindex(env_arr)
        _, Sul, _ = svd_full(env_arr[i].ul)
        _, Sur, _ = svd_full(env_arr[i].ur)
        _, Sdl, _ = svd_full(env_arr[i].dl)
        _, Sdr, _ = svd_full(env_arr[i].dr)

        #=
        sv_ul = diag(Sul.data)
        sv_ur = diag(Sur.data)
        sv_dl = diag(Sdl.data)
        sv_dr = diag(Sdr.data)
        =#
        
        Sul_data = convert(Array, Sul)
        Sur_data = convert(Array, Sur)
        Sdl_data = convert(Array, Sdl)
        Sdr_data = convert(Array, Sdr)
        sv_ul = diag(Sul_data)
        sv_ur = diag(Sur_data)
        sv_dl = diag(Sdl_data)
        sv_dr = diag(Sdr_data)


        append!(sv_arr, sv_ul)
        append!(sv_arr, sv_ur)
        append!(sv_arr, sv_dl)
        append!(sv_arr, sv_dr)
    end
    return sv_arr
end

function compare_sv(sv_arr, sv_arr_old)
        if length(sv_arr) != length(sv_arr_old)
            #@info "the sectors of the environment tensors have not yet converged!"
            return :unconverged
        else
            max = maximum(maximum.(abs.(sv_arr .- sv_arr_old)))
        end
        return max
end

function rank(A::TensorMap)
    rank_dom = length(dims(codomain(A)))
    rank_codom = length(dims(domain(A)))
    
    return rank_dom + rank_codom
end

function sqrtTM(S)
    Smat = convert_TM_to_mat(S)
    #display(typeof(Smat))
    B = sqrt.(diag(Smat))
    Ssqrt = TensorMap(diagm(B), codomain(S), domain(S))
    return Ssqrt
end

function p_arr_inv(a, tol)
    
    N = maximum(a)
    
    res = [el ≥ tol*N ? 1/el : 0.0 for el in a]
    
    if 0.0 in res
        trunc = count(i->(i == 0.0), res)
        l = length(res)
        @warn "we used a pseudo inverse for inverting the SV as they were smaller than $tol. We cut $trunc of $l SV."
    end

    return res
end



function pinv_sqrt(S, tol)
    #Smat = S.data
    
    #this awkward way was somehow nessessary - will be changed in future.
    S_dict = convert(Dict,S)
    A = S_dict[:data]["Trivial()"]
    Smat = A*Matrix(I,size(A)[1],size(A)[1])
    #Smat = A
    #display(size(Smat))
    
    B = sqrt.(diag(Smat))
    C = p_arr_inv(B, tol)
    F = diagm(C)
    #display(size(C))
    S_inv_sqrt = TensorMap(F, codomain(S) ← domain(S))
    return S_inv_sqrt
end



function convert_TM_to_mat(A)
    """
    Convert a TensorMap to a matrix.
    """
    Adict = convert(Dict, A)
    AMat = Adict[:data]["Trivial()"]
    return AMat
end


function apply_mat(x, L)
    b = Tensor(x, domain(L))
    c = L * b
    return c.data[:,1]
end

function apply_mat_adj(x, L)
    b = Tensor(x, domain(L'))
    c = L' * b
    return c.data[:,1]
end

function tsvd_GKL(A; χ::Int = 20, space_type = ℂ)
    """
    Truncated SVD via Golub-Kahan-Lanczos (GKL) bidiagonalization algorithm.
    
    A ≈ S_kr . U_kr . V_kr',
    U_kr is a bidiagonal matrix with the largest χ singular values of A on the diagonal.
    """
    
    b = rand(TensorKit.dim(codomain(A)))
    
    f_A = x -> apply_mat(x, A)
    f_A_adj = x -> apply_mat_adj(x, A)            
    S_kr, U_kr, V_kr, info = svdsolve((f_A, f_A_adj), b, χ, :LR, krylovdim = 2*χ)
                
    #put some warning here if the GKL did not work.
    if χ > info.converged
        @warn "here not the GKL procedure was not able to converge the desired number of singular values! the SVD is now calculated by conventional means..."
        #@info "number of Krylov-subspace restarts, number of operations with the linear map, list of residuals" info.numiter info.numops info.normres

        #@info "the SVD is now calculated by conventional means..."
        U, S, Vd = svd_trunc(A, trunc = truncspace(space_type^χ))
        return U, S, Vd

    end
    
    # turn output into TensorMaps
    U_mat = U_kr[1]
    for i in 2:χ     
        U_mat = hcat(U_mat, U_kr[i])
    end
    U_TM = TensorMap(U_mat, codomain(A) ← space_type^χ)

    V_mat = adjoint(V_kr[1])
    for i in 2:χ
        V_mat = vcat(V_mat, adjoint(V_kr[i]))
    end
    V_d_TM = TensorMap(V_mat, space_type^χ ← domain(A))

    S_TM = TensorMap(diagm(S_kr[1:χ]), space_type^χ ← space_type^χ)
                    
    return U_TM, S_TM, V_d_TM
end


#from TensorKitAD
function _elementwise_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end


function truncate_sector_by_error(aTensor::TensorMap, atol::Real)
    I = sectortype(aTensor)
    T = eltype(aTensor)

    # Dictionaries to hold matrix data for each sector
    U_data = Dict{I, Matrix{T}}()
    S_data = Dict{I, Matrix{T}}()
    V_data = Dict{I, Matrix{T}}()
    
    new_dims = Dict{I, Int}()
    #ϵ_total = 0.0

    for (c, b) in blocks(aTensor)
        U, S, V_T, ϵ = svd_trunc(b, trunc=truncerror(; atol = atol))
        
        U_data[c] = U
        S_data[c] = S
        V_data[c] = V_T
        
        new_dims[c] = size(S, 1)
    #    ϵ_total += ϵ^2
    end

    SpaceType = typeof(space(aTensor, 1))
    V_virt = SpaceType(new_dims...)
    
    U_trunc = TensorMap(U_data, codomain(aTensor), V_virt)    
    S_trunc = TensorMap(S_data, V_virt, V_virt)
    V_trunc = TensorMap(V_data, V_virt, domain(aTensor))

    #return U_trunc, S_trunc, V_trunc, ϵ_total
    return U_trunc, S_trunc, V_trunc
end


function wrapper_tsvd(A, Bond_env; Space_type = ℝ, svd_type = :GKL)

    if svd_type == :accuracy
        U, S, Vd = svd_trunc(A, trunc = truncerr(0.02))

    elseif svd_type == :envbond
        U, S, Vd = svd_trunc(A, trunc = truncrank(Bond_env))

    elseif svd_type == :trial1
        V_trunc = ℤ₂Space(0 => Bond_env, 1 => Bond_env)
        U, S, Vd = svd_trunc(A,  trunc=truncerror(; atol = 1e-10) & truncspace(V_trunc))
    
    elseif svd_type == :trial2
        U, S, Vd = truncate_sector_by_error(A, 1.0e-10)
    
    elseif svd_type == :GKL
        U, S, Vd = tsvd_GKL(A, χ = Bond_env, space_type = Space_type)
            
    else
        display("You have not specified a truncation type in function: unique_svd")                
    end

    return U, S, Vd
end


function unique_tsvd(A, Bond_env; Space_type = ℝ, svd_type = :GKL, split = :yes)
    """
    Truncated SVD with gauge fixing for environment tensors. This is needed when the 
    derivative of the algorithm is taken wrt the tensors, as in the quantum case.
    """

    if svd_type == :accuracy
        U, S, Vd = tsvd(A, trunc = notrunc(), alg = TensorKit.SDD())

    elseif svd_type == :envbond

        U, S, Vd = tsvd(A, trunc = truncspace(Space_type^Bond_env), alg = TensorKit.SDD())
    
    elseif svd_type == :GKL

        U, S, Vd = tsvd_GKL(A, χ = Bond_env, space_type = Space_type)
            
    else
        display("you have not specified a truncation type in function: unique_svd")                
    end

    #display("SDD is used now!")
    

    #be careful- when we include symmetries we have to go through the symmetry sectors indvidually
    UMat = convert_TM_to_mat(U)
    VdMat = convert_TM_to_mat(Vd)
    
    #here we fix the gauge for elementwise convergence of the environment tensors
    if Space_type == ℝ
        absmax = x -> abs(minimum(x)) > abs(maximum(x)) ? minimum(x) : maximum(x)
        index_comp = x -> findmax(x)[2] > findmin(x)[2] ? minimum(x) : maximum(x)
        descide_func = x -> isapprox(abs(minimum(x)), abs(maximum(x)); rtol = 1e-8) ? index_comp(x) : absmax(x)
        fix_mat = diagm(sign.(map(descide_func, eachcol(UMat))))
    end
    
    if Space_type == ℂ
        absmax = x -> x[partialsortperm(abs.(x), 1:2; rev = true)][1]
        index_comp = x -> partialsortperm(abs.(x), 1:2; rev = true)[2] > partialsortperm(abs.(x), 1:2; rev = true)[1] ? x[partialsortperm(abs.(x), 1:2; rev = true)][1] : x[partialsortperm(abs.(x), 1:2; rev = true)][2]
        descide_func = x -> isapprox(abs.(x)[partialsortperm(abs.(x), 1:2; rev = true)][1] , abs.(x)[partialsortperm(abs.(x), 1:2; rev = true)][2] ; rtol = 10^-8) ? index_comp(x) : absmax(x)
        fix_mat = diagm(exp.(-angle.(map(descide_func, eachcol(UMat)))*im))
    end
    
    Ufixed1 = UMat*fix_mat
    Vdfixed1 = fix_mat' * VdMat
    if split == :yes
        Ufixed = reshape(Ufixed1, (dim(codomain(U)[1]), dim(codomain(U)[2]), dim(codomain(U)[3]), dim(domain(U))))
        Vdfixed = reshape(Vdfixed1, (dim(codomain(Vd)), dim(domain(Vd)[1]), dim(domain(Vd)[2]), dim(domain(Vd)[3]) ))

        UfTensor = TensorMap(Ufixed, codomain(U) ← domain(U))
        VdfTensor = TensorMap(Vdfixed, codomain(Vd) ← domain(Vd))
        
    elseif split == :no
        UfTensor = TensorMap(Ufixed1, codomain(U) ← domain(U))
        VdfTensor = TensorMap(Vdfixed1, codomain(Vd) ← domain(Vd))
        
    end
        
    return UfTensor , S , VdfTensor
end


#here I adjust the function to do not cut into degenerate singular values!
#It might be good to define an additional stuct TruncationDimensionDegenerate that can be imported such that we can use multiple dispatch.
#This might also be good, as we can pass an accuracy for the comparison of the singular values.
struct TruncationDimensionDegenerate
    dim::Int
    rtol::Real
end
truncdimdeg(d::Int, t::Real) = TruncationDimensionDegenerate(d,t)

function _truncate!(V::TensorKit.SectorVector, trunc::TruncationDimensionDegenerate, p=2)
    I = keytype(V)
    S = real(eltype(valtype(V)))
    truncdim = TensorKit.SectorDict{I,Int}(c => length(v) for (c, v) in V)

    last_cut_sv = 0.0 #define variable for the last SV we cut.

    while sum(dim(c) * d for (c, d) in truncdim) > trunc.dim
        cmin = TensorKit._findnexttruncvalue(V, truncdim, p)
        cmin === nothing && break

        last_cut_sv = V[cmin][end] #remember last SV that you cut

        truncdim[cmin] -= 1 
    end

    if last_cut_sv == 0.0
        
    else

        cmin = TensorKit._findnexttruncvalue(V, truncdim, p)
        isnothing(cmin) && return truncdim #if there is nothing left to truncate, return ...
        
        largest_sv_remaining = V[cmin][truncdim[cmin]] #find the smallest SV remaining

        while isapprox(largest_sv_remaining, last_cut_sv; rtol = trunc.rtol) #here one compares - the rtol could also be passed in a new TruncationDimensionDegenerate. trunc.rtol
            #display("I am cutting extra!")
            truncdim[cmin] -= 1 #if the smallest remaining SV is roughly degenerate with the last one we cut, cut it as well. 
            cmin = TensorKit._findnexttruncvalue(V, truncdim, p) #find the next one and see if it is degenerate as well.
            isnothing(cmin) && break 
            largest_sv_remaining = V[cmin][truncdim[cmin]] #new smallest SV that is remaining
        end
    end

    truncerr = TensorKit._norm((c => view(v, (truncdim[c] + 1):length(v)) for (c, v) in V), p,
                     zero(S))
    for (c, v) in V
        TensorKit.resize!(v, truncdim[c])
    end
    return V, truncerr
end


function initialize_multisite(loc; Space_type=ℝ, initialize_seed = 1236, initialize_random = false, log_info = false)
    """
    Params:
    - Space_type: type of field or vector space for the environment tensor
    """
    env_arr = Array{Any}(undef, length(loc))
    
    if Space_type == ℝ
        trivialspace = ProductSpace{CartesianSpace, 0}()
    elseif Space_type == ℂ
        trivialspace =  ProductSpace{ComplexSpace, 0}()
    elseif Space_type == :Z0
        trivialspace =  ProductSpace{ComplexSpace, 0}()
    elseif Space_type == :U1
        trivialspace = ProductSpace{GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 0}()
    elseif Space_type == :Z2
        #trivialspace = one(ℤ₂Space(0 => 1))
        trivialspace = one(ℤ₂Space(0 => 1, 1 => 1))
    elseif Space_type == :vZ2
        trivialspace = ProductSpace{GradedSpace{Z2Irrep, Tuple{Int64, Int64}}, 0}()
    elseif Space_type == :O2_int
        trivialspace = ProductSpace{GradedSpace{CU1Irrep, TensorKit.SortedVectorDict{CU1Irrep, Int64}}, 0}()
    elseif Space_type == :O2_halfint
        trivialspace = ProductSpace{GradedSpace{CU1Irrep, TensorKit.SortedVectorDict{CU1Irrep, Int64}}, 0}()
    else
        @warn "something went wrong when initializing the environments."
    end


    if Space_type == ℝ

        C_ul = TensorMap([1.0], Space_type^1 ← Space_type^1)
        C_dr = TensorMap([1.0], Space_type^1 ← Space_type^1)
        C_dl = TensorMap([1.0], trivialspace ← Space_type^1 ⊗ Space_type^1)
        C_ur = TensorMap([1.0], Space_type^1 ⊗ Space_type^1 ← trivialspace)

    elseif Space_type == ℂ

        C_ul = TensorMap([1.0 + 0.0*im], Space_type^1 ← Space_type^1)
        C_dr = TensorMap([1.0 + 0.0*im], Space_type^1 ← Space_type^1)
        C_dl = TensorMap([1.0 + 0.0*im], trivialspace ← Space_type^1 ⊗ Space_type^1)
        C_ur = TensorMap([1.0 + 0.0*im], Space_type^1 ⊗ Space_type^1 ← trivialspace)

    elseif Space_type == :Z0

        V = ComplexSpace(1)
        C_ul = TensorMap(ones, V ← V)
        C_dr = TensorMap(ones, V ← V)
        C_dl = TensorMap(ones, trivialspace ← V ⊗ V)
        C_ur = TensorMap(ones, V ⊗ V ← trivialspace)

    elseif Space_type == :U1
        
        V = U₁Space(0 => 1)
        
        #=
        C_ul = TensorMap([ComplexF64(1.0+0.0*im)], V ← V)
        C_dr = TensorMap([ComplexF64(1.0+0.0*im)], V ← V)
        C_dl = TensorMap([ComplexF64(1.0+0.0*im)], trivialspace ← V ⊗ V)
        C_ur = TensorMap([ComplexF64(1.0+0.0*im)], V ⊗ V ← trivialspace)
        =#
#= 
        C_ul = TensorMap(randn, V ← V)
        C_dr = TensorMap(randn, V ← V)
        C_dl = TensorMap(randn, trivialspace ← V ⊗ V)
        C_ur = TensorMap(randn, V ⊗ V ← trivialspace) =#

        
        C_ul = TensorMap([1.0], V ← V)
        C_dr = TensorMap([1.0], V ← V)
        C_dl = TensorMap([1.0], trivialspace ← V ⊗ V)
        C_ur = TensorMap([1.0], V ⊗ V ← trivialspace)
        
    elseif Space_type == :Z2 || Space_type == :vZ2
        
        #V = ℤ₂Space(0 => 1)
        V = ℤ₂Space(0 => 1, 1 => 1)
        C_ul = ones(V ← V)
        C_dr = ones(V ← V)
        C_dl = ones(trivialspace ← V ⊗ V)
        C_ur = ones(V ⊗ V ← trivialspace)

    else
        error("beware! you are trying to initialize CTMRG-environments with new symmetries! generalize the function ini_multisite()!")
    end


    for i in eachindex(loc)

        dim_loc_l = TensorKit.dim(codomain(loc[i])[1])
        dim_loc_d = TensorKit.dim(codomain(loc[i])[2])
        dim_loc_r = TensorKit.dim(domain(loc[i])[1])
        dim_loc_u = TensorKit.dim(domain(loc[i])[2])
        
        space_loc_l = codomain(loc[i])[1]
        space_loc_d = codomain(loc[i])[2]
        space_loc_r = domain(loc[i])[1]
        space_loc_u = domain(loc[i])[2]

        if Space_type == ℝ || Space_type == ℂ
            
            # Tr_l = TensorMap(Matrix(1.0I, dim_loc_l,dim_loc_l), Space_type^1 ← (space_loc_l)' ⊗ space_loc_l ⊗ Space_type^1)
            # Tr_d = TensorMap(Matrix(1.0I, dim_loc_d,dim_loc_d), Space_type^1 ← Space_type^1 ⊗ (space_loc_d)' ⊗ space_loc_d)
            # Tr_r = TensorMap(Matrix(1.0I, dim_loc_r,dim_loc_r), (space_loc_r)' ⊗ space_loc_r ⊗ Space_type^1 ← Space_type^1)
            # Tr_u = TensorMap(Matrix(1.0I, dim_loc_u,dim_loc_u), Space_type^1 ⊗ (space_loc_u)' ⊗ space_loc_u ← Space_type^1)
            
            Tr_l = ones(Space_type^1 ← space_loc_l ⊗ Space_type^1)
            Tr_d = ones(Space_type^1 ← Space_type^1 ⊗ space_loc_d)
            Tr_r = ones(space_loc_r ⊗ Space_type^1 ← Space_type^1)
            Tr_u = ones(Space_type^1 ⊗ space_loc_u ← Space_type^1)

        elseif Space_type == :Z0

            V = ComplexSpace(1)

            # Tr_l = TensorMap(Matrix((1.0)I, dim_loc_l, dim_loc_l), V ← (space_loc_l)' ⊗ space_loc_l ⊗ V)
            # Tr_d = TensorMap(Matrix((1.0)I, dim_loc_d, dim_loc_d), V ← V ⊗ (space_loc_d)' ⊗ space_loc_d)
            # Tr_r = TensorMap(Matrix((1.0)I, dim_loc_r, dim_loc_r), (space_loc_r)' ⊗ space_loc_r ⊗ V ← V)
            # Tr_u = TensorMap(Matrix((1.0)I, dim_loc_u, dim_loc_u), V ⊗ (space_loc_u)' ⊗ space_loc_u ← V)

            Tr_l = TensorMap(ones, V ← (space_loc_l)' ⊗ space_loc_l ⊗ V)
            Tr_d = TensorMap(ones, V ← V ⊗ (space_loc_d)' ⊗ space_loc_d)
            Tr_r = TensorMap(ones, (space_loc_r)' ⊗ space_loc_r ⊗ V ← V)
            Tr_u = TensorMap(ones, V ⊗ (space_loc_u)' ⊗ space_loc_u ← V)
            
        elseif Space_type == :U1
            
            V = U₁Space(0 => 1)

            # Tr_l = TensorMap(randn, V ← (space_loc_l)' ⊗ space_loc_l ⊗ V)
            # Tr_d = TensorMap(randn, V ← V ⊗ (space_loc_d)' ⊗ space_loc_d)
            # Tr_r = TensorMap(randn, (space_loc_r)' ⊗ space_loc_r ⊗ V ← V)
            # Tr_u = TensorMap(randn,  V ⊗ (space_loc_u)' ⊗ space_loc_u ← V)
            
            Tr_l = TensorMap(Matrix((1.0)I, dim_loc_l,dim_loc_l), V ← (space_loc_l)' ⊗ space_loc_l ⊗ V)
            Tr_d = TensorMap(Matrix((1.0)I, dim_loc_d,dim_loc_d), V ← V ⊗ (space_loc_d)' ⊗ space_loc_d)
            Tr_r = TensorMap(Matrix((1.0)I, dim_loc_r,dim_loc_r), (space_loc_r)' ⊗ space_loc_r ⊗ V ← V)
            Tr_u = TensorMap(Matrix((1.0)I, dim_loc_u,dim_loc_u),  V ⊗ (space_loc_u)' ⊗ space_loc_u ← V)
            
        elseif Space_type == :Z2 || Space_type == :vZ2
            
            #V = ℤ₂Space(0 => 1)
            V = ℤ₂Space(0 => 1, 1 => 1)
            
            # Tr_l = TensorMap(randn, V ← (space_loc_l)' ⊗ space_loc_l ⊗ V)
            # Tr_d = TensorMap(randn, V ← V ⊗ (space_loc_d)' ⊗ space_loc_d)
            # Tr_r = TensorMap(randn, (space_loc_r)' ⊗ space_loc_r ⊗ V ← V)
            # Tr_u = TensorMap(randn,  V ⊗ (space_loc_u)' ⊗ space_loc_u ← V)
            
            # Tr_l = TensorMap(Matrix((1.0)I, dim_loc_l, dim_loc_l), V ← (space_loc_l)' ⊗ space_loc_l ⊗ V)
            # Tr_d = TensorMap(Matrix((1.0)I, dim_loc_d, dim_loc_d), V ← V ⊗ (space_loc_d)' ⊗ space_loc_d)
            # Tr_r = TensorMap(Matrix((1.0)I, dim_loc_r, dim_loc_r), (space_loc_r)' ⊗ space_loc_r ⊗ V ← V)
            # Tr_u = TensorMap(Matrix((1.0)I, dim_loc_u, dim_loc_u),  V ⊗ (space_loc_u)' ⊗ space_loc_u ← V)

            Tr_l = ones(V ← space_loc_l ⊗ V)
            Tr_d = ones(V ← V ⊗ space_loc_d)
            Tr_r = ones(space_loc_r ⊗ V ← V)
            Tr_u = ones(V ⊗ space_loc_u ← V)

            # Tr_l = TensorMap(ones, V ← (space_loc_l)' ⊗ space_loc_l ⊗ V)
            # Tr_d = TensorMap(ones, V ← V ⊗ (space_loc_d)' ⊗ space_loc_d)
            # Tr_r = TensorMap(ones, (space_loc_r)' ⊗ space_loc_r ⊗ V ← V)
            # Tr_u = TensorMap(ones,  V ⊗ (space_loc_u)' ⊗ space_loc_u ← V)
            
            Tr_l /= norm(Tr_l)
            Tr_u /= norm(Tr_u)
            Tr_r /= norm(Tr_r)
            Tr_d /= norm(Tr_d)

        end

        env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)

        #display(C_ul)
        #display(Tr_u)
        #env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)
    end
    #display("initialization with symmetric tensors works.")
    return env_arr
end


function initialize_multisite_dp(loc; Space_type = Space_type, initialize_seed = 1236, initialize_real = false)

    env_arr = Array{Any}(undef, length(loc))
    
    if Space_type == ℝ
        nullspace = ProductSpace{CartesianSpace, 0}()
    elseif Space_type == ℂ
        nullspace =  ProductSpace{ComplexSpace, 0}()
    else
        @warn "Something went wrong when initializing the environments."
    end
    #=
    if initialize_real == true
        C_ul = TensorMap([1.0 + 0.0*im], Space_type^1 ← Space_type^1)
        C_dr = TensorMap([1.0 + 0.0*im], Space_type^1 ← Space_type^1)
        C_dl = TensorMap([1.0 + 0.0*im], nullspace ← Space_type^1 ⊗ Space_type^1)
        C_ur = TensorMap([1.0 + 0.0*im], Space_type^1 ⊗ Space_type^1 ← nullspace)
    else
        C_ul = TensorMap([ComplexF64(1.0+1.0*im)], Space_type^1 ← Space_type^1)
        C_dr = TensorMap([ComplexF64(1.0+1.0*im)], Space_type^1 ← Space_type^1)
        C_dl = TensorMap([ComplexF64(1.0+1.0*im)], nullspace ← Space_type^1 ⊗ Space_type^1)
        C_ur = TensorMap([ComplexF64(1.0+1.0*im)], Space_type^1 ⊗ Space_type^1 ← nullspace)
    end
    =#

    dimen = dim(codomain(loc[5])[1])
    n = Int((dimen-1)/2)

    svddimen = dim(codomain(loc[1])[1])

    A = loc[1]
    Adata = reshape(A.data, (svddimen, 2n+1, svddimen, 4n+1))

    B = loc[2]
    Bdata = reshape(B.data, (svddimen, 4n+1, svddimen, 2n+1))

    C = loc[3]
    Cdata = reshape(C.data, (svddimen, 2n+1, svddimen, 4n+1))

    D = loc[4]
    Ddata = reshape(D.data, (svddimen, 4n+1, svddimen, 2n+1))

    O1 = loc[5]
    O1data = reshape(O1.data, (2n+1, 2n+1, 2n+1, 2n+1))

    δc_a_in = loc[6]
    δcdata = reshape(δc_a_in.data, (2n+1, 4n+1, 2n+1, 4n+1))

    O4 = loc[7]
    O4data = reshape(O4.data, (2n+1, 2n+1, 2n+1, 2n+1))

    

    for i in eachindex(loc)
        #display(i)
        if i == 1
            C_ul = TensorMap(O4data[n+1,:,:,n+1], codomain(O4)[2]←domain(O4)[1])
            C_ur = TensorMap(O1data[:,:,n+1,n+1], codomain(O1)[1] ⊗ codomain(O1)[2] ← nullspace)
            C_dr = TensorMap(δcdata[:,2n+1,n+1,:], codomain(δc_a_in)[1] ← domain(δc_a_in)[2])
            C_dl = TensorMap(δcdata[n+1,2n+1,:,:], nullspace ← domain(δc_a_in)[1] ⊗ domain(δc_a_in)[2])

            Tr_l = TensorMap(Ddata[1,:,:,:], codomain(D)[2] ← domain(D))
            Tr_r = TensorMap(Bdata[:,:,1,:], codomain(B) ← domain(B)[2])
            Tr_u = TensorMap(δcdata[:,:,:,2n+1], codomain(δc_a_in)←domain(δc_a_in)[1])
            Tr_d = TensorMap(O1data[:,n+1,:,:], codomain(O1)[1]←domain(O1))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)
        
        elseif i == 2
            C_ul = TensorMap(δcdata[n+1,:,:,2n+1], codomain(δc_a_in)[2]←domain(δc_a_in)[1])
            C_ur = TensorMap(δcdata[:,:,n+1,2n+1], codomain(δc_a_in)[1] ⊗ codomain(δc_a_in)[2] ← nullspace)
            C_dr = TensorMap(O4data[:,n+1,n+1,:], codomain(O4)[1] ← domain(O4)[2])
            C_dl = TensorMap(O1data[n+1,n+1,:,:], nullspace ← domain(O1)[1] ⊗ domain(O1)[2])

            Tr_l = TensorMap(Adata[1,:,:,:], codomain(A)[2] ← domain(A))
            Tr_r = TensorMap(Cdata[:,:,1,:], codomain(C) ← domain(C)[2])
            Tr_u = TensorMap(O1data[:,:,:,n+1], codomain(O1)←domain(O1)[1])
            Tr_d = TensorMap(δcdata[:,2n+1,:,:], codomain(δc_a_in)[1]←domain(δc_a_in))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)

        elseif i == 3
            C_ul = TensorMap(O1data[n+1,:,:,n+1], codomain(O1)[2]←domain(O1)[1])
            C_ur = TensorMap(O4data[:,:,n+1,n+1], codomain(O4)[1] ⊗ codomain(O4)[2] ← nullspace)
            C_dr = TensorMap(δcdata[:,2n+1,n+1,:], codomain(δc_a_in)[1] ← domain(δc_a_in)[2])
            C_dl = TensorMap(δcdata[n+1,2n+1,:,:], nullspace ← domain(δc_a_in)[1] ⊗ domain(δc_a_in)[2])

            Tr_l = TensorMap(Bdata[1,:,:,:], codomain(B)[2] ← domain(B))
            Tr_r = TensorMap(Ddata[:,:,1,:], codomain(D) ← domain(D)[2])
            Tr_u = TensorMap(δcdata[:,:,:,2n+1], codomain(δc_a_in)←domain(δc_a_in)[1])
            Tr_d = TensorMap(O4data[:,n+1,:,:], codomain(O4)[1]←domain(O4))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)

        elseif i == 4
            C_ul = TensorMap(δcdata[n+1,:,:,2n+1], codomain(δc_a_in)[2]←domain(δc_a_in)[1])
            C_ur = TensorMap(δcdata[:,:,n+1,2n+1], codomain(δc_a_in)[1] ⊗ codomain(δc_a_in)[2] ← nullspace)
            C_dr = TensorMap(O1data[:,n+1,n+1,:], codomain(O1)[1] ← domain(O1)[2])
            C_dl = TensorMap(O4data[n+1,n+1,:,:], nullspace ← domain(O4)[1] ⊗ domain(O4)[2])

            Tr_l = TensorMap(Cdata[1,:,:,:], codomain(C)[2] ← domain(C))
            Tr_r = TensorMap(Adata[:,:,1,:], codomain(A) ← domain(A)[2])
            Tr_u = TensorMap(O4data[:,:,:,n+1], codomain(O4)←domain(O4)[1])
            Tr_d = TensorMap(δcdata[:,2n+1,:,:], codomain(δc_a_in)[1]←domain(δc_a_in))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)

        elseif i == 5
            C_ul = TensorMap(Ddata[1,:,:,n+1], codomain(D)[2]←domain(D)[1])
            C_ur = TensorMap(Bdata[:,:,1,n+1], codomain(B)[1] ⊗ codomain(B)[2] ← nullspace)
            C_dr = TensorMap(Cdata[:,n+1,1,:], codomain(C)[1] ← domain(C)[2])
            C_dl = TensorMap(Adata[1,n+1,:,:], nullspace ← domain(A)[1] ⊗ domain(A)[2])

            Tr_l = TensorMap(δcdata[n+1,:,:,:], codomain(δc_a_in)[2] ← domain(δc_a_in))
            Tr_r = TensorMap(δcdata[:,:,n+1,:], codomain(δc_a_in) ← domain(δc_a_in)[2])
            Tr_u = TensorMap(Adata[:,:,:,2n+1], codomain(A)←domain(A)[1])
            Tr_d = TensorMap(Bdata[:,2n+1,:,:], codomain(B)[1]←domain(B))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)

        elseif i == 6
            C_ul = TensorMap(Adata[1,:,:,2n+1], codomain(A)[2]←domain(A)[1])
            C_ur = TensorMap(Cdata[:,:,1,2n+1], codomain(C)[1] ⊗ codomain(C)[2] ← nullspace)
            C_dr = TensorMap(Ddata[:,2n+1,1,:], codomain(D)[1] ← domain(D)[2])
            C_dl = TensorMap(Bdata[1,2n+1,:,:], nullspace ← domain(B)[1] ⊗ domain(B)[2])

            Tr_l = TensorMap(O1data[n+1,:,:,:], codomain(O1)[2] ← domain(O1))
            Tr_r = TensorMap(O4data[:,:,n+1,:], codomain(O4) ← domain(O4)[2])
            Tr_u = TensorMap(Bdata[:,:,:,n+1], codomain(B)←domain(B)[1])
            Tr_d = TensorMap(Cdata[:,n+1,:,:], codomain(C)[1]←domain(C))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)

        elseif i == 7
            C_ul = TensorMap(Bdata[1,:,:,n+1], codomain(B)[2]←domain(B)[1])
            C_ur = TensorMap(Ddata[:,:,1,n+1], codomain(D)[1] ⊗ codomain(D)[2] ← nullspace)
            C_dr = TensorMap(Adata[:,n+1,1,:], codomain(A)[1] ← domain(A)[2])
            C_dl = TensorMap(Cdata[1,n+1,:,:], nullspace ← domain(C)[1] ⊗ domain(C)[2])

            Tr_l = TensorMap(δcdata[n+1,:,:,:], codomain(δc_a_in)[2] ← domain(δc_a_in))
            Tr_r = TensorMap(δcdata[:,:,n+1,:], codomain(δc_a_in) ← domain(δc_a_in)[2])
            Tr_u = TensorMap(Cdata[:,:,:,2n+1], codomain(C)←domain(C)[1])
            Tr_d = TensorMap(Ddata[:,2n+1,:,:], codomain(D)[1]←domain(D))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)
        
        elseif i == 8
            C_ul = TensorMap(Cdata[1,:,:,2n+1], codomain(C)[2]←domain(C)[1])
            C_ur = TensorMap(Adata[:,:,1,2n+1], codomain(A)[1] ⊗ codomain(A)[2] ← nullspace)
            C_dr = TensorMap(Bdata[:,2n+1,1,:], codomain(B)[1] ← domain(B)[2])
            C_dl = TensorMap(Ddata[1,2n+1,:,:], nullspace ← domain(D)[1] ⊗ domain(D)[2])

            Tr_l = TensorMap(O4data[n+1,:,:,:], codomain(O4)[2] ← domain(O4))
            Tr_r = TensorMap(O1data[:,:,n+1,:], codomain(O1) ← domain(O1)[2])
            Tr_u = TensorMap(Ddata[:,:,:,n+1], codomain(D)←domain(D)[1])
            Tr_d = TensorMap(Adata[:,n+1,:,:], codomain(A)[1]←domain(A))

            env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)
        end
        #=loc_in =    [A, B, C, D, 
        O1, δc_a_in, O4, δc_a_in,
        B, C, D, A,
        δc_a_in, O4, δc_a_in, O1,
        C, D, A, B,
        O4, δc_a_in, O1, δc_a_in,
        D, A, B, C,
        δc_a_in, O1, δc_a_in, O4]=#

        
        #=
        dim_loc_l = TensorKit.dim(codomain(loc[i])[1])
        dim_loc_d = TensorKit.dim(codomain(loc[i])[2])
        dim_loc_r = TensorKit.dim(domain(loc[i])[1])
        dim_loc_u = TensorKit.dim(domain(loc[i])[2])

        Tr_l = TensorMap(fill(1.0+1.0*im,dim_loc_l), Space_type^1 ← Space_type^dim_loc_l ⊗ Space_type^1)
        Tr_d = TensorMap(fill(1.0+1.0*im,dim_loc_d), Space_type^1 ← Space_type^1 ⊗ Space_type^dim_loc_d)
        Tr_r = TensorMap(fill(1.0+1.0*im,dim_loc_r), Space_type^dim_loc_r ⊗ Space_type^1 ← Space_type^1)
        Tr_u = TensorMap(fill(1.0+1.0*im,dim_loc_u), Space_type^1 ⊗ Space_type^dim_loc_u ← Space_type^1)
        =#
        #=dim_loc_l = TensorKit.dim(codomain(loc[i])[1])
        dim_loc_d = TensorKit.dim(codomain(loc[i])[2])
        dim_loc_r = TensorKit.dim(domain(loc[i])[1])
        dim_loc_u = TensorKit.dim(domain(loc[i])[2])
        
        space_loc_l = codomain(loc[i])[1]
        space_loc_d = codomain(loc[i])[2]
        space_loc_r = domain(loc[i])[1]
        space_loc_u = domain(loc[i])[2]
        @info "hello5 - seed $initialize_seed"

        rng = MersenneTwister(initialize_seed)


        if initialize_real == true
            @info "we initialize real & random"
            
            Tr_l = TensorMap(ComplexF64.(fill(1.0,dim_loc_l)), Space_type^1 ← space_loc_l ⊗ Space_type^1)
            Tr_d = TensorMap(ComplexF64.(fill(1.0,dim_loc_d)), Space_type^1 ← Space_type^1 ⊗ space_loc_d)
            Tr_r = TensorMap(ComplexF64.(fill(1.0,dim_loc_r)), space_loc_r ⊗ Space_type^1 ← Space_type^1)
            Tr_u = TensorMap(ComplexF64.(fill(1.0,dim_loc_u)), Space_type^1 ⊗ space_loc_u ← Space_type^1)
            #=
            Tr_l = TensorMap(ComplexF64.(randn(rng,dim_loc_l)), Space_type^1 ← space_loc_l ⊗ Space_type^1)
            Tr_d = TensorMap(ComplexF64.(randn(rng,dim_loc_d)), Space_type^1 ← Space_type^1 ⊗ space_loc_d)
            Tr_r = TensorMap(ComplexF64.(randn(rng,dim_loc_r)), space_loc_r ⊗ Space_type^1 ← Space_type^1)
            Tr_u = TensorMap(ComplexF64.(randn(rng,dim_loc_u)), Space_type^1 ⊗ space_loc_u ← Space_type^1)
            =#
        else

            Tr_l = TensorMap(randn(rng,ComplexF64,dim_loc_l), Space_type^1 ← space_loc_l ⊗ Space_type^1)
            Tr_d = TensorMap(randn(rng,ComplexF64,dim_loc_d), Space_type^1 ← Space_type^1 ⊗ space_loc_d)
            Tr_r = TensorMap(randn(rng,ComplexF64,dim_loc_r), space_loc_r ⊗ Space_type^1 ← Space_type^1)
            Tr_u = TensorMap(randn(rng,ComplexF64,dim_loc_u), Space_type^1 ⊗ space_loc_u ← Space_type^1)
        end


        env_arr[i] = envs(C_ul, C_dl, C_dr, C_ur, Tr_l, Tr_d, Tr_r, Tr_u)
        =#
    end
    env_arr[12] = deepcopy(env_arr[1])
    env_arr[19] = deepcopy(env_arr[1])
    env_arr[26] = deepcopy(env_arr[1])

    env_arr[9] = deepcopy(env_arr[2])
    env_arr[20] = deepcopy(env_arr[2])
    env_arr[27] = deepcopy(env_arr[2])

    env_arr[10] = deepcopy(env_arr[3])
    env_arr[17] = deepcopy(env_arr[3])
    env_arr[28] = deepcopy(env_arr[3])

    env_arr[11] = deepcopy(env_arr[4])
    env_arr[18] = deepcopy(env_arr[4])
    env_arr[25] = deepcopy(env_arr[4])

    env_arr[16] = deepcopy(env_arr[5])
    env_arr[23] = deepcopy(env_arr[5])
    env_arr[30] = deepcopy(env_arr[5])

    env_arr[14] = deepcopy(env_arr[7])
    env_arr[21] = deepcopy(env_arr[7])
    env_arr[32] = deepcopy(env_arr[7])

    env_arr[13] = deepcopy(env_arr[6])
    env_arr[24] = deepcopy(env_arr[6])
    env_arr[31] = deepcopy(env_arr[6])

    env_arr[15] = deepcopy(env_arr[8])
    env_arr[22] = deepcopy(env_arr[8])
    env_arr[29] = deepcopy(env_arr[8])
    return env_arr
end

function initialize_PEPS(Bond_loc, Dim_loc, Number_of_PEPS; Number_type = Float64, lattice = :square, identical = false, seed = 1236, data_type = :TensorMap, Space_type = ℂ)
    #=
    NOTE1: 
    The number of tensors refers to the number of coarse grained tensors that result on the square lattice!
    For example if we are working fundamentally with a model on a honeycomb lattice we thus need an array with 2*Number_of_PEPS tensors as input!
    
    NOTE2:
    "Dim_loc" can refer to the dimension of different spaces. If we work with the d.o.f. on the original lattice before coarse graining,
    it refers to the dimension of the local Hilbert space of that lattice.
    E.g. for a spin 1/2 model on the honeycomb lattice we would have Dim_loc = 2 . 
    However if we want/need to optimize directly on the coarse grained lattice it refers to the product of the local Hilbert spaces
    that are encompassed by the coarse grained tensor. 
    E.g. for a spin 1/2 model on the dice lattice- where we optimize on the coarse grained square lattice tensors- we would have Dim_loc = 8 = 2*2*2
    =#

    loc_in = []
    rng = MersenneTwister(seed)
    
    space_virt = Space_type^Bond_loc
    space_loc = Space_type^Dim_loc

    randn_with_seed = (tuple) -> randn(rng, Number_type, tuple)

    #generically the input will be TensorMaps---> the alternative will be removed at some point.
    if data_type == :TensorMap

        if lattice == :square

            for i in 1:Number_of_PEPS
                input_tensor = TensorMap(randn_with_seed, space_virt ⊗ space_virt  ←  space_virt ⊗ space_virt ⊗ (space_loc)')
                push!(loc_in, normalization_convention(input_tensor))
            end
            
        elseif lattice == :honeycomb

            if identical == false

                for i in 1:Number_of_PEPS

                    if i%2 == 1
                        #make a TensorMap out of the Arrays in loc_in
                        input_tensor = TensorMap(randn_with_seed, Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc)')
                        push!(loc_in, normalization_convention(input_tensor))

                    else

                        input_tensor = TensorMap(randn_with_seed, Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc)')
                        push!(loc_in, normalization_convention(input_tensor))

                    end

                end
            
            end

            if identical == true

                for i in 1:Number_of_PEPS 

                    loc_in_data = randn_with_seed((Bond_loc,Bond_loc,Bond_loc,Dim_loc))
                    loc_in_data_twisted = permutedims(loc_in_data, (2,3,1,4))
                    #make a TensorMap out of the Arrays in loc_in
                    input_tensor = TensorMap(loc_in_data, Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc)')
                    push!(loc_in, normalization_convention(input_tensor))

                    input_tensor = TensorMap(loc_in_data_twisted, Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc)')
                    push!(loc_in, normalization_convention(input_tensor))

                end
                
            end

        elseif lattice == :dice
            #=note here, that if we were to create three tensors (one for every site in the UC of the dice lattice),
            we run into the problem that the resulting coarse grained tensor has a bond dimenison of d^2 where d is the bond dimension of the 
            tensors of on the dice lattice. This only then leads to very few bond dimensions that are feasable.
            Hence we take as the input tensors, aka the parameters we optimize on a already coase grained - square lattice- tensor with
            an increased bond dimension.=#
            for i in 1:Number_of_PEPS
                input_tensor = TensorMap(randn_with_seed, space_virt ⊗ space_virt  ←  space_virt ⊗ space_virt ⊗ (space_loc)')
                push!(loc_in, normalization_convention(input_tensor))
            end
        elseif lattice == :kagome
            #=here a number of elegant mappings (e.g.: Kagome -> Triangle -> Honeycomb -> Square) are in principle possible.
            as a first approach however we just put a square lattice PEPS one the Kagome-lattice.=#
            for i in 1:Number_of_PEPS
                input_tensor = TensorMap(randn_with_seed, space_virt ⊗ space_virt  ←  space_virt ⊗ space_virt ⊗ (space_loc)')
                push!(loc_in, normalization_convention(input_tensor))
            end
        end
    else

        if lattice == :square

            for i in 1:Number_of_PEPS
                push!(loc_in, normalization_convention(randn(rng, Number_type, Bond_loc, Bond_loc, Bond_loc, Bond_loc, Dim_loc)))
            end
            
        elseif lattice == :honeycomb

            if identical == false
                for i in 1:2*Number_of_PEPS
                    push!(loc_in, normalization_convention(randn(rng, Number_type, Bond_loc, Bond_loc, Bond_loc, Dim_loc)))
                end
            end

            if identical == true
                for i in 1:Number_of_PEPS
                    push!(loc_in, normalization_convention(randn(rng, Number_type, Bond_loc, Bond_loc, Bond_loc, Dim_loc)))
                end
            end

        else
            println("you have not defined this lattice in the function 'initialize_PEPS'")
        end
    
    end
        
        
    return loc_in
end

function pattern_function(arr_in, Pattern_arr)
    buf = Buffer(arr_in, size(Pattern_arr))
    for i in 1:size(Pattern_arr)[1], j in 1:size(Pattern_arr)[2]
        buf[i,j] = arr_in[Pattern_arr[i,j]]
    end
    return copy(buf)
end

function normalization_convention(A; fix_phase = false)
    N = Zygote.@ignore norm(A,Inf)  # l_inf norm
    A1 = (A/N)

    # when the data is real, don't do the rotation! Can be done with keyword as well.
    if A isa TensorMap
        if spacetype(A) == CartesianSpace
            return A1
        end
    elseif isreal(A)
        return A1
    end

    # global phase fixing
    if fix_phase == false
        return A1
    elseif A isa TensorMap
        absmax = Zygote.@ignore A.data[findmax(abs.(A.data))[2]]
    else 
        absmax = Zygote.@ignore A[findmax(abs.(A))[2]]
    end

    N2 = exp(-angle(absmax) * im)
    A2 = (A1*N2)
    
    return A2
end

function normalization_convention_without_phase(A)
    N = Zygote.@ignore norm(A) 
    A1 = (A/N)
    return A1 , N
end


function convert_input(loc_in; lattice = :square, Space_type = ℝ, identical = false, inputisTM = false)    
    
    #=
    In this function we take the input as an array, which is needed for the optimizer, and convert it 
    into TensorMaps (WHICH WE NORMALIZE). These TensorMaps are then an array. This functions is different based on the lattice of the 
    model considered. One should add a specific procedure for all new lattices that we implement.
    =#
    
    if lattice == :square
        if inputisTM == true
            return loc_in
        end

        #infer Bond-dimension of the PEPS-tensor and local Hilbert-Space dimension from the local PEPS tensor
        Bond_loc = size(loc_in[1])[1]
        Dim_loc = size(loc_in[1])[5]

        #buf = Buffer([], TensorMap, size(loc_in)[1])
        buf = Buffer([], TensorMap, length(loc_in))

        #for i in 1:size(loc_in)[1]
        for i in 1:length(loc_in)   
            #make a TensorMap out of the Arrays in loc_in
            buf[i] = TensorMap( normalization_convention(loc_in[i]), Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc)')
        end

    elseif lattice == :triangular
        if inputisTM == true
            return loc_in
        end

    elseif lattice == :honeycomb
        #= when inputting tensors for the honeycomb lattice, one should supply them in the following form:
        every tensor in the unit cell of our square lattice, that we have mapped the honeycomb lattice onto corrsponds to two Tensors on the 
        honeycomb lattice. These two tensors are inequivalent (at least that is what I am thinking right now). For a unit cell with n-sites 
        one should thus supply an input array of 2n tensors of rank 4 - they each have 3 virtual indices and one physical index. This function than contracts 
        two adjacent tensors to form the coarse grained tensors that live on the square lattice that we perform the CTM-RG on.
        =#        

        if inputisTM == false
            #infer Bond-dimension of the honeycomb-PEPS-tensor and local Hilbert-Space dimension from the local honeycomb-PEPS tensor
            Bond_loc = size(loc_in[1])[1]
            Dim_loc_honey = size(loc_in[1])[4] 

            #=We can motivated by the bond structure of the Kitaev model enforce that the two honeycomb tensors that get coarse grained to a
            single square lattice tensor are related to each other by rotation... This has been implemented here, if one chooses the keyword "identical" to
            be true. =#
            if identical == true
                loc_in_TM = convert_loc_in_to_TM(loc_in, lattice = :honeycomb, Space_type = Space_type, identical = true)
            else
                loc_in_TM = convert_loc_in_to_TM(loc_in, lattice = :honeycomb, Space_type = Space_type)

            end
        else 
            Bond_loc = dim(space(loc_in[1])[1])
            Dim_loc_honey = dim(space(loc_in[1])[4]) 

            
            loc_in_TM = loc_in
            
        end

        Fuse = isomorphism(fuse(Space_type^Dim_loc_honey ⊗ Space_type^Dim_loc_honey), Space_type^Dim_loc_honey ⊗ Space_type^Dim_loc_honey)
        
        buf = Buffer([], TensorMap, Int(length(loc_in_TM)/2))
        
        for (i,j) in enumerate(1:2:length(loc_in_TM))
            
            @tensor sqr_lat_tens[(v1,v2);(v3,v4,p)] := loc_in_TM[j][v1,c,v4,p1] * loc_in_TM[j+1][c,v2,v3,p2] * Fuse[p,p1,p2]
            
            sqr_lat_tens_norm = normalization_convention(sqr_lat_tens)
            
            buf[i] = sqr_lat_tens
            
        end    

    elseif lattice == :dice

        if inputisTM == true
            return loc_in
        end
    elseif lattice ==:kagome

        if inputisTM == true
            return loc_in
        end
        
    else
        println("the lattice you have chosen is not yet included in the function 'convert_input'! You should add it!")
        
    end

    
    return copy(buf)
end

function convert_loc_in_to_TM(loc_in; lattice = :honeycomb, Space_type = ℝ, identical = false)
    if lattice == :honeycomb
        #infer Bond-dimension of the PEPS-tensor and local Hilbert-Space dimension from the local PEPS tensor
        Bond_loc = size(loc_in[1])[1]
        Dim_loc_honey = size(loc_in[1])[4]

        if identical == false

            buf = Buffer([], TensorMap, length(loc_in))

            for i in 1:length(loc_in)   

                if i%2 == 1
                    #make a TensorMap out of the Arrays in loc_in
                    buf[i] = TensorMap( normalization_convention( loc_in[i] ), Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc_honey)')
                
                else

                    buf[i] = TensorMap( normalization_convention( loc_in[i] ), Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc_honey)')
                end

            end
        
        end

        if identical == true

            buf = Buffer([], TensorMap, 2 * length(loc_in))

            for i in 1:length(loc_in)   

                turned_loc_in = permutedims(loc_in[i], (2,3,1,4))

                #make a TensorMap out of the Arrays in loc_in
                buf[2*i-1] = TensorMap( normalization_convention( loc_in[i] ), Space_type^Bond_loc ← Space_type^Bond_loc ⊗ Space_type^Bond_loc ⊗ (Space_type^Dim_loc_honey)')
                
                buf[2*i] = TensorMap( normalization_convention( turned_loc_in ), Space_type^Bond_loc ⊗ Space_type^Bond_loc ← Space_type^Bond_loc  ⊗ (Space_type^Dim_loc_honey)')
                
            end
            
        end


    end
    return copy(buf)
end

function test_elementwise_convergence(env_arr, env_arr_old, Pattern_arr, conv_crit)
    
    #this counts the number of environment-tensors that are NOT YET converged element wise.
    number = 0
    
   
    value_of_conv_max = 0
    value_of_conv_min = 0

    
    for i in 1:size(env_arr)[1]
        comp_ul = env_arr[i].ul - env_arr_old[i].ul 
        #this is converged element-wise if every element in the mask "comp_mask_..." is 1
        comp_mask_ul = abs.(comp_ul.data) .< conv_crit
        
        comp_dl = env_arr[i].dl - env_arr_old[i].dl 
        comp_mask_dl = abs.(comp_dl.data) .< conv_crit
        
        comp_dr = env_arr[i].dr - env_arr_old[i].dr 
        comp_mask_dr = abs.(comp_dr.data) .< conv_crit
        
        comp_ur = env_arr[i].ur - env_arr_old[i].ur 
        comp_mask_ur = abs.(comp_ur.data) .< conv_crit
        
        comp_u = env_arr[i].u - env_arr_old[i].u 
        comp_mask_u = abs.(comp_u.data) .< conv_crit
        
        comp_r = env_arr[i].r - env_arr_old[i].r 
        comp_mask_r = abs.(comp_r.data) .< conv_crit
        
        comp_d = env_arr[i].d - env_arr_old[i].d 
        comp_mask_d = abs.(comp_d.data) .< conv_crit
        
        comp_l = env_arr[i].l - env_arr_old[i].l 
        comp_mask_l = abs.(comp_l.data) .< conv_crit
        

        
        if minimum(comp_mask_ul) == 0 
            #=
            if the minimum of the mask "comp_mask_..." is 0 that means 
            that there are elements that are NOT YET converged element wise
            --> Thus add 1 to the number of these tensors
            =#
            number += 1
        
            #a = norm(comp_ul)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_dl) == 0 
            number += 1
            
            #a = norm(comp_dl)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_dr) == 0 
            number += 1
            
        
            #a = norm(comp_dr)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_ur) == 0 
            number += 1
        
            #a = norm(comp_ur)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_u) == 0 
            number += 1
        
            #a = norm(comp_u)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_r) == 0 
            number += 1
        
            #a = norm(comp_r)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_d) == 0 
            number += 1
        
            #a = norm(comp_d)
            #display("the norm difference is $a")
        end
        
        if minimum(comp_mask_l) == 0 
            number += 1
        
            #a = norm(comp_l)
            #display("the norm difference is $a")
        end
        
        
    end

    
    return number
end

function make_lin_array(A)
    return ArrayPartition(A...)
end

function get_symmetric_tensors(parameter_array, D, Dphys, sym_tensors)

    if sym_tensors == :R_PT
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_R_and_PT_sym(D,Dphys)
    elseif sym_tensors == :R
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_R_sym(D,Dphys)
    elseif sym_tensors == :PT
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :P
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_P_sym(D,Dphys)
    elseif sym_tensors == :P_minus
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_P_minus_sym(D,Dphys)
    elseif sym_tensors == :PT_phase
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :PT_test
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :PT_minus
        symmetric_tensors_imag, symmetric_tensors_real = Zygote.@ignore create_sym_tensors_PT_sym(D,Dphys)
    elseif sym_tensors == :PT_2
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_sym_2(D,Dphys)    
    elseif sym_tensors == :PT_I
        symmetric_tensors_real, symmetric_tensors_imag = Zygote.@ignore create_sym_tensors_PT_and_I_sym(D,Dphys)
    else
        println("you have not implemented the local symmetry that you ask of the tensor")
    end
    #display(symmetric_tensors_real ≈ real(symmetric_tensors_real))
    #display(symmetric_tensors_imag ≈ real(symmetric_tensors_imag))
    N_real = size(symmetric_tensors_real)[2]
    N_imag = size(symmetric_tensors_imag)[2]


    buf = Zygote.Buffer([], Array, length(parameter_array))

    #loc_d_in = []
    for i in 1:length(parameter_array)
        
        loc_d_in_vec = zeros(size(symmetric_tensors_real)[1])
        if sym_tensors == :PT_test
            for j in 1:N_real
                loc_d_in_vec += parameter_array[i][j] * symmetric_tensors_real[:,j]
            end
            for j in 1:N_imag
                loc_d_in_vec += parameter_array[i][N_real + j] * symmetric_tensors_imag[:,j]
            end
        else
            for j in 1:N_real
                loc_d_in_vec += parameter_array[i][j] * symmetric_tensors_real[:,j]
            end
            for j in 1:N_imag
                loc_d_in_vec += im * parameter_array[i][N_real + j] * symmetric_tensors_imag[:,j]
            end
        end

        if sym_tensors == :PT_phase
            phase = parameter_array[i][end] / norm(parameter_array[i][end])
            loc_d_in_vec_phase = phase .* loc_d_in_vec
            #display(loc_d_in_vec_phase)
        else 
            loc_d_in_vec_phase = loc_d_in_vec
        end

        buf[i] = reshape(loc_d_in_vec_phase, D, D, D, D, Dphys)
        
    end

        

    return copy(buf)
end

function create_sym_tensors_R_and_PT_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)

    #create the rotation operator
    @tensor R[δ,α,β,γ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape the rotation operator into a matrix
    R_mat = reshape(R, D^4*Dphys, D^4*Dphys);
    #calculate the eigenvalues/vectors of the Rotation matrix
    values_R, vectors_R = eigen(R_mat);
    #take only those eigenvectors that have eigenvalue 1 -> create the appropriate mask for this purpose
    mask_R = [values_R[i] ≈ 1 ? true : false for i in eachindex(values_R)];
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);
    #look at the reflection matrix in the subspace of vectors that span the eigenspace of R corresponding to eigenvalue 1
    Rx_mat_sub = vectors_R'[mask_R,:] * Rx_mat * vectors_R[:,mask_R];
    #look at the eigenvalues and eigenvectors in this subspace
    values_Rx, vectors_Rx = eigen(Rx_mat_sub);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]
    #now we can concatenate these to get the tensors that have eigenvalue +1 for rotation and reflextion
    symmetric_tensors_real = vectors_R[:,mask_R] * vectors_Rx[:,mask_Rx_plus]
    #now we can concatenate these to get the tensors that have eigenvalue +1 for rotation and -1 for reflextion
    symmetric_tensors_imag = vectors_R[:,mask_R] * vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_R_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)

    #create the rotation operator
    @tensor R[δ,α,β,γ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape the rotation operator into a matrix
    R_mat = reshape(R, D^4*Dphys, D^4*Dphys);
    #calculate the eigenvalues/vectors of the Rotation matrix
    values_R, vectors_R = eigen(R_mat);
    #take only those eigenvectors that have eigenvalue 1 -> create the appropriate mask for this purpose
    mask_R = [values_R[i] ≈ 1 ? true : false for i in eachindex(values_R)];

    symmetric_tensors_real = vectors_R[:,mask_R] 
    symmetric_tensors_imag = Matrix{Float64}(I,0,0)
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_PT_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus]
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_P_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus]
    symmetric_tensors_imag = []
    
    return symmetric_tensors_real, symmetric_tensors_imag
end


function create_sym_tensors_P_minus_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = []
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_PT_sym_2(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator
    @tensor Rx[γ,β,α,δ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus]
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end

function create_sym_tensors_PT_and_I_sym(D,Dphys)
    idy =  Matrix{Float64}(I,D,D)
    idy_phys =  Matrix{Float64}(I,Dphys,Dphys)
        
    #create the reflexion-operator on the x_axis
    @tensor Rx[α,δ,γ,β,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Rx_mat = reshape(Rx, D^4*Dphys, D^4*Dphys);

    values_Rx, vectors_Rx = eigen(Rx_mat);
    mask_Rx_plus = [values_Rx[i] ≈ 1 ? true : false for i in eachindex(values_Rx)]
    mask_Rx_minus = [values_Rx[i] ≈ -1 ? true : false for i in eachindex(values_Rx)]
    
    @tensor Ry[γ,β,α,δ,p1,a,b,c,d,p2] := idy[α,a]*idy[β,b]*idy[γ,c]*idy[δ,d]*idy_phys[p1,p2];
    #reshape into a matrix
    Ry_mat = reshape(Ry, D^4*Dphys, D^4*Dphys);
    
    Ry_mat_sub_plus = vectors_Rx'[mask_Rx_plus,:] * Ry_mat * vectors_Rx[:,mask_Rx_plus];
    Ry_mat_sub_minus = vectors_Rx'[mask_Rx_minus,:] * Ry_mat * vectors_Rx[:,mask_Rx_minus];

    #look at the eigenvalues and eigenvectors in this subspace
    values_Ry_plus, vectors_Ry_plus = eigen(Ry_mat_sub_plus);
    mask_Ry_plus = [values_Ry_plus[i] ≈ 1 ? true : false for i in eachindex(values_Ry_plus)]
    
    values_Ry_minus, vectors_Ry_minus = eigen(Ry_mat_sub_minus);
    mask_Ry_minus = [values_Ry_minus[i] ≈ -1 ? true : false for i in eachindex(values_Ry_minus)]

    symmetric_tensors_real = vectors_Rx[:,mask_Rx_plus] * vectors_Ry_plus[:,mask_Ry_plus]
    symmetric_tensors_imag = vectors_Rx[:,mask_Rx_minus] * vectors_Ry_minus[:,mask_Ry_minus]
    
    return symmetric_tensors_real, symmetric_tensors_imag
end