function ctmrg(loc_arr, χ, Pattern_arr; maxiter = 400, ϵ = 1e-7 , Space_type = ℝ, Projector_type = :half,
    svd_type = :envbond, conv_info = false, log_info = false, adjust_χ = false,
    trunc_check = true, reuse_envs = false, lognorm = 0, temp = false, ftol = false)
    """
    Params:
    - Projector_type: Projectors used to renormalize the CTM environment
    - lognorm: log of the normalization factor
    - ϵ: convergence threshold for norm of SV spectrum of the corner tensors b/w 2 successive CTM steps
    - ftol: convergence of energy
    - trunc_check: displays truncation error in compputing the projector via SVD

    Returns:
    - f: Free energy
    - env_arr: Boundary environment for loc_arr
    """

    
    if reuse_envs == false
        env_arr = initialize_multisite(loc_arr; Space_type = Space_type) 
    else
        env_arr = deepcopy(reuse_envs)
        display("We are reusing environments")
        
    end


    sv_arr_old = 0
    sv_arr_old2 = 0
    f = 10
    f_old = 0
    number = 10
    trunc_iters = zeros(Float64, maxiter)
    
    converging = true
    for i in 1:maxiter        
        if conv_info 
            println("CTM-RG iteration $i")
        end
        
        if trunc_check
            trunc_arr, env_arr = CTMRG_step(env_arr, loc_arr, χ, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_check = true)
            trunc_iters[i] = maximum(trunc_arr)
        else 
            env_arr = CTMRG_step(env_arr, loc_arr, χ, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end

        # dynamically increase the bond dimension
        if adjust_χ != false
            #trunc_sv_arr, env_arr = CTMRG_step(env_arr, loc_arr, χ, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_check = true)

            if maximum(trunc_arr) > adjust_χ[1] #if the largest SV cut during generation of the projectors is larger than the threshhold value increase chi
                while χ < adjust_χ[2] && maximum(trunc_arr) > adjust_χ[1]
                    @info "The environment bond dimension is being increased from $(χ) to $χ+10"
                    χ = χ +10
                    trunc_arr, env_arr = CTMRG_step(env_arr, loc_arr, χ, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_check = true)
                    test = maximum(trunc_arr)
                    @info "The largest truncation error is $(test)"
                end
                if χ ≥ adjust_χ[2]
                    @info "The maximal allowed environment bond dimension was reached"
                else
                    @info "The SV at which we truncate is now smaller than the threshhold!"
                end
            else
                @show maximum(trunc_arr)
            end
        end
        
        sv_arr = get_corner_sv(env_arr)

        if conv_info
            if trunc_check
                max_trunc = maximum(trunc_arr)
                min_trunc = minimum(trunc_arr)
                println("the largest truncation error for the projectors is $max_trunc")
                println("the smallest truncation error for the projectors is $min_trunc")
            end
        end
        
        if i>2
            if conv_info
                m = compare_sv(sv_arr, sv_arr_old)
                println("largest SV-difference of the C tensors is $m")
            end
        end
        
        if i>5
            if conv_info
                m2 = compare_sv(sv_arr, sv_arr_old2)
                println("largest SV-difference of the C tensors 2 steps appart is $m2")
            end
        end

        if i>5 && compare_sv(sv_arr, sv_arr_old) isa Number && compare_sv(sv_arr, sv_arr_old) < ϵ
            break
        end

        if i>15 && compare_sv(sv_arr, sv_arr_old2) isa Number && compare_sv(sv_arr, sv_arr_old2) < ϵ
            break
        end

        sv_arr_old2 = sv_arr_old
        sv_arr_old = sv_arr

        if i > 0
            f = free_energy_ff_plaquette_dice(loc_arr, env_arr, Pattern_arr, temp, lognorm)
            if log_info
                println("the current free energy is f=$f")
            end
        end
        
        if conv_info
            fdiff = abs.(real(f) - (real(f_old)))/abs(real(f))
            println("the relative difference of the real part of the free energies is $fdiff")
        end
        
        #if ftol != false
        #    if i>2 && abs.(real(f) - (real(f_old)))/abs(real(f)) < ftol && abs(imag(f))<5*10^-8
        #        break
        #    end
        #end
        f_old = f

        
        
        #=
        #this is just some convergence check based on the SV of the environment tensors. 
        #For the calculation of the gradient we check for element wise convergence.
        if  i>2 && maximum(abs.(S_test_array - S_test_array_old)) < ϵ && maximum(abs.(S_test_array2 - S_test_array2_old)) < ϵ 
            break
        end
        
        env_arr_old = env_arr #why am I doing this??
       
        #this just prints some convergence info in case it is wanted.
        if i>2
            if conv_info
                println("this shows the convergence of two environment tensors")
                println(maximum(abs.(S_test_array - S_test_array_old)))
                println(maximum(abs.(S_test_array2 - S_test_array2_old)))
            end
        end
        #@show S_test_array         
        S_test_array_old = S_test_array
        S_test_array2_old = S_test_array2
        =#
        
        if i == maxiter
            @info "CTMRG did not converge after maxiter = $(maxiter) steps."
            converging = false
        end
            
    end
     
    if trunc_check == true
        return f, trunc_iters, env_arr
    else
        return f, env_arr
    end

end



function free_energy_ff_plaquette_dice(loc_arr, env_arr, pattern_arr, temp, lognorm)

    β = 1 / temp
    Lx = size(pattern_arr)[1]
    Ly = size(pattern_arr)[2]
    
    f_pre = 0
    Z = 1
    for x in 1:Lx, y in 1:Ly

        i = pattern_arr[x,y]
        l1 = pattern_arr[x,mod(Ly+y-2,Ly)+1]
        l2 = pattern_arr[mod(Lx+x-2,Lx)+1,y]
        l3 = pattern_arr[mod(Lx+x-2,Lx)+1,mod(Ly+y-2,Ly)+1]

        @tensor order = (v2, v1, w3, u2, w2, u1, w1, y1, x1,x2,y2,x3) begin contr1[] := env_arr[i].ul[v1,w1] * env_arr[i].l[v2,w2,v1] * env_arr[i].dl[w3,v2]*
                                env_arr[i].u[w1,u1,x1] * loc_arr[i][w2,u2,x2,u1] * env_arr[i].d[w3,x3,u2] * 
                                env_arr[i].ur[x1,y1]*env_arr[i].r[x2,y2,y1]*env_arr[i].dr[x3,y2] end 

        @tensor begin contr3[] := env_arr[i].ul[v1,v2] * env_arr[l1].dl[v3,v1] *
                                    env_arr[i].u[v2,v,v2t] * env_arr[l1].d[v3,v3t,v] *
                                    env_arr[l1].dr[v3t,v4] * env_arr[i].ur[v2t,v4] end

        @tensor begin contr4[] := env_arr[i].ul[v1,v2] * env_arr[l2].ur[v2,v3] *
                                    env_arr[i].l[v1t,v,v1] * env_arr[l2].r[v,v3t,v3] *
                                    env_arr[i].dl[v4,v1t] * env_arr[l2].dr[v4,v3t] end

        @tensor begin contr2[] := env_arr[i].ul[v1,v2] * env_arr[l1].dl[v3,v1] * env_arr[l3].dr[v3,v4] * env_arr[l2].ur[v2,v4] end

        Z_xy = TensorKit.scalar(contr1) * TensorKit.scalar(contr2) / TensorKit.scalar(contr3) / TensorKit.scalar(contr4)
        f_pre += log(Complex(Z_xy))
    end
    f = -1/β *  (f_pre + lognorm)
    return f
end