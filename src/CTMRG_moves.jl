function create_projector_l(env, loc, χ::Int, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    """
    Params:
    - (k, l): indices of site for which the projector is created
    """
    
    m = mod(l-2,Ly)+1  # neighbor above
    n = mod(k,Lx) + 1
  
    
    if Projector_type == :half
    
        #options with half of the environemnt:
        @tensor order = (v1, w1, w2, u1) begin Ku[(i,j1);(α,β1)] := env[k,m].ul[w1,v1] * env[k,m].u[v1,w2,α] * 
                                                                    env[k,m].l[i,u1,w1] * loc[k,m][u1,j1,β1,w2] end

        @tensor order = (w1, v1, w2, u1) begin Kd[(i,j1);(α1,β)] := env[k,l].dl[v1,w1] * env[k,l].d[v1,β,w2] * 
                                                                    env[k,l].l[w1,u1,i] * loc[k,l][u1,w2,α1,j1] end 
    end

    if Projector_type == :full
        
        #options with full of the environemnt:
        @tensor order = (v1, w1, u1u, w2u, v3, w4, u3u, w3u, v2, u2u) begin Ku[(i,j1);(β1,α)] := env[k,m].ul[w1,v1] * env[k,m].u[v1,w2u,v2] * env[n,m].u[v2,w3u,v3] * env[n,m].ur[v3,w4] *
                                            env[k,m].l[i,u1u,w1] * loc[k,m][u1u,j1,u2u,w2u] *
                                            loc[n,m][u2u,β1,u3u,w3u] * env[n,m].r[u3u,α,w4] end

        @tensor order = (v3, w4, u3u, w3u, w1, v1, u1u, w2u, v2, u2u) begin Kd[(i,j1);(β1,α)] := env[k,l].dl[v1,w1] * env[k,l].d[v1,v2,w2u] * env[n,l].d[v2,v3,w3u] * env[n,l].dr[v3,w4] * 
                                            env[k,l].l[w1,u1u,i] * loc[k,l][u1u,w2u,u2u,j1] *
                                            loc[n,l][u2u,w3u,u3u,β1] * env[n,l].r[u3u,w4,α] end 
    end
    
    if Projector_type == :fullfishman
       
        @tensor order = (v1, w1, u1u, w2u, v3, w4, u3u, w3u, v2, u2u) begin Ku_pre[(i,j1);(β1,α)] := env[k,m].ul[w1,v1] * env[k,m].u[v1,w2u,v2] * env[n,m].u[v2,w3u,v3] * env[n,m].ur[v3,w4] *
                                            env[k,m].l[i,u1u,w1] * loc[k,m][u1u,j1,u2u,w2u] *
                                            loc[n,m][u2u,β1,u3u,w3u] * env[n,m].r[u3u,α,w4] end 



        @tensor order = (v3, w4, u3u, w3u, w1, v1, u1u, w2u, v2, u2u) begin Kd_pre[(i,j1);(β1,α)] := env[k,l].dl[v1,w1] * env[k,l].d[v1,v2,w2u] * env[n,l].d[v2,v3,w3u] * env[n,l].dr[v3,w4] * 
                                            env[k,l].l[w1,u1u,i] * loc[k,l][u1u,w2u,u2u,j1] *
                                            loc[n,l][u2u,w3u,u3u,β1] * env[n,l].r[u3u,w4,α] end 
        
        U_u, S_u, V_u_dag = wrapper_tsvd(Ku_pre, χ, Space_type = Space_type, svd_type = :accuracy)
        
        S_u_sqrt = sqrt(S_u)
        
        U_d, S_d, V_d_dag = wrapper_tsvd(Kd_pre, χ, Space_type = Space_type, svd_type = :accuracy)

        S_d_sqrt = sqrt(S_d)

        Ku = U_u * S_u_sqrt
        
        Kd = U_d * S_d_sqrt

        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[v1,v2,β]*Ku[v1,v2,α]
        
        U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type = svd_type);

        S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-6)

        @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] * Kd[j1,j2,β]
        @tensor Pdown[(i1,i2);(j,)] := Ku[i1,i2,α] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]
        P = (Pu = Pup, Pd = Pdown)
        
        if trunc_sv_out == true

            #this might need to be ignored for AD - but probably not
            #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
            trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2))) #take the reals to prevent the value becoming negative at machine precision

            return P, trunc_err 
        else
            return P
        end
        
    end


    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2);(α1,α2)] := Kd[v1,v2,β1,β2]*Ku[v1,v2,α1,α2]

    U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type = svd_type);

    S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-8)

    @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2] * Kd[j1,j2,β1,β2]
    @tensor Pdown[(i1,i2);(j,)] := Ku[i1,i2,α1,α2] * V_L_chi_d'[α1,α2,v2] * S_L_chi_inv_sqrt[v2,j]

    P = (Pu = Pup, Pd = Pdown)

    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2))) #take the reals to prevent the value becoming negative at machine precision
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2))) #take the reals to prevent the value becoming negative at machine precision
        #display("calculating the truncation error works.")
        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_l(env, loc, p_arr, k, l, Lx, Ly)
    
    m = mod(l,Ly) + 1
   
    @tensor  C_ul_tilde[(i,);(j,)] := p_arr[l].Pu[i,i1,i2] * env[k,l].ul[i1,v1] * env[k,l].u[v1,i2,j] 

    @tensor  begin Tr_l_tilde[(i,);(j1,k)] := p_arr[m].Pu[i,i1,i2] * 
        env[k,l].l[i1,v1,k1] *  loc[k,l][v1,i2,j1,k2] * p_arr[l].Pd[k1,k2,k] end 
                        
    @tensor C_dl_tilde[();(i1,j)] := env[k,l].dl[v1,j1] * env[k,l].d[v1,i1,j2] * p_arr[m].Pd[j1,j2,j] 

    #normalize the resulting tensors
    C_ul_new = normalization_convention(C_ul_tilde)
    Tr_l_new = normalization_convention(Tr_l_tilde)
    C_dl_new = normalization_convention(C_dl_tilde)
    
    return C_ul_new, Tr_l_new, C_dl_new
end

function update_donealready(arr, k, i, Pattern_arr)
    
    el = Pattern_arr[k,i]
    push!(arr, el)
    return arr
end

function projectors_l(env, loc, χ, k, donealready, Pattern_arr; Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k,Lx)+1
    
    proj = Array{NamedTuple}(undef,Ly)
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end
        
        #as in this version of the function we DO NOT want to output also the truncations from the projectors we choose the keyword "trunc_sv_out = false" (default)
        proj[i] = create_projector_l(env, loc, χ, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

    end
    
    return proj
end

#the function below has an additional argument in multiple dispach!
function projectors_l(env, loc, χ, k, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k,Lx)+1
   
    proj = Array{NamedTuple}(undef,Ly)
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end
        
        #as in this version of the function we want to output also the truncations from the projectors we choose the keyword "trunc_sv_out = true"
        proj[i], sv_trunc_ratio = create_projector_l(env, loc, χ, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)

        append!(trunc_sv_arr, sv_trunc_ratio)

    end


    return proj, trunc_sv_arr
end

function absorb_and_project_l(env, env_arr, loc, p_arr, k, donealready, Pattern_arr)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k,Lx)+1
   
    new_env_arr = Array{TensorMap}(undef,Ly,3)

    for i in 1:Ly

        
        if Pattern_arr[m,i] in donealready
            continue
        end

        C_ul_new, Tr_l_new, C_dl_new = absorb_and_project_tensors_l(env, loc, p_arr, k, i, Lx, Ly)
        new_env_arr[i,:] = [C_ul_new, Tr_l_new, C_dl_new]
    end

    for i in 1:Ly

        if Pattern_arr[m,i] in donealready
            continue
        end

        ind = Pattern_arr[m,i]
        env_arr[ind].ul = new_env_arr[i,1]
        env_arr[ind].l = new_env_arr[i,2]
        env_arr[ind].dl = new_env_arr[i,3]

        donealready = update_donealready(donealready, m, i, Pattern_arr)
    end

    return env_arr, donealready
end


function multi_left_move(env_arr, loc_arr, χ, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
    
    loc = pattern_function(loc_arr, Pattern_arr)
    env = pattern_function(env_arr, Pattern_arr)

    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    #the absorption happens column by column for all L_x columns of the unit cell
     
    donealready = []
    
    for k in 1:Lx  

        if trunc_sv_arr == false
            p_arr = projectors_l(env, loc, χ, k, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        else 
            #here I pass the list of truncations "trunc_sv_arr" as well and use multiple dispach.
            p_arr, trunc_sv_arr = projectors_l(env, loc, χ, k, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end

        env_arr, donealready = absorb_and_project_l(env, env_arr, loc, p_arr, k, donealready, Pattern_arr)
        
        #put the updated tensors into the environment array
        #build in a condition for k = Lx
        
        #env_arr, donealready = update_l(new_env, env_arr, k, Lx, Ly, donealready, Pattern_arr)
        
    end
    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
    
end


function create_projector_u(env, loc, χ, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    
    m = mod(k-2,Lx)+1 
    n = mod(l,Ly)+1
    
    
    if Projector_type == :half
        #options with half of the environemnt:
        @tensor order = (v1, w1, u1, w2) begin Ku[(i,j1);(β1,α)] := env[k,l].ur[w1,v1] * env[k,l].r[w2,α,v1] * env[k,l].u[i,u1,w1] *
                                                loc[k,l][j1,β1,w2,u1] end                                         

        @tensor order = (v1, w1, w2, u1) begin Kd[(i,j1);(β,α1)] := env[m,l].ul[v1,w1] * env[m,l].l[β,w2,v1] * env[m,l].u[w1,u1,i] * 
                                                loc[m,l][w2,α1,j1,u1] end 
    end
    
    if Projector_type == :full
        #options with full of the environemnt:
        @tensor order = (w4, v3, u3u, w3u, v1, w1, u1u, w2u, v2, u2u) begin Ku[(i,j1);(β1,α)] := env[k,l].ur[w1,v1] * env[k,l].r[w2u,v2,v1] * env[k,n].r[w3u,v3,v2] * env[k,n].dr[w4,v3] *
                                            env[k,l].u[i,u1u,w1] * loc[k,l][j1,u2u,w2u,u1u] *
                                            loc[k,n][β1,u3u,w3u,u2u] * env[k,n].d[α,w4,u3u] end

        @tensor order = (w1, v1, u1u, w2u, v3, w4, w3u, u3u, v2, u2u) begin Kd[(i,j1);(β1,α)] := env[m,l].ul[v1,w1] * env[m,l].l[v2,w2u,v1] * env[m,n].l[v3,w3u,v2] * env[m,n].dl[w4,v3] *
                                            env[m,l].u[w1,u1u,i] * loc[m,l][w2u,u2u,j1,u1u] *
                                            loc[m,n][w3u,u3u,β1,u2u] * env[m,n].d[w4,α,u3u] end

    end
    
    if Projector_type == :fullfishman
        @tensor order = (w4, v3, u3u, w3u, v1, w1, u1u, w2u, v2, u2u) begin Ku_pre[(i,j1);(β1,α)] := env[k,l].ur[w1,v1] * env[k,l].r[w2u,v2,v1] * env[k,n].r[w3u,v3,v2] * env[k,n].dr[w4,v3] *
                                            env[k,l].u[i,u1u,w1] * loc[k,l][j1,u2u,w2u,u1u] *
                                            loc[k,n][β1,u3u,w3u,u2u] * env[k,n].d[α,w4,u3u] end 

        @tensor order = (w1, v1, u1u, w2u, v3, w4, w3u, u3u, v2, u2u) begin Kd_pre[(i,j1);(β1,α)] := env[m,l].ul[v1,w1] * env[m,l].l[v2,w2u,v1] * env[m,n].l[v3,w3u,v2] * env[m,n].dl[w4,v3] *
                                            env[m,l].u[w1,u1u,i] * loc[m,l][w2u,u2u,j1,u1u] *
                                            loc[m,n][w3u,u3u,β1,u2u] * env[m,n].d[w4,α,u3u] end 

        
        U_u, S_u, V_u_dag = wrapper_tsvd(Ku_pre, χ, Space_type = Space_type, svd_type = :accuracy)
        
        S_u_sqrt = sqrt(S_u)

        U_d, S_d, V_d_dag = wrapper_tsvd(Kd_pre, χ, Space_type = Space_type, svd_type = :accuracy)
        
        S_d_sqrt = sqrt(S_d)

        Ku = U_u * S_u_sqrt
        
        Kd = U_d * S_d_sqrt
        
        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[v1,v2,β]*Ku[v1,v2,α]
        
        U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type = svd_type);

        S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-6)

        @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] * Kd[j1,j2,β]
        @tensor Pdown[(i1,i2);(j,)] := Ku[i1,i2,α] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]

        P = (Pu = Pup, Pd = Pdown)
    
        if trunc_sv_out == true

            #this might need to be ignored for AD - but probably not
            #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
            #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
            trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

            return P, trunc_err 
        else
            return P
        end
        
    end
    
    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2);(α1,α2)] := Kd[v1,v2,β1,β2]*Ku[v1,v2,α1,α2]

    U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type= svd_type);

    S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-8)


    @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2] * Kd[j1,j2,β1,β2]
    @tensor Pdown[(i1,i2);(j,)] := Ku[i1,i2,α1,α2] * V_L_chi_d'[α1,α2,v2] * S_L_chi_inv_sqrt[v2,j]
    
    P = (Pu = Pup, Pd = Pdown)
    
    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_u(env, loc, p_arr, k, l, Lx, Ly)
    
    m = mod(k,Lx)+1
 
    
    #step1_u: define absorption into the up direction 
    @tensor C_ul_tilde[(i);(j)] := env[k,l].l[i,j2,v1] * env[k,l].ul[v1,j1] * p_arr[k].Pd[j1,j2,j]  

    @tensor begin Tr_u_tilde[(i,j1);(k)] := p_arr[k].Pu[i,i1,i2] *
                                                loc[k,l][i2,j1,k2,v1]*env[k,l].u[i1,v1,k1] * 
                                                p_arr[m].Pd[k1,k2,k] end 

    @tensor C_ur_tilde[(i,j);()] := p_arr[m].Pu[i,i1,i2] * env[k,l].r[i2,j,v1] * env[k,l].ur[i1,v1] 
    
    C_ul_new = normalization_convention(C_ul_tilde)
    Tr_u_new = normalization_convention(Tr_u_tilde)
    C_ur_new = normalization_convention(C_ur_tilde)

    return C_ul_new, Tr_u_new, C_ur_new
end


function projectors_u(env, loc, χ, l, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(l,Ly)+1
    
    proj = Array{NamedTuple}(undef,Lx)
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        proj[i] = create_projector_u(env, loc, χ, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

    end
    return proj
end

function projectors_u(env, loc, χ, l, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    
    m = mod(l,Ly) + 1
    
    proj = Array{NamedTuple}(undef,Lx)
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        proj[i], sv_trunc_ratio = create_projector_u(env, loc, χ, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)
        append!(trunc_sv_arr, sv_trunc_ratio)

    end
    return proj, trunc_sv_arr
end

function absorb_and_project_u(env, env_arr, loc, p_arr, l, donealready, Pattern_arr)
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    
    m = mod(l, Ly) + 1
   
    new_env_arr = Array{TensorMap}(undef,3,Lx)

    for i in 1:Lx

        if Pattern_arr[i,m] in donealready
            continue
        end

        C_ul_new, Tr_u_new, C_ur_new = absorb_and_project_tensors_u(env, loc, p_arr, i, l, Lx, Ly)
        new_env_arr[:,i] = [C_ul_new, Tr_u_new, C_ur_new]
    end

    for i in 1:Lx 
        if Pattern_arr[i,m] in donealready
            continue
        end

        ind = Pattern_arr[i,m]
        env_arr[ind].ul = new_env_arr[1,i]
        env_arr[ind].u = new_env_arr[2,i]
        env_arr[ind].ur = new_env_arr[3,i]

        donealready = update_donealready(donealready, i, m, Pattern_arr)
    end

    return env_arr, donealready
end


function multi_up_move(env_arr, loc_arr, χ, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
    
    #the absorption happens row by row for all L_y rows of the unit cell
    loc = pattern_function(loc_arr, Pattern_arr)
    env = pattern_function(env_arr, Pattern_arr)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    donealready = []
    
    for l in 1:Ly  
        #create all projectors in this row
        if trunc_sv_arr == false
            p_arr = projectors_u(env, loc, χ, l, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        else
            p_arr, trunc_sv_arr = projectors_u(env, loc, χ, l, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end
        #perform absorption and projection for this row
        
        env_arr, donealready = absorb_and_project_u(env, env_arr, loc, p_arr, l, donealready, Pattern_arr)
    end

    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
end


function create_projector_r(env, loc, χ, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    
    m = mod(l-2,Ly) + 1
    n = mod(k-2,Lx) + 1
    
    
    if Projector_type == :half

        #options with half of the environment:
        @tensor order = (w4, v3, w3, u3) begin Ku[(α, β1);(k1,l)] := env[k,m].u[α,w3,v3] * env[k,m].ur[v3,w4] *
                                                    loc[k,m][β1,k1,u3,w3] * env[k,m].r[u3,l,w4] end

        @tensor order = (w4, v3, w3, u3) begin Kd[(α1, β);(k1,l)] := env[k,l].d[β,v3,w3] * env[k,l].dr[v3,w4] * 
                                                    loc[k,l][α1,w3,u3,k1] * env[k,l].r[u3,w4,l] end 

    end
    
    if Projector_type == :full

        #options with full environment:
        @tensor order = (v1, w1, u1u, w2u, v3, w4, u3u, w3u, v2, u2u) begin Ku[(α, β1);(k1,l)] := env[n,m].ul[w1,v1] * env[n,m].u[v1,w2u,v2] * env[k,m].u[v2,w3u,v3] * env[k,m].ur[v3,w4] * 
                                                env[n,m].l[α,u1u,w1] * loc[n,m][u1u,β1,u2u,w2u] *
                                                loc[k,m][u2u,k1,u3u,w3u] * env[k,m].r[u3u,l,w4] end

        @tensor order = (v3, w4, u3u, w3u, w1, v1, u1u, w2u, v2, u2u) begin Kd[(α,β1);(k1,l)] := env[n,l].dl[v1,w1] * env[n,l].d[v1,v2,w2u] * env[k,l].d[v2,v3,w3u] * env[k,l].dr[v3,w4] *
                                                env[n,l].l[w1,u1u,α] * loc[n,l][u1u,w2u,u2u,β1] *
                                                loc[k,l][u2u,w3u,u3u,k1] * env[k,l].r[u3u,w4,l] end
    end
    
    if Projector_type == :fullfishman

        @tensor order = (v1, w1, u1u, w2u, v3, w4, u3u, w3u, v2, u2u) begin Ku_pre[(α, β1);(k1,l)] := env[n,m].ul[w1,v1] * env[n,m].u[v1,w2u,v2] * env[k,m].u[v2,w3u,v3] * env[k,m].ur[v3,w4] * 
                                                env[n,m].l[α,u1u,w1] * loc[n,m][u1u,β1,u2u,w2u] *
                                                loc[k,m][u2u,k1,u3u,w3u] * env[k,m].r[u3u,l,w4] end 

        @tensor order = (v3, w4, u3u, w3u, w1, v1, u1u, w2u, v2, u2u) begin Kd_pre[(α,β1);(k1,l)] := env[n,l].dl[v1,w1] * env[n,l].d[v1,v2,w2u] * env[k,l].d[v2,v3,w3u] * env[k,l].dr[v3,w4] *
                                                env[n,l].l[w1,u1u,α] * loc[n,l][u1u,w2u,u2u,β1] *
                                                loc[k,l][u2u,w3u,u3u,k1] * env[k,l].r[u3u,w4,l] end 
        
        U_u, S_u, V_u_dag = wrapper_tsvd(Ku_pre, χ, Space_type = Space_type, svd_type = :accuracy)

        S_u_sqrt = sqrt(S_u)

        U_d, S_d, V_d_dag = wrapper_tsvd(Kd_pre, χ, Space_type = Space_type, svd_type = :accuracy)

        S_d_sqrt = sqrt(S_d)

        Ku = S_u_sqrt * V_u_dag
        
        Kd = S_d_sqrt * V_d_dag
        
        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[β,v1,v2]*Ku[α,v1,v2]
        
        U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type = svd_type);


        S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-6)

        @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] *Kd[β,j1,j2]
        @tensor Pdown[(i1,i2);(j,)] := Ku[α,i1,i2] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]

        P = (Pu = Pup, Pd = Pdown)

        if trunc_sv_out == true

            #this might need to be ignored for AD - but probably not
            #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
            #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
            trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

            return P, trunc_err 
        else
            return P
        end
        
    end
    
    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2);(α1,α2)] := Kd[β1,β2,v1,v2]*Ku[α1,α2,v1,v2]

    U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type = svd_type);

    S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-8)
    
    @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2] *Kd[β1,β2,j1,j2]
    @tensor Pdown[(i1,i2);(j,)] := Ku[α1,α2,i1,i2] * V_L_chi_d'[α1,α2,v2] * S_L_chi_inv_sqrt[v2,j]
    
    P = (Pu = Pup, Pd = Pdown)

    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_r(env, loc, p_arr, k, l, Lx, Ly)
    
    m = mod(l, Ly) + 1
   
        
    #step1_r: define absorption into the right direction
    @tensor C_ur_tilde[(i,j);()] := env[k,l].u[i,j1,v1] * env[k,l].ur[v1,j2] * p_arr[l].Pu[j,j1,j2] 

    @tensor begin Tr_r_tilde[(i1,j);(k,)] := p_arr[l].Pd[k1,k2,k] *
                                                loc[k,l][i1,j1,v2,k1] * env[k,l].r[v2,j2,k2] * 
                                                p_arr[m].Pu[j,j1,j2] end 

    @tensor C_dr_tilde[(i,);(j,)] := p_arr[m].Pd[j1,j2,j] * env[k,l].d[i,v1,j1] * env[k,l].dr[v1,j2] 

    #normalize the resulting tensors
    C_ur_new = normalization_convention(C_ur_tilde)
    Tr_r_new = normalization_convention(Tr_r_tilde)
    C_dr_new = normalization_convention(C_dr_tilde)

    return C_ur_new, Tr_r_new, C_dr_new
        
end

function projectors_r(env, loc, χ, k, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(k-2,Lx)+1

    proj = Array{NamedTuple}(undef,Ly)
    
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end

        proj[i] = create_projector_r(env, loc, χ, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
    end
    return proj
    
end

function projectors_r(env, loc, χ, k, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    
    m = mod(k-2,Lx)+1
    
    proj = Array{NamedTuple}(undef,Ly)
    
    for i in 1:Ly    
        
        if Pattern_arr[m,i] in donealready
            continue
        end

        proj[i], trunc_sv_ratio = create_projector_r(env, loc, χ, k, i, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)
        append!(trunc_sv_arr, trunc_sv_ratio)
    end
    return proj, trunc_sv_arr
end

function absorb_and_project_r(env, env_arr, loc, p_arr, k, donealready, Pattern_arr)
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    
    m = mod(k-2, Lx) + 1
    
    new_env_arr = Array{TensorMap}(undef,Ly,3)

    for i in 1:Ly
        if Pattern_arr[m,i] in donealready
            continue
        end

        C_ur_new, Tr_r_new, C_dr_new = absorb_and_project_tensors_r(env, loc, p_arr, k, i, Lx, Ly)
        new_env_arr[i,:] = [C_ur_new, Tr_r_new, C_dr_new]
    end

    for i in 1:Ly 
        if Pattern_arr[m,i] in donealready
            continue
        end

        ind = Pattern_arr[m,i]
        env_arr[ind].ur = new_env_arr[i,1]
        env_arr[ind].r = new_env_arr[i,2]
        env_arr[ind].dr = new_env_arr[i,3]

        donealready = update_donealready(donealready, m, i, Pattern_arr)
    end
    return env_arr, donealready
end


function multi_right_move(env_arr, loc_arr, χ, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
    
    #the absorption happens row by row for all L_y rows of the unit cell
    loc = pattern_function(loc_arr, Pattern_arr)
    env = pattern_function(env_arr, Pattern_arr)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    donealready = []
    
    for k in reverse(1:Lx)  
        

        if trunc_sv_arr == false
            p_arr = projectors_r(env, loc, χ, k, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)

        else
            p_arr, trunc_sv_arr = projectors_r(env, loc, χ, k, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end
        
        env_arr, donealready = absorb_and_project_r(env, env_arr, loc, p_arr, k, donealready, Pattern_arr)
        
        #put the updated tensors into the environment dictionary
        
        #env_arr, donealready = update_r(new_env, env_arr, k, Lx, Ly, donealready, Pattern_arr)
    end
    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
    
end


function create_projector_d(env, loc, χ, k, l, Lx, Ly; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_out = false)
    
    m = mod(k-2, Lx) +1
    n = mod(l-2, Ly) + 1
   
    
    if Projector_type == :half

        #options with half of the environment:
       

        @tensor order = (w4, v3, u3, w3) begin Ku[(β1, α);(k1,l)] := env[k,l].r[w3,v3,α] * env[k,l].dr[w4,v3] * 
                                                    loc[k,l][k1,u3,w3,β1] * env[k,l].d[l,w4,u3] end 

        @tensor order = (w4, v3, u3, w3) begin Kd[(β, α1);(k1,l)] := env[m,l].l[v3,w3,β] * env[m,l].dl[w4,v3] * 
                                                    loc[m,l][w3,u3,k1,α1] * env[m,l].d[w4,l,u3] end 
        
    end
    
    if Projector_type == :full

        #options with the full environment:
        @tensor order = (w4, v3, u3u, w3u, v1, w1, u1u, w2u, v2, u2u) begin Ku[(α, β1);(k1,l)] := env[k,n].ur[w1,v1] * env[k,n].r[w2u,v2,v1] * env[k,l].r[w3u,v3,v2] * env[k,l].dr[w4,v3] *
                                        env[k,n].u[α,u1u,w1] * loc[k,n][β1,u2u,w2u,u1u] *
                                        loc[k,l][k1,u3u,w3u,u2u] * env[k,l].d[l,w4,u3u] end

        @tensor order = (w1, v1, u1u, w2u, v3, w4, w3u, u3u, v2, u2u) begin Kd[(α, β1);(k1,l)] := env[m,n].ul[v1,w1] * env[m,n].l[v2,w2u,v1] * env[m,l].l[v3,w3u,v2] * env[m,l].dl[w4,v3] * 
                                        env[m,n].u[w1,u1u,α] * loc[m,n][w2u,u2u,β1,u1u] *
                                        loc[m,l][w3u,u3u,k1,u2u] * env[m,l].d[w4,l,u3u] end
    end
    
    if Projector_type == :fullfishman
    
        @tensor order = (w4, v3, u3u, w3u, v1, w1, u1u, w2u, v2, u2u) begin Ku_pre[(α, β1);(k1,l)] := env[k,n].ur[w1,v1] * env[k,n].r[w2u,v2,v1] * env[k,l].r[w3u,v3,v2] * env[k,l].dr[w4,v3] *
                                        env[k,n].u[α,u1u,w1] * loc[k,n][β1,u2u,w2u,u1u] *
                                        loc[k,l][k1,u3u,w3u,u2u] * env[k,l].d[l,w4,u3u] end 
        
        @tensor order = (w1, v1, u1u, w2u, v3, w4, w3u, u3u, v2, u2u) begin Kd_pre[(α, β1);(k1,l)] := env[m,n].ul[v1,w1] * env[m,n].l[v2,w2u,v1] * env[m,l].l[v3,w3u,v2] * env[m,l].dl[w4,v3] * 
                                        env[m,n].u[w1,u1u,α] * loc[m,n][w2u,u2u,β1,u1u] *
                                        loc[m,l][w3u,u3u,k1,u2u] * env[m,l].d[w4,l,u3u] end 
        
        U_u, S_u, V_u_dag = wrapper_tsvd(Ku_pre, χ, Space_type = Space_type, svd_type = :accuracy)

        S_u_sqrt = sqrt(S_u)

        U_d, S_d, V_d_dag = wrapper_tsvd(Kd_pre, χ, Space_type = Space_type, svd_type = :accuracy)

        #S_d_sqrt = sqrtTM(S_d)
        S_d_sqrt = sqrt(S_d)

        Ku = S_u_sqrt * V_u_dag
        
        Kd = S_d_sqrt * V_d_dag
        
        Ku = normalization_convention(Ku)
        Kd = normalization_convention(Kd)

        @tensor L[(β);(α)] := Kd[β,v1,v2]*Ku[α,v1,v2]
        
        U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type = svd_type);


        S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-6)

        @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β] *Kd[β,j1,j2]
        @tensor Pdown[(i1,i2);(j,)] := Ku[α,i1,i2] * V_L_chi_d'[α,v2] * S_L_chi_inv_sqrt[v2,j]

        P = (Pu = Pup, Pd = Pdown)

        if trunc_sv_out == true 

            #this might need to be ignored for AD - but probably not
            #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
            #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
            trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

            return P, trunc_err 
        else
            return P
        end
        
    end
    
    Ku = normalization_convention(Ku)
    Kd = normalization_convention(Kd)
    
    @tensor L[(β1,β2);(α1,α2)] := Kd[β1,β2,v1,v2] * Ku[α1,α2,v1,v2]

    U_L_chi, S_L_chi, V_L_chi_d = wrapper_tsvd(L, χ, Space_type = Space_type, svd_type = svd_type);
    S_L_chi_inv_sqrt = pinv(sqrt(S_L_chi); rtol = 10^-8)

    @tensor Pup[(i,);(j1,j2)] := S_L_chi_inv_sqrt[i,v1] *  U_L_chi'[v1,β1,β2] *Kd[β1,β2,j1,j2]
    @tensor Pdown[(i1,i2);(j,)] := Ku[α1,α2,i1,i2] * V_L_chi_d'[α1,α2,v2] * S_L_chi_inv_sqrt[v2,j]
    
    P = (Pu = Pup, Pd = Pdown)
    
    if trunc_sv_out == true

        #this might need to be ignored for AD - but probably not
        #sv_trunc_ratio = diag(S_L_chi.data)[end] / diag(S_L_chi.data)[1]
        #trunc_err = sqrt(abs(1-(sum(diag(S_L_chi.data).^2)/norm(L)^2)))
        trunc_err = sqrt(abs(1-(norm(S_L_chi)^2/norm(L)^2)))

        return P, trunc_err 
    else
        return P
    end
end

function absorb_and_project_tensors_d(env, loc, p_arr, k, l, Lx, Ly)
    
    m = mod(k,Lx)+1
    if k == Lx
        m = 1
    else
        m = k + 1
    end
    
    #step1_d: absorption
    @tensor C_dl_tilde[();(i,j)] := env[k,l].dl[i2,v1] * env[k,l].l[v1,i1,j] * p_arr[k].Pd[i1,i2,i] 


    @tensor begin Tr_d_tilde[(i);(j,k1)] := p_arr[k].Pu[i,i1,i2] * 
                                            env[k,l].d[i2,j2,v2] * loc[k,l][i1,v2,j1,k1] * 
                                            p_arr[m].Pd[j1,j2,j] end 

    @tensor C_dr_tilde[(i,);(j,)] := p_arr[m].Pu[i,i1,i2] * env[k,l].dr[i2,v1] * env[k,l].r[i1,v1,j] 
   
    #normalize the resulting tensors
    C_dl_new = normalization_convention(C_dl_tilde)
    Tr_d_new = normalization_convention(Tr_d_tilde)
    C_dr_new = normalization_convention(C_dr_tilde)

    return C_dl_new, Tr_d_new, C_dr_new
end

function projectors_d(env, loc, χ, l, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    m = mod(l-2,Ly) + 1
   
    proj = Array{NamedTuple}(undef,Lx)
    
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        proj[i] = create_projector_d(env, loc, χ, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
    end
    return proj
   
end

function projectors_d(env, loc, χ, l, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = :GKL)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    
    m = mod(l-2,Ly) + 1
    
    proj = Array{NamedTuple}(undef,Lx)
    
    for i in 1:Lx    
        
        if Pattern_arr[i,m] in donealready
            continue
        end

        proj[i], sv_trunc_ratio = create_projector_d(env, loc, χ, i, l, Lx, Ly; Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type, trunc_sv_out = true)
        append!(trunc_sv_arr, sv_trunc_ratio)
    end
    return proj, trunc_sv_arr
end

function absorb_and_project_d(env, env_arr, loc, p_arr, l, donealready, Pattern_arr)
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]
    
    m = mod(l - 2, Ly) +1
  
    
    new_env_arr = Array{TensorMap}(undef,3,Lx)

    for i in 1:Lx

        if Pattern_arr[i,m] in donealready
            continue
        end

        C_dl_new, Tr_d_new, C_dr_new = absorb_and_project_tensors_d(env, loc, p_arr, i, l, Lx, Ly)
        new_env_arr[:,i] = [C_dl_new, Tr_d_new, C_dr_new]
     
    end
    for i in 1:Lx
        if Pattern_arr[i,m] in donealready
            continue
        end

        ind = Pattern_arr[i,m]
        env_arr[ind].dl = new_env_arr[1,i]
        env_arr[ind].d = new_env_arr[2,i]
        env_arr[ind].dr = new_env_arr[3,i]

        donealready = update_donealready(donealready, i, m, Pattern_arr)
    end

    return env_arr, donealready
end


function multi_down_move(env_arr, loc_arr, χ, Pattern_arr; Space_type = ℝ, Projector_type = Projector_type, svd_type = :GKL, trunc_sv_arr = false)
    
    #the absorption happens row by row for all L_y rows of the unit cell
    loc = pattern_function(loc_arr, Pattern_arr)
    env = pattern_function(env_arr, Pattern_arr)
    
    Lx = size(Pattern_arr)[1]
    Ly = size(Pattern_arr)[2]

    donealready = []
    
    for l in reverse(1:Ly)   

        #create projectors for the row l
        if trunc_sv_arr == false
            p_arr = projectors_d(env, loc, χ, l, donealready, Pattern_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        else
            p_arr, trunc_sv_arr = projectors_d(env, loc, χ, l, donealready, Pattern_arr, trunc_sv_arr;  Space_type = Space_type, Projector_type = Projector_type, svd_type = svd_type)
        end
        #absorb and project the tensors in the row l

        env_arr, donealready = absorb_and_project_d(env, env_arr, loc, p_arr, l, donealready, Pattern_arr)
        
     
    end
    if trunc_sv_arr == false
        return env_arr
    else
        return env_arr, trunc_sv_arr
    end
end