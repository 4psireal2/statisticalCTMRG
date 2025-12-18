function generate_local_tensor(a,b)
    """
    For QI model
    """
    T = zeros(2,2,2,2)
    T[1,1,1,1] = 1
    T[2,2,2,2] = a^2 + b
    T[1,1,2,2] = a
    T[1,2,1,2] = a
    T[1,2,2,1] = a
    T[2,1,1,2] = a
    T[2,1,2,1] = a
    T[2,2,1,1] = a
    T_TM = TensorMap(T, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2);
    return T_TM
end


function generate_modified_local_tensor(a,b)
    """
    For QI model
    """
    M = zeros(2,2,2,2)
    M[1,1,1,1] = 1
    M[2,2,2,2] = -(a^2 + b)
    M[1,1,2,2] = a
    M[1,2,1,2] = a
    M[1,2,2,1] = a
    M[2,1,1,2] = -a
    M[2,1,2,1] = -a
    M[2,2,1,1] = -a
    M_TM = TensorMap(M, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2);
    return M_TM
end


function generate_local_tensor_Z2(a,b)
    """
    For QI model
    """
    V_Z2 = Z2Space(0 => 1, 1 => 1) 
    T = zeros(2,2,2,2)
    T[1,1,1,1] = 1
    T[2,2,2,2] = a^2 + b
    T[1,1,2,2] = a
    T[1,2,1,2] = a
    T[1,2,2,1] = a
    T[2,1,1,2] = a
    T[2,1,2,1] = a
    T[2,2,1,1] = a

    T_TM = TensorMap(T, V_Z2 ⊗ V_Z2 ← V_Z2 ⊗ V_Z2);
    return T_TM
end


function generate_modified_local_tensor_Z2(a,b)
    """
    For QI model
    """
    V_Z2 = Z2Space(0 => 1, 1 => 1) 
    M = zeros(2,2,2,2)
    M[1,1,1,1] = 1
    M[2,2,2,2] = -(a^2 + b)
    M[1,1,2,2] = a
    M[1,2,1,2] = a
    M[1,2,2,1] = a
    M[2,1,1,2] = -a
    M[2,1,2,1] = -a
    M[2,2,1,1] = -a
    M_TM = TensorMap(M,  V_Z2 ⊗ V_Z2 ← V_Z2 ⊗ V_Z2);
    return M_TM
end


function magnetization(loc_arr::TensorMap, mod_loc_arr::TensorMap, env_arr)

    @tensor order = (v2, v1, w3, u2, w2, u1, w1, y1, x1,x2,y2,x3) begin contr1[] := env_arr.ul[v1,w1] * env_arr.l[v2,w2,v1] * env_arr.dl[w3,v2]*
                                env_arr.u[w1,u1,x1] * mod_loc_arr[w2,u2,x2,u1] * env_arr.d[w3,x3,u2] * 
                                env_arr.ur[x1,y1]*env_arr.r[x2,y2,y1]*env_arr.dr[x3,y2] end 

    @tensor order = (v2, v1, w3, u2, w2, u1, w1, y1, x1,x2,y2,x3) begin contr2[] := env_arr.ul[v1,w1] * env_arr.l[v2,w2,v1] * env_arr.dl[w3,v2]*
                                env_arr.u[w1,u1,x1] * loc_arr[w2,u2,x2,u1] * env_arr.d[w3,x3,u2] * 
                                env_arr.ur[x1,y1]*env_arr.r[x2,y2,y1]*env_arr.dr[x3,y2] end 

    m = TensorKit.scalar(contr1) / TensorKit.scalar(contr2)

    return m
end


# Data collapse functions for finite-entanglement scaling inherited from [DOI: 10.1103/PhysRevLett.132.226502]

scale_x(x::Vector{Float64}, xc::Real, α::Real, D::Int64) = ((x .- xc) ./ xc) .* D^α


scale_y(y::Vector{Float64}, β::Real, D::Int64) = y .* D^(-β)


function linear_fit(x, y, x_eval)	
		Kxx = sum(x.^2)
		Ky = sum(y)
		Kx = sum(x)
		Kxy = sum(x.*y)
		K = length(x)
		Δ = K*Kxx - Kx^2
		b = (Kxx * Ky - Kx * Kxy)/Δ
		a = (K*Kxy - Kx*Ky)/Δ


		y_eval = a * x_eval + b
		Δy_eval = sqrt(
			(Kxx - 2*x_eval*Kx + x_eval^2 * K)/Δ
		)
		return y_eval, Δy_eval
end


function cost_function(ydata, Ydata, ΔYdata)
        """ 
        Ydata: interpolated y-scaled data (with `scale_y` and `linear_fit`)
        ΔYdata: interpolation error
        """
		S = 0
		N = 0

		for k in keys(ydata)
			y = ydata[k]
			Y = Ydata[k]
			ΔY = Ydata[k]
			N += length(Y)
			S += sum((y .- Y).^2 ./ ΔY.^2)
		end
		return S/N
end


function cost_function(α, β, Xc, xdata, ydata)
        """
        Xc: critical point
        Xdata: x-scaled data (with `scale_x`)
        """
		y_scaled = Dict{Int64, Vector{Float64}}()
		x_scaled = Dict{Int64, Vector{Float64}}()

		for (k,v) in ydata
			y_scaled[k] = scale_y(v, β, k)
			x_scaled[k] = scale_x(xdata, Xc, α, k)
		end
		# now we want to define the master function by getting correct points
		Y, ΔY = construct_master_curve(x_scaled, y_scaled)

		# now calculate the cost function
		return cost_function(y_scaled, Y, ΔY)
end


function construct_master_curve(x_scaled::Dict{Int64, Vector{Float64}}, y_scaled::Dict{Int64, Vector{Float64}})

		master_curve = Dict{Int64, Vector{Float64}}()
		master_curve_std = Dict{Int64, Vector{Float64}}()

		L_array = collect(keys(y_scaled))
		for (j,k) in enumerate(L_array)
			res_keys = deleteat!(copy(L_array), j)
			x_opt = x_scaled[k]		
			# now we compute the linear fit for the key 'k'
			master_curve[k]  = Vector{Float64}(undef, length(x_opt))
			master_curve_std[k] = Vector{Float64}(undef, length(x_opt))
			for (jx, x) in enumerate(x_opt)
				x_selected = Float64[]
				y_selected = Float64[]
				for rk in res_keys
					x_rk = x_scaled[rk]
					y_rk = y_scaled[rk]
					ilow = findlast(x_rk .≤ x)
					iup  = findfirst(x_rk .> x)
					if !isnothing(ilow) && !isnothing(iup)					
						push!(x_selected, x_rk[ilow])
						push!(x_selected, x_rk[iup])
						push!(y_selected, y_rk[ilow])
						push!(y_selected, y_rk[iup])
					end
				end
				if isempty(x_selected)
					Y, ΔY = y_scaled[k][j], 1
				else
					Y,ΔY = linear_fit(x_selected, y_selected, x)
				end
				master_curve[k][jx] = Y
				master_curve_std[k][jx] = ΔY
			end
		end
	return master_curve, master_curve_std
end


function error_α(α, β, Xc, xdata, ydata, η = 0.01)
		cost_min = cost_function(α, β,Xc, collect(xdata), ydata)
		cost_η   = cost_function(α*(1+η), β, Xc, collect(xdata), ydata)

		η*α*(2* log(cost_η/cost_min))^(-1/2)
end


function error_β(α, β, Xc, xdata, ydata, η = 0.01)
	cost_min = cost_function(α, β, Xc, collect(xdata), ydata)
	cost_η   = cost_function(α, β*(1+η), Xc, collect(xdata), ydata)

	η*β*(2*log(cost_η/cost_min))^(-1/2)
end


function perform_data_collapse(X, Y, bnddims, init_guess::Vector{Float64})

		# β₀ = rand()*0.001 + 1.0 # kappa
		# α₀ = rand()*0.001 + 1.0 # kappa/nu
		# _, id = findmax(Y[last(bnddims)])
		# Xc₀ = X[id]

        corr_data = Dict{Int64, Vector{Float64}}()
		for bnd in bnddims
			corr_data[bnd] = Y[bnd]
		end
        Y = corr_data
		
		res = optimize(params -> cost_function(params[1], params[2], params[3], collect(X), Y), init_guess, NelderMead())
		α, β, Xc = Optim.minimizer(res)

		Δα = error_α(α,β,Xc, X, Y)
		Δβ = error_β(α,β,Xc, X, Y)
		Δν = sqrt((Δβ/α)^2 +(Δα * β/α^2)^2)
		
		@info "κν = $α ± $(Δα)"
		@info "κ = $β ± $(Δβ)"
		@info "Xc = $Xc"
		ν = 1/α * β
		
		@info "ν = $ν ± $(Δν)"
		z = -sqrt(3) + sqrt(6/β + 3)
		Δz = 3/β^2/sqrt(6/β + 3) * Δβ
		c = z^2
		Δc = 2*z*Δz

		b_re = 6/c / (sqrt(12/c) + 1)
		@info "c - 1 = $(c-1) ± $(Δc)"
		@info "Cost = $(cost_function(α,β,Xc, collect(X), Y))"
		return ν, Δν, β, Δβ, Xc
end


function M_row(phi_y)
    M = zeros(ComplexF64, 2,2,2,2)
    Id = [1 0; 0 1];
    sigma_x = [0 1; 1 0];
    M[:, 1, :, 1] = exp(im *phi_y) * Id
    M[:, 2, :, 2] = sigma_x

    return TensorMap(M,  ℂ^2 ⊗  ℂ^2 ←  ℂ^2 ⊗  ℂ^2)
end

function M_col(phi_x)
    M = zeros(ComplexF64, 2,2,2,2)
    Id = [1 0; 0 1];
    sigma_x = [0 1; 1 0];
    M[1, :, 1, :] = exp(im *phi_x) * Id
    M[2, :, 2, :] = sigma_x

    return TensorMap(M,  ℂ^2 ⊗  ℂ^2 ←  ℂ^2 ⊗  ℂ^2)
end

function S()
    S = zeros(ComplexF64, 2,2,2,2)
    for i in 1:2
        for j in 1:2
            for k in 1:2
                for l in 1:2
                    Ph = (i != k) ? 1 : 0
                    Pv = (j != l) ? 1 : 0
                    
                    arf_sign = (-1)^(Ph + Pv + Ph * Pv)
                    S[i, j, k, l] = 0.25 * arf_sign
                end
            end
        end
    end

    return TensorMap(S,  ℂ^2 ⊗  ℂ^2 ←  ℂ^2 ⊗  ℂ^2)
end

function winding_number(r_arr)
    raw_phases = angle.(r_arr)
    deltas = diff(raw_phases)
    deltas .= mod2pi.(deltas .+ π) .- π
    total_change = sum(deltas)
    return round(Int, total_change / 2π)
end

function pi_flux_invar(a, b, phi_y, contract_func)
    phi_x_arr = range(0, step = 0.05, stop= 2*pi - 0.05)
    Z_phi_x_y = [contract_func(a, b, phi_x, phi_y) for phi_x in phi_x_arr]
    Z_phi_x_y_pi = [contract_func(a, b, phi_x, mod((phi_y+pi), 2*pi)) for phi_x in phi_x_arr]
    r_arr = Z_phi_x_y_pi ./ Z_phi_x_y

    return winding_number(r_arr)
end

function Z2_flux_invar(a, b, contract_func)
    Z_01 = contract_func(a, b, 0.0, pi)
    Z_10 = contract_func(a, b, pi, 0.0)
    Z_00 = contract_func(a, b, 0.0, 0.0)
    Z_11 = contract_func(a, b, pi, pi)
    sign_Z = sign((real(Z_01) * real(Z_10)) / (real(Z_00) * real(Z_11)))
    if sign_Z == -1
        return 1
    else
        return 0
    end
end

function contract_tn_torus_3_2(a, b, phi_x, phi_y)
    """
    Return Z(φ_x, φ_y) for a 3 x 2 network on a torus
    """
    local_t = generate_local_tensor(a, b)
    M_r = M_row(phi_y)
    M_c = M_col(phi_x)
    crossing_tensor = S()

    @tensor col_1[-1 -2 -3 -4; -5 -6 -7 -8] := local_t[-1, 1, -8, 4] * local_t[-2, 2, -7, 1] *
                                            local_t[-3, 3, -6, 2] * M_r[-4, 4, -5, 3];
    @tensor col_2[-1 -2 -3 -4; -5 -6 -7 -8] := col_1[-1, -2, -3, -4, 4, 3, 2, 1] *
                                            col_1[1, 2, 3, 4, -5, -6, -7, -8];
    @tensor col_end[-1 -2 -3 -4; -5 -6 -7 -8] := M_c[-1, 1, -8, 4] * M_c[-2, 2, -7, 1] *
                                            M_c[-3, 3, -6, 2] * crossing_tensor[-4, 4, -5, 3];
    @tensor Z = col_end[1, 2, 3, 4, 5, 6, 7, 8] * col_2[8, 7, 6, 5, 4, 3, 2, 1];
    return Z
end

function contract_tn_torus_6_4(a, b, phi_x, phi_y)
    """
    Return Z(φ_x, φ_y) for a 6 x 4 network on a torus
    """
    local_t = generate_local_tensor(a, b)
    M_r = M_row(phi_y)
    M_c = M_col(phi_x)
    crossing_tensor = S()

    @tensor row_1[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := M_c[1,-1,2,-10] * local_t[2,-2,3,-9] *
                                                    local_t[3,-3,4,-8] * local_t[4,-4,5,-7] * 
                                                    local_t[5,-5,1,-6];
    @tensor row_2[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := row_1[-1, -2, -3, -4, -5, 5, 4, 3, 2, 1] *
                                                    row_1[1, 2, 3, 4, 5, -6, -7, -8, -9, -10];
    @tensor row_3[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := row_1[-1, -2, -3, -4, -5, 5, 4, 3, 2, 1] *
                                                    row_2[1, 2, 3, 4, 5, -6, -7, -8, -9, -10];
    @tensor row_4[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := row_1[-1, -2, -3, -4, -5, 5, 4, 3, 2, 1] *
                                                    row_3[1, 2, 3, 4, 5, -6, -7, -8, -9, -10];
    @tensor row_5[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := row_1[-1, -2, -3, -4, -5, 5, 4, 3, 2, 1] *
                                                    row_4[1, 2, 3, 4, 5, -6, -7, -8, -9, -10];
    @tensor row_6[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := row_1[-1, -2, -3, -4, -5, 5, 4, 3, 2, 1] *
                                                    row_5[1, 2, 3, 4, 5, -6, -7, -8, -9, -10];
    @tensor row_end[-1 -2 -3 -4 -5; -6 -7 -8 -9 -10] := crossing_tensor[1,-1,2,-10] * M_r[2,-2,3,-9] *
                                                    M_r[3,-3,4,-8] * M_r[4,-4,5,-7] * 
                                                    M_r[5,-5,1,-6];
    @tensor Z = row_end[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 
                row_6[10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    return Z
end

function contract_tn_torus_10_5(a, b, phi_x, phi_y)
    """
    Return Z(φ_x, φ_y) for a 10 x 5 network on a torus
    """
    local_t = generate_local_tensor(a, b)
    M_r = M_row(phi_y)
    M_c = M_col(phi_x)
    crossing_tensor = S()

    @tensor row_1[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := M_c[1,-1,2,-12] * local_t[2,-2,3,-11] *
                                                            local_t[3,-3,4,-10] * local_t[4,-4,5,-9] * 
                                                            local_t[5,-5,6,-8] * local_t[6,-6,1,-7];
    @tensor row_2[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_1[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_3[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_2[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_4[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_3[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_5[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_4[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_6[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_5[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_7[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_6[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_8[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_7[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_9[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_8[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];
    @tensor row_10[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := row_1[-1, -2, -3, -4, -5, -6, 6, 5, 4, 3, 2, 1] *
                                                            row_9[1, 2, 3, 4, 5, 6, -7, -8, -9, -10, -11, -12];                                                            
    @tensor row_end[-1 -2 -3 -4 -5 -6; -7 -8 -9 -10 -11 -12] := crossing_tensor[1,-1,2,-12] * M_r[2,-2,3,-11] *
                                                            M_r[3,-3,4,-10] * M_r[4,-4,5,-9] * 
                                                            M_r[5,-5,6,-8] * M_r[6,-6,1,-7];
    @tensor Z = row_end[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 
                row_10[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    return Z
end

function contract_tn_torus_9_6(a, b, phi_x, phi_y)
    """
    Return Z(φ_x, φ_y) for a 9 x 6 network on a torus
    """
    local_t = generate_local_tensor(a, b)
    M_r = M_row(phi_y)
    M_c = M_col(phi_x)
    crossing_tensor = S()

    @tensor row_1[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := M_c[1,-1,2,-14] * local_t[2,-2,3,-13] *
                                                            local_t[3,-3,4,-12] * local_t[4,-4,5,-11] * 
                                                            local_t[5,-5,6,-10] * local_t[6,-6,7,-9] *
                                                            local_t[7,-7,1,-8];
    @tensor row_2[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_1[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
    @tensor row_3[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_2[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
    @tensor row_4[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_3[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
    @tensor row_5[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_4[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
    @tensor row_6[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_5[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
    @tensor row_7[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_6[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
    @tensor row_8[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_7[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
    @tensor row_9[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] := row_1[-1, -2, -3, -4, -5, -6, -7, 7, 6, 5, 4, 3, 2, 1] *
                                                            row_8[1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14];
                                                            
    @tensor row_end[-1 -2 -3 -4 -5 -6 -7; -8 -9 -10 -11 -12 -13 -14] :=crossing_tensor[1,-1,2,-14] * M_r[2,-2,3,-13] *
                                                            M_r[3,-3,4,-12] * M_r[4,-4,5,-11] * 
                                                            M_r[5,-5,6,-10] * M_r[6,-6,7,-9] *
                                                            M_r[7,-7,1,-8];
    @tensor Z = row_end[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 
                row_9[14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    return Z
end