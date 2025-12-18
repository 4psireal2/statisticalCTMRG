module statisticalCTMRG

using JLD2
using KrylovKit
using LinearAlgebra
using LsqFit
using Optim
using TensorKit
using Random
using Zygote
using Zygote: Buffer, @ignore


export ctmrg
export generate_local_tensor, generate_modified_local_tensor
export generate_local_tensor_Z2, generate_modified_local_tensor_Z2
export contract_tn_torus_3_2, contract_tn_torus_6_4, contract_tn_torus_9_6, contract_tn_torus_10_5
export magnetization, pi_flux_invar, Z2_flux_invar
export perform_data_collapse, scale_x, scale_y


include("CTMRG_moves.jl")
include("CTMRG_procedure.jl")
include("CTMRG_step.jl")
include("special_functions.jl")
include("aux_functions.jl")


end