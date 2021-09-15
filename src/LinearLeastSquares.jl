
mutable struct LinearLeastSquares{BFVT}

    # Basis functions linear with respect to quantities to be estimated
    # h[i] ∀ i ∈ [0, n] should be a function which takes the independant
    # variable as an argument (i.e h[i](tₖ) where tₖ is the time at 
    # the kth epoch.
    h::BFVT

    # Basis function matrix
    H::Matrix{Float64}

    # Estimate vector
    xhat::Vector{Float64}
    
    # Measurements
    y::Vector{Float64}

    # Independant variables
    t::Vector{Float64} 

end

function LinearLeastSquares(h::AbstractVector)
    LinearLeastSquares(h, Matrix{Float64}(undef, (0, 0)), Vector{Float64}(undef, length(h)), 
        Vector{Float64}(undef, 0), Vector{Float64}(undef, 0))
end

function FeedMeasurementBatch!(lls::LinearLeastSquares, ys::AbstractVector, ts::AbstractVector)
    m = length(ys)
    n = length(lls.h)
    if m != length(ts)
        throw(ArgumentError("Measurement vector and indepandant variable vector are not the same length."))
    end

    # Set measurements and indepandant variable vectors
    lls.y = ys
    lls.t = ts

    # Size and fill H matrix
    H = zeros(m, n)
    @inbounds for row in 1:m
        for col in 1:n
            H[row, col] = lls.h[col](lls.t[row])
        end
    end
    lls.H = H

    return nothing
end

function ComputeEstimate!(lls::LinearLeastSquares)
    # xhat = (HᵀH)⁻¹Hᵀy

    # Compute HᵀH and factorize result
    HTH = factorize(transpose(lls.H)*lls.H)

    # Compute estimate
    ldiv!(lls.xhat, HTH, transpose(lls.H)*lls.y)

    return nothing
end

function GetEstimate(lls::LinearLeastSquares)
    return lls.xhat
end

function GetMeasurementEstimate(lls::LinearLeastSquares, t::Real)
    y = 0.0
    @inbounds for i in 1:length(lls.h)
        y += lls.h[i](t)*lls.xhat[i]
    end
    return y
end
    

