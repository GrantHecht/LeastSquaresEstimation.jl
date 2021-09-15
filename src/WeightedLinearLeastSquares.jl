mutable struct WeightedLinearLeastSquares{BFVT}

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

    # Measurement Weights
    W::Symmetric{Float64, Matrix{Float64}} 

    # Independant variables
    t::Vector{Float64} 

end

function WeightedLinearLeastSquares(h::AbstractVector)
    WeightedLinearLeastSquares(h, Matrix{Float64}(undef, (0, 0)), Vector{Float64}(undef, length(h)), 
        Vector{Float64}(undef, 0), Symmetric(Matrix{Float64}(undef, (0,0))),Vector{Float64}(undef, 0))
end

function FeedMeasurementBatch!(wlls::WeightedLinearLeastSquares, ys::AbstractVector, W::AbstractMatrix, ts::AbstractVector)
    m = length(ys)
    n = length(wlls.h)
    if m != length(ts)
        throw(ArgumentError("Measurement vector and indepandant variable vector are not the same length."))
    end
    r, c = size(W)
    if r != c
        throw(ArgumentError("Weight matrix must be square!"))
    end

    # Set measurements and indepandant variable vectors
    wlls.y = ys
    wlls.W = Symmetric(W)
    wlls.t = ts

    # Size and fill H matrix
    H = zeros(m, n)
    @inbounds for row in 1:m
        for col in 1:n
            H[row, col] = wlls.h[col](wlls.t[row])
        end
    end
    wlls.H = H

    return nothing
end

function ComputeEstimate!(wlls::WeightedLinearLeastSquares)
    # Compute HᵀWH and factorize result
    HTWH = factorize(transpose(wlls.H)*wlls.W*wlls.H)

    # Compute estimate
    ldiv!(wlls.xhat, HTWH, transpose(wlls.H)*wlls.W*wlls.y)

    return nothing
end

function GetEstimate(wlls::WeightedLinearLeastSquares)
    return wlls.xhat
end

function GetMeasurementEstimate(wlls::WeightedLinearLeastSquares, t::Real)
    y = 0.0
    @inbounds for i in 1:length(wlls.h)
        y += wlls.h[i](t)*wlls.xhat[i]
    end
    return y
end
    

