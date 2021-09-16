using LeastSquaresEstimation
using Plots

function main()
    # x = [k₁, k₂, k₃, k₄, k₅, λ₁, λ₂, λ₃, ω₁, ω₂, ω₃, δ₁, δ₂, δ₃]ᵀ

    # Define basis function vector
    θ(x,t) = x[1]*exp(x[6]*t)*cos(x[9]*t + x[12]) + 
             x[2]*exp(x[7]*t)*cos(x[10]*t + x[13]) + 
             x[3]*exp(x[8]*t)*cos(x[11]*t + x[14]) + x[4]
    ϕ(x,t) = x[1]*exp(x[6]*t)*sin(x[9]*t + x[12]) + 
             x[2]*exp(x[7]*t)*sin(x[10]*t + x[13]) +
             x[3]*exp(x[8]*t)*sin(x[11]*t + x[14]) + x[5]
    f = [θ, ϕ]

    # Instantiate Linear Least Squares Estimator
    xhat0 = [0.5, 0.25, 0.125, 0.0, 0.0, -0.15, 0.06, 
            -0.03, 0.26, 0.55, 0.95, 0.01, 0.01, 0.01]
    nllse = NonLinearLeastSquares(f, xhat0)

    # Generate Measurements
    ft(x,t) = [f[1](x,t), f[2](x,t)] .+ 0.0002.*randn(2)
    m    = 202
    p    = length(f)
    xt   = [0.2, 0.1, 0.05, 0.0001, 0.0001, -0.1, -0.05, 
           -0.025, 0.25, 0.5, 1.0, 0.0, 0.0, 0.0]

    ml   = Int(m/p)
    ts = zeros(ml)
    ys = zeros(m)
    @inbounds for i in 1:ml
        if i != 1
            ts[i] = ts[i-1] + 1.0
        end
        idxs = ((i-1)*2 + 1):((i-1)*2 + p)
        ys[idxs] = ft(xt, ts[i])
    end

    # Feed estimator batch of measurements
    W = zeros(m,m)
    @inbounds for i in 1:m
        W[i,i] = 1.0
    end
    FeedMeasurementBatch!(nllse, ys, W, ts)

    # Compute estimate 
    ComputeEstimate!(nllse)

    # Compute estimated measurements and plot 
    yhat = zeros(m)
    @inbounds for i in 1:m
        yhat[i] = GetMeasurementEstimate(nllse, ts[i])
    end

    # Display 
    println("Estimated parameters:")
    display(GetEstimate(nllse))

    # Plot generated and estimated measurements 
    plot(ts, ys; label = "Simulated measurements")
    plot!(ts, yhat; label = "Estimated measurements")
end

main()