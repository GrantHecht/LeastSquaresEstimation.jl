using LeastSquaresEstimation
using Plots

function main()
    # Define basis function vector
    h = [t->sin(t), t->cos(t), t->t, t->cos(t)*sin(t), t->t^2]

    # Instantiate Linear Least Squares Estimator
    wllse = WeightedLinearLeastSquares(h)

    # Generate Measurements
    m    = 101
    m1   = 5
    σ1   = sqrt(0.0001)
    σ2   = sqrt(0.01)
    y(t,σ) = 0.3*sin(t) + 0.5*cos(t) + 0.1*t + σ*randn()

    ts = zeros(m)
    ys = zeros(m)
    @inbounds for i in 1:m
        if i != 1
            ts[i] = ts[i-1] + 0.1
        end
        if i <= m1
            ys[i] = y(ts[i], σ1)
        else
            ys[i] = y(ts[i], σ2)
        end
    end

    # Feed estimator batch of measurements
    W = zeros(m,m)
    W1 = 1.0 / σ1^2
    W2 = 1.0 / σ2^2
    @inbounds for i in 1:m
        if i <= m1
            W[i,i] = W1
        else
            W[i,i] = W2
        end
    end
    FeedMeasurementBatch!(wllse, ys, W, ts)

    # Compute estimate 
    ComputeEstimate!(wllse)

    # Compute estimated measurements and plot 
    yhat = zeros(m)
    @inbounds for i in 1:m
        yhat[i] = GetMeasurementEstimate(wllse, ts[i])
    end

    # Display 
    println("Estimated parameters:")
    display(GetEstimate(wllse))

    # Plot generated and estimated measurements 
    plot(ts, ys; label = "Simulated measurements")
    plot!(ts, yhat; label = "Estimated measurements")
end

main()