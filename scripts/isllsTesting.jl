using LeastSquaresEstimation
using Plots

function main()
    # Define basis function vector
    h = [t->sin(t), t->cos(t), t->t, t->cos(t)*sin(t), t->t^2]

    # Sequential Least Squares Estimator
    xhat0 = zeros(length(h))
    islse = SequentialLinearLeastSquares(h, xhat0, InformationForm())

    # Generate Measurements
    m    = 101
    m1   = 5
    σ1   = sqrt(0.0001)
    σ2   = sqrt(0.001)
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

    # Plot generated measurements
    p = plot(ts, ys; label = "Simulated measurements", legend = :outertopright)

    # Setup batch sizes
    batchSteps = [25, 50, 75, 101]

    # Update filter 
    for step in 1:length(batchSteps)

        # Compute batch size 
        if step == 1
            ms = batchSteps[step]
        else
            ms = batchSteps[step] - batchSteps[step - 1]
        end

        # Compute measurement and time vector idxs 
        if step == 1
            idxs = 1:batchSteps[1]
        else
            idxs = batchSteps[step-1]+1:batchSteps[step]
        end

        # Feed estimator batch of measurements
        W = zeros(ms,ms)
        W1 = 1.0 / σ1^2
        W2 = 1.0 / σ2^2
        @inbounds for i in 1:ms
            if idxs[i] <= m1
                W[i,i] = W1
            else
                W[i,i] = W2
            end
        end
        FeedMeasurementBatch!(islse, ys[idxs], W, ts[idxs])

        # Compute estimate 
        ComputeEstimate!(islse)

        # Compute estimated measurements and plot 
        yhat = zeros(m)
        @inbounds for i in 1:m
            yhat[i] = GetMeasurementEstimate(islse, ts[i])
        end

        # Display 
        println("Estimated parameters for update $step:")
        display(GetEstimate(islse))

        # Plot generated and estimated measurements 
        plot!(p, ts, yhat; label = "Estimated measurements #$step")
    end
    
    return p
end

main()