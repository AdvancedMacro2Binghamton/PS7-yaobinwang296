function [LLH] = PS7_YW_llh(params, data, N, T)
p.rho1 = params(1);
p.rho2 = params(2);
p.phi1 = params(3);
p.phi2 = params(4);
p.beta = params(5);
p.sigma = params(6);
p.sigma_A = params(7);
p.sigma_B = params(8);

T = min(T, length(data));

% simulate long-run transition
rng(0);
t=5000;
x=zeros(t+3,1);
epsilon=p.sigma * randn(t+3,1);

for t = 3:t+3
    x(t) = p.rho1*x(t-1) + p.rho2*x(t-2) + p.phi1*epsilon(t-1) + ...
        p.phi2*epsilon(t-2) + epsilon(t);
end

particle = zeros(T, N , 6);
llhs = zeros(T,1);

init_sample = randsample(t,N);
particle(1,:,1)=x(init_sample+2);
particle(1,:,2)=x(init_sample+1);
particle(1,:,3)=x(init_sample);
particle(1,:,4)=epsilon(init_sample+2);
particle(1,:,5)=epsilon(init_sample+1);
particle(1,:,6)=epsilon(init_sample);

llhs(1) = log( mean( exp( ...
        log( normpdf(log(data(1,1)), particle(1,:,1), p.sigma_A) ) + ...
        log( normpdf(data(1,2), p.beta*particle(1,:,1).^2 , p.sigma_B) )...
        ) ) );

% predict, filter, update particles and collect the likelihood 
    %%% Prediction:
for t = 2:T
    particle(t,:,1) = p.rho1*particle(t-1,:,2) + ...
        p.rho2*particle(t-1,:,3) + p.phi1*particle(t-1,:,5) + ...
        p.phi2*particle(t-1,:,6) + p.sigma*randn(1,N);
    particle(t,:,2) = particle(t-1,:,1);
    particle(t,:,3) = particle(t-1,:,2);
    particle(t,:,4) = p.sigma * randn(1,N);
    particle(t,:,5) = particle(t-1,:,6);
    particle(t,:,6) = particle(t-1,:,5);
  
    %%% Filtering:
    llh = log( normpdf(log(data(t,1)), particle(t,:,1), p.sigma_A) ) + ...
        log( normpdf(data(t,2), p.beta*particle(t,:,1).^2 , p.sigma_B) );
    lh = exp(llh);
    
    weight = exp( llh - log( sum(lh) ) );
    if sum(lh)==0
        weight(:) = 1 / length(weight);
    end
    % store the log(mean likelihood)
    llhs(t) = log(mean(lh));
    
    %%% Sampling:
    particle(t,:,1) = datasample(particle(t,:,1), N, 'Weights', weight);
end

LLH = sum(llhs);