%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ECON 634 Macro II
%%% Problem Set 7
%%% The Particle Filter
%%% Yaobin Wang
%%% 12/07/2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 1
% State:      S_t = X_t
% Obervables: Y_t = (A_t; B_t)
% Shocks:     W_t = (epsilon_t-1; epsilon_t-2; epsilon_t)
%             V_t = (v^A_t; v^B_t)
% Parameters: Theta = (rho_1; rho_2; phi_1; phi_2; beta; sigma; sigma_A; sigma_B)
% Transition Equation:
% S_t = g(S_t-1, S_t-2, W_t; Theta)
% X_t = rho_1*X_{t-1}+rho_2*X_{t-2}+phi_1*epsilon_{t-1}+phi*epsilon_{t-1}+epsilon_t
% Observation Equation:
% Y_t = h(St, Vt; Theta)
% (A_t; B_t) = (exp(X_t+v^A_t); beta*X^2_t+v^B_t)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 2 and 3
clear; close all; clc;
load('data.mat');
N = 1000; % number of particles
T = size(data,1); % length of time series

% priors:
prior.rho1 = @(x) unifpdf(x,-0.5,0.5);
prior.rho2 = @(x) unifpdf(x,-0.5,0.5);
prior.phi1 = @(x) unifpdf(x,-0.5,0.5);
prior.phi2 = @(x) unifpdf(x,-0.5,0.5);
prior.beta = @(x) unifpdf(x,4,7);
prior.sigma = @(x) lognpdf(x,-0.5,1);
prior.sigma_A = @(x) lognpdf(x,-0.5,1);
prior.sigma_B = @(x) lognpdf(x,-0.5,1);
prior.all = @(p) log(prior.rho1(p(1))) + log(prior.rho2(p(2)))+...
    log(prior.phi1(p(3))) + log(prior.phi2(p(4))) + ...
    log(prior.beta(p(5))) + log(prior.sigma(p(6))) + ...
    log(prior.sigma_A(p(7))) + log(prior.sigma_B(p(8)));

% proposals according to random walk with parameter sd's:
prop_sig.rho1 = 0.05;
prop_sig.rho2 = 0.05;
prop_sig.phi1 = 0.05;
prop_sig.phi2 = 0.05;
prop_sig.beta = 0.05;
prop_sig.sigma = 0.05;
prop_sig.sigma_A = 0.05;
prop_sig.sigma_B = 0.05;
prop_sig.all = [prop_sig.rho1 prop_sig.rho2 prop_sig.phi1 prop_sig.phi2 ...
    prop_sig.beta prop_sig.sigma prop_sig.sigma_A prop_sig.sigma_B];

% initial values for parameters
init_params = [0.4 0.2 0.4 -0.3 5.4 1.45 0.26 1];

% length of sample
M = 5000;
acc = zeros(M,1);

llhs = zeros(M,1);
parameters = zeros(M,8);
parameters(1,:) = init_params;

% evaluate model with initial parameters
log_prior = prior.all(parameters(1,:));
llh = PS7_YW_llh(parameters(1,:), data, N, T);
llhs(1) = log_prior + llh;

% sample
rng(0)
proposal_chance = log(rand(M,1));
prop_step = randn(M,8);
tic;
for m = 2:M
    % proposal draw:
    prop_param = parameters(m-1,:) + prop_step(m,:) .* prop_sig.all;
    
    % evaluate prior and model with proposal parameters:
    prop_prior = prior.all(prop_param);
    if prop_prior > -Inf % theoretically admissible proposal
        prop_llh = PS7_YW_llh(prop_param, data, N, T);
        llhs(m) = prop_prior + prop_llh;
        if llhs(m) - llhs(m-1) > proposal_chance(m)
            accept = 1;
        else
            accept = 0;
        end
    else % reject proposal since disallowed by prior
        accept = 0;
    end
    
    % update parameters (or not)
    if accept
        parameters(m,:) = prop_param;
        acc(m) = 1;
    else
        parameters(m,:) = parameters(m-1,:);
        llhs(m) = llhs(m-1);
    end
    
    waitbar(m/M)
end
toc
acc_rt=(sum(acc)/length(acc))*100;
disp(['The acceptance rate is ', num2str(acc_rt),'%']);

% Plot posteriors
label = ["\rho_1","\rho_2","\phi_1","\phi_2","\beta","\sigma","\sigma_A","\sigma_B"];
figure;
for i = 1:8
    subplot(2,4,i), hold on
    histfit(parameters(:,i),50,'kernel')
    ylabel('Density')
    title(label(:,i))
end