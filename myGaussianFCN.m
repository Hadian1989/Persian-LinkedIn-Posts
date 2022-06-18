function g = myGaussianFCN(x, u, s)
    g = 1/sqrt(2*pi*s^2)*exp(-(x -u)^2/(2*s^2));
end