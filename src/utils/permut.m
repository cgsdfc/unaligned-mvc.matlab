function P = permut(c)
P = zeros(size(c,1));
for k = 1:length(c)
    P(c(k),k) = 1;
end
