function bn_icp = bai_ng(lsout)

%     lsout = [fac,lambda,tss,ssr,r2vec,nobs,nt,ns]

ssr = lsout.ssr; nobs = lsout.nobs; nt = lsout.nt;  % extract input
nfac = size(lsout.fac,2);                           % number of factors
nbar = nobs/nt;                                     % average obs per period
g = log(min([nbar;nt]))*(nbar+nt)/nobs;
bn_icp = log(ssr/nobs) + nfac*g;

end