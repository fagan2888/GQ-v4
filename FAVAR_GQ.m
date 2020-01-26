%FAVAR CODE

clear all;
clc;
addpath('VAR_Toolbox_2.0_2015.08.12 v2/')
%% USER INPUT

% Select a subset of the 6 variables in Y

subset = 2;     % 1: DF1, DF2 (2 vars)
                % 2: FFR, DF1, DF2 (3 vars)
                % 3: Indus. Prod., Inflation CPI, S&P 500, 10Y TB, DF1, DF2 (6 vars)

%% DATA

% ----- LOAD DATA -----
% X : factors 
xdata = xlsread('data v3.xlsx','X 2007');
xdatabis = xlsread('data v3.xlsx','X 2007 bis'); %datastream donc déjà transformées
% Y 
ydata = xlsread('data v3.xlsx','Y 2007');

if subset == 1
    ydata = ydata(:,[1 2]);
elseif subset == 2
    ydata = ydata(:,[1 2 3]);
elseif subset == 3
    ydata = ydata(:,[1 2 3 4 5 6]);  
end

% Transformation
tcode = xlsread('data v3.xlsx','code');
tcode = tcode';
% Slow/Fast
slowcode = xlsread('data v3.xlsx','slow-fast');
% Dates
yearlab = xlsread('data v3.xlsx','year 2007');
% Names de X
[~,namesX1] = xlsread('data v3.xlsx','X 2007');
[~,namesX2] = xlsread('data v3.xlsx','X 2007 bis');
namesX1 = namesX1(1,2:end)';
namesX2 = namesX2(1,2:end)';
namesX = [namesX1;namesX2];

clearvars namesX1 namesX2 subset;

% ----- Description de Y (Graphiques) -----

% FED Rates
figure(1)
plot(yearlab,ydata(:,1),'Color','black','LineWidth',2);
grid on;
xlabel('Time');
ylabel('Federal Funds Rates');

% US Loans
figure(2)
plot(yearlab,ydata(:,2),'Color','black','LineWidth',2);
grid on;
xlabel('Time');
ylabel('Loans US default rates');

% US bonds
figure(3)
plot(yearlab,ydata(:,3),'Color','black','LineWidth',2);
grid on;
xlabel('Time');
ylabel('US corp bonds Default rates');

% ----- First test de stationnarité ADF ----- 
% Y
for i = 1:size(ydata,2)
    [hY1(i),pvalY1(i)] = adftest(ydata(:,i));
end 

% X 
for i = 1:size(xdata,2)
    [hX1(i),pvalX1(i)] = adftest(xdata(:,i));
end 

% Transformation pour stationnariser les séries 
% Pour X on prend les diff(log())
% Pour Y on prend diff

%pour X
for i_x = 1:size(xdata,2)   % Transform "X"
    %xtempraw(:,i_x) = diff(log(xdata(:,i_x)));
    xtempraw(:,i_x) = transx(xdata(:,i_x),tcode(i_x)); %#ok<AGROW>
end

%pour Y
for i_y = 1:size(ydata,2)
    ytempraw(:,i_y) = diff(ydata(:,i_y));
end

% Correct size after stationarity transformation
xdata = xtempraw(3:end,:);
xdatabis = xdatabis(2:end,:);
ydata = ytempraw(2:end,:);
yearlab = yearlab(3:end);

% Test de stationnarité ADF après transformations
% Y
for i = 1:size(ydata,2)
    [hY2(i),pvalY2(i)] = adftest(ydata(:,i));
end 

% X
for i = 1:size(xdata,2)
    [hX2(i),pvalX2(i)] = adftest(xdata(:,i));
end 

clearvars xtempraw ytempraw i_x i_y i;

% Define X et Y
X = [xdata xdatabis];   % Factors
Y = ydata; % Y 
namesXY = [namesX ; 'FFR'; 'US Loans DF' ; 'US Bonds DF']; % Noms

% Number of observations and dimension of X and Y
t0=size(Y,1); % T time series observations
t1=size(X,2); % N series from which we extract factors
t2=size(Y,2); % Taille de Y

clearvars xdata xdatabis ydata;

%------ Low frequency trends ------%

% Pour lisser les data
% Removing low frequency trends using a Bi-Weight trend
bw_bw = 100; % Bi-Weight Parameter for local demeaning
for is = 1:t2; 
    tmp = bw_trend(Y(:,is),bw_bw);
   	Y_trend(:,is)= tmp;
   	Y(:,is) = Y(:,is) - Y_trend(:,is); 
end;

for is = 1:t1; 
    tmp = bw_trend(X(:,is),bw_bw);
   	X_trend(:,is)= tmp;
   	X(:,is) = X(:,is) - X_trend(:,is); 
end;

clearvars tmp X_trend Y_trend is;

%------ Standardization ------%
ymean = nanmean(Y)';                                        % mean (ignoring NaN)
mult = sqrt((sum(~isnan(Y))-1)./sum(~isnan(Y)));     % num of non-NaN entries for each series
ystd = (nanstd(Y).*mult)';                                  % std (ignoring NaN)
Y = (Y - repmat(ymean',t0,1))./repmat(ystd',t0,1);  % standardized data

xmean = nanmean(X)';                                        
mult = sqrt((sum(~isnan(X))-1)./sum(~isnan(X)));     
xstd = (nanstd(X).*mult)';                                  
X = (X - repmat(xmean',t0,1))./repmat(xstd',t0,1); 

%% FAVAR


%------ Lag Selection ------%

[InformationCriterion, aicL, bicL, hqcL, fpeL, CVabs, CVrel, CVrelTr] = AicBicHqcFpe(Y, 12);
p=2;
% Ici ils appliquent la méthode de cross validation (voir les ref)
% Comment faisons-nous pour déterminer le nb de retards ? 
% PCA ?
% Pour l'instant j'ai choisi de laisser à 2 retards 

% ----- Number of factors ----- 

% Cross Validation : voir papier de recherche qui est en référence dans le
% mémoire des étudiants
constant = 1;
[numFt, FAVAR_CV_RMSE, CVabs, CVrel] = favarTuneFactorNum(X, Y(:, :), p, 12, constant);

% Onatski(2010)
numFtOnatski = onatski(X,12);

% Bai & Ng
% à voir 
BG = NbFactors(X);

% Autre méthode : ici ils font avec une interprétation économique. PCA ?
% Technique de Kaiser
% Technique du coude 
% Screeplot 
% Voir comment on peut interpréter le pca 

screeplot(X, 10);
% Nb factors = 6

% Les résultats du PCA
[coeff, score, latent, ~, explained] = pca(X);

%------ Factors estimation ------

F = factorsFAVAR(X,6);

% Même chose mais en mieux 
F = score(:,1:6);


%------ FAVAR estimation ------

% Data
DATA_FAVAR = [Y F];

% Parameters
constant = 1;              % include the constant
hor = 20;                  % horizon
iter = 500;                % number of iterations
conf = [90 60];            % level for confidence bands

% ----- Forecast error variance decomposition -----
%%
for i = 1:3
    
    dataset_fevd = [Y(:,i) F];
    [VAR, VARopt] = VARmodel(dataset_fevd,p,constant);
    [~, xlstext] = xlsread('data v3.xlsx','names 2');
    VARopt.vnames = xlstext;
    VARopt.nsteps = 48;
    
    % Compute FEVD
    [FEVD, VAR] = VARfevd(VAR,VARopt);
    
    % Horizon 48 mois pour plusieurs variables
    % Il faut encore faire le dataset pour ça 
    % Il faut add l'idiosynchratique + Total + les titres 
    for j = 1:7
        %fevd_40_months(i,j) = xlstext(i,j);
        fevd_48_months(i+1,j+1) = FEVD(40,j,1);
    end 
    
    % Table 13
    % Il faut ajouter idiosyncratique + Total + les titres 
    ok = 1;
    for k = 1:7
        %fevd_40_months(i,j) = xlstext(i,j);
        
        for l = 6:48:6
            fevd_months_DF(ok,k) = FEVD(l,k,1);
           ok = ok+ 1;
        end
    end 
    
    
    % Ensuite nous avons les graphiques pour le fevd mais pour l'instant je
    % laisse ça en commentaire pcq ça recommence à chaque fois, ça écrase
    % tout avant. A voir si on l'intègre. 
    
    % Compute error bands
    %[FEVDINF,FEVDSUP,FEVDMED] = VARfevdband(VAR,VARopt);
    
    % Plot
    %VARfevdplot(FEVDMED,VARopt,FEVDINF,FEVDSUP);
end


Test = DATA_FAVAR(:,3:end);
[VAR, VARopt] = VARmodel(Test,p,constant);
[~, xlstext] = xlsread('data v3.xlsx','names');
VARopt.vnames = xlstext;
% Compute FEVD
[FEVD, VAR] = VARfevd(VAR,VARopt);
% Compute error bands
[FEVDINF,FEVDSUP,FEVDMED] = VARfevdband(VAR,VARopt);
% Plot
VARfevdplot(FEVDMED,VARopt,FEVDINF,FEVDSUP);

load Data_JDanish
[VAR, VARopt] = VARmodel(DataTable.Series,2,constant);
VARopt.vnames = DataTable.Properties.VariableNames;
[FEVD, VAR] = VARfevd(VAR,VARopt);
% Compute error bands
[FEVDINF,FEVDSUP,FEVDMED] = VARfevdband(VAR,VARopt);
% Plot
VARfevdplot(FEVDMED,VARopt,FEVDINF,FEVDSUP);

% ---- IR -----
% Estimation of the FAVAR
Cf_FAVAR = ir(DATA_FAVAR,p,constant,hor,'cholimpact');
[ULbf_FAVAR,~,~,~,ULb2f_FAVAR] = bootbands(DATA_FAVAR,p,constant,iter,hor,conf,'cholimpact');

% Normalement, on obtient ici les IR en fonction d'un choc de la fed

% Plot resutls
% A MODIFIER
irfs_FAVAR = permute(Cf_FAVAR,[3 1 2]);
irfs_FAVAR = irfs_FAVAR(:,:,1);
conf_FAVAR1 = permute(ULbf_FAVAR,[3 1 2 4]);
conf_FAVAR_down1 = conf_FAVAR1(:,:,1,1);
conf_FAVAR_up1 = conf_FAVAR1(:,:,1,2);
conf_FAVAR2 = permute(ULb2f_FAVAR,[3 1 2 4]);
conf_FAVAR_down2 = conf_FAVAR2(:,:,1,1);
conf_FAVAR_up2 = conf_FAVAR2(:,:,1,2);
plot_titles = { 'Feds Funds Rate', 'US Loans Default' , 'US Corps Default'};

figure(7);
for ii = 1:3
subplot(2,2,ii); 
% IRFs of the SVAR
%plot(irfs_SVAR(:,ii),'red','LineWidth',1.5,'LineStyle','-');hold on; 
% IRFs of the FAVAR
plot(irfs_FAVAR(:,ii),'black','LineWidth',2);title(plot_titles(:,ii));  
%formatGraph('Quarter', '');
% Horizontal line at 0
hold on;plot(xlim, [0 0], 'color', 0.5*[1 1 1],'LineWidth',1.1);                     
% 90% confidence interval
hold on;plot(conf_FAVAR_down1(:,ii),'black','LineStyle',':');hold on;plot(conf_FAVAR_up1(:,ii),'black','LineStyle',':')
% 60% confidence interval
hold on;plot(conf_FAVAR_down2(:,ii),'black','LineStyle','-');hold on;plot(conf_FAVAR_up2(:,ii),'black','LineStyle','-')
end

% Sortir la contribution des variables à chaque composante principale 
% IRF avec mgoretti
% Slow/fast crtieria et comment les utiliser 
% Les tests avec les contraintes 
% Table 11 : Dynamic/static factors
% Table 12 : 48 Months Forecast Error Decompositions
% Results of Exclusions Restrictions tests 
