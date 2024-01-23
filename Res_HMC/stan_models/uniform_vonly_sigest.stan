data {
    int nt;
    real dt;
    real delta_true;
    real eta_true;
    real J_true;
    real r_init;
    real v_init;
    real rlim[2];
    real vlim[2];
    vector[nt] rs;
    vector[nt] vs;
    vector[nt] I_input;
	int ds; 
}

transformed data {
    real std_prior=0.1;

    int nts=nt/ds;
    vector[nts] rs_decim;
    vector[nts] vs_decim;
    
    for (i in 1:nts){
         rs_decim[i]=rs[ds*(i-1)+1];
         vs_decim[i]=vs[ds*(i-1)+1];
         }

}

parameters {
    vector<lower=rlim[1], upper=rlim[2]>[nts] r;
    vector<lower=vlim[1], upper=vlim[2]>[nts] v;
    real<lower=0, upper=5> delta;
    real<lower=-10, upper=-3>  eta;
    real<lower=5, upper=20>  J;
    real<lower=0.0> eps; 
	real<lower=0.0> sig; 
}

transformed parameters {
}

model {
    vector[nts] rhat;
    vector[nts] vhat;

    /* priors*/
 

    sig ~ normal(0., 1); 
	eps ~ normal(0., 0.1); 
 
    
    /* integrate & predict */

    r[1]~ normal(r_init, std_prior);
    v[1]~ normal(v_init, std_prior);
    
        
    for (t in 1:(nts-1)) {
            real dr = 2*r[t]*v[t]+(delta/pi());
            real dv = v[t]*v[t] -(pi()*r[t])*(pi()*r[t]) + J*r[t]+ eta +I_input[t];
            r[t+1] ~ normal(r[t] + dt*dr, sig); 
            v[t+1] ~ normal(v[t] + dt*dv, sig); 
    }  

    rhat=(r);
    vhat=(v);

    target+=normal_lpdf(vs_decim| vhat, eps);

}


generated quantities {
    vector[nts] rhat_qqc;
    vector[nts] vhat_qqc;
    vector[nts] r_ppc;
    vector[nts] v_ppc;
    vector[nts] log_lik;

    rhat_qqc=(r);
    vhat_qqc=(v);

    
    for (i in 1:nts){
        r_ppc[i] = normal_rng(rhat_qqc[i], eps);
        v_ppc[i] = normal_rng(vhat_qqc[i], eps);
        log_lik[i] = normal_lpdf(vs[i]| vhat_qqc[i], eps);
      }
      
}
