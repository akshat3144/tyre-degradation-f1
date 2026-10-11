/* Hierarchical race model: their state-space structure, our pooling.
 *
 * WHAT THIS CHANGES ABOUT THEIR MODEL
 *
 * Their best model (full_race_1driver_and_fuel_base_t.stan) tracks ONE driver.
 * Its degradation term is a single scalar slope `v`, with `v ~ normal(.05,.1)`
 * and no compound index -- so the rate is estimated from roughly sixty laps of
 * one car, and it cannot be attributed to a compound at all. That is exactly
 * why their intervals for Hard and Medium overlap almost completely.
 *
 * This model keeps the part of theirs that works -- a latent pace state per car
 * that random-walks with a slope and resets at a pit stop -- and replaces the
 * single slope with `v[compound]`, shared across every car in the race. Twenty
 * cars on different compounds at different tyre ages identify the slopes that
 * one car cannot.
 *
 * WHY THE FUEL COEFFICIENT IS SHARED
 *
 * Mass sensitivity is a property of the circuit, not of the car, so `gamma` is
 * one parameter for the whole field. Pooling is what makes it separable from
 * degradation: for a single car fuel mass and tyre age both march with lap
 * number and are collinear, which is how their fit ended up putting roughly
 * half the fuel effect into the latent state. Across the field, cars pit at
 * different laps, so the two stop moving together.
 *
 * IDENTIFICATION, STATED PLAINLY
 *
 * A per-car level plus a per-car latent walk plus a shared slope is only
 * identified because the slope is shared. Nothing here recovers a compound
 * ordering from a single car, and it is not meant to.
 */
data {
  int<lower=1> N;                        // observations (all cars, all laps)
  int<lower=1> D;                        // drivers
  int<lower=1> T;                        // laps in the race window
  int<lower=1> C;                        // compounds present

  array[N] int<lower=1, upper=D> driver;
  array[N] int<lower=1, upper=T> lap;
  array[N] int<lower=1, upper=C> compound;
  vector[N] tyre_life;                   // laps on the current set
  vector[N] fuel_mass;                   // kg, from the physical burn model
  vector[N] y;                           // lap time, seconds

  // Pit[d, t] == 1 means driver d took new tyres entering lap t, so the latent
  // pace state resets rather than continuing its walk.
  array[D, T] int<lower=0, upper=1> pit;

  real level0;                           // rough pace level, for centring priors
  real<lower=0> sdo0;                    // rough observation scale

  // Prior on the fuel coefficient. Defaults (0.032, 0.02) are set in
  // hierarchical.build_data; the paper's gamma audit also fits a diffuse prior
  // to see what the data say on their own.
  real gamma_prior_mean;
  real<lower=0> gamma_prior_sd;

  // The lap we are forecasting, one step past the training window.
  int<lower=0> n_pred;
  array[n_pred] int<lower=1, upper=D> pred_driver;
  array[n_pred] int<lower=1, upper=C> pred_compound;
  vector[n_pred] pred_tyre_life;
  vector[n_pred] pred_fuel_mass;
}

parameters {
  // Scales are bounded away from zero. At the default random inits an unbounded
  // scale starts at or near 0 and every normal_lpdf throws before the sampler
  // has moved, which is what filled the first run's log.
  real<lower=0.02> sdo;                  // observation scale
  real<lower=0.01> sdp;                  // latent walk scale

  // NON-CENTRED. The latent walk is built from standard normals and scaled in
  // transformed parameters. Centred, the walk's scale and its 1400 increments
  // are strongly coupled and the sampler funnels: the first run put chain 4 at
  // max treedepth on 15% of iterations. This is the standard fix, not a tuning
  // choice, and it changes the geometry rather than the model.
  matrix[D, T] z_raw;
  vector[D] level_raw;

  real level_mu;                         // common pace level across the field
  real<lower=0.01> level_sd;             // spread of car pace, hierarchical
  vector<lower=0>[C] v;                  // degradation, seconds per lap, POOLED
  real gamma;                            // fuel, seconds per kg

  // Degrees of freedom for the observation errors. Estimated, not fixed: how
  // heavy the tails are is a property of the race, and a race with a safety
  // car or a spin has heavier tails than a clean one.
  real<lower=2> nu;
}

transformed parameters {
  vector[D] level = level_mu + level_sd * level_raw;
  matrix[D, T] z;                        // latent pace state per car

  for (d in 1:D) {
    // The first lap of the window has no predecessor, so it sits on the car's
    // own level with a wider scale than a single lap's walk.
    z[d, 1] = level[d] + 0.5 * z_raw[d, 1];
    for (t in 2:T) {
      if (pit[d, t] == 1) {
        z[d, t] = level[d] + sdp * z_raw[d, t];
      } else {
        z[d, t] = z[d, t - 1] + sdp * z_raw[d, t];
      }
    }
  }
}

model {
  sdo ~ normal(sdo0, 0.5);
  sdp ~ normal(0.1, 0.1);

  // Cars are exchangeable around a common pace level. This is the hierarchy
  // their single-driver model has no room for.
  level_mu ~ normal(level0, 2.0);
  level_sd ~ normal(0, 1.0);
  to_vector(level_raw) ~ std_normal();
  to_vector(z_raw) ~ std_normal();

  // Same prior their model puts on its single slope, now per compound. Kept
  // deliberately at their numbers so the comparison is of structure, not of
  // whose prior is sharper.
  v ~ normal(0.05, 0.1);

  // Physical mass sensitivity is 0.030-0.035 s/kg. Weak enough that the data
  // can overrule it, tight enough that the latent states cannot quietly eat
  // the fuel effect the way theirs did.
  gamma ~ normal(gamma_prior_mean, gamma_prior_sd);

  // HEAVY TAILS, and this is the point of the model rather than a detail.
  //
  // A driver locks a brake, runs wide or catches traffic, loses a second, and
  // is back on target pace the next lap. Austria 2025 lap 48 is exactly that:
  // Hamilton 1.31 s slower than the field median while the field was unaffected,
  // a +2.8 sigma event inside the five laps their scheme scores. Under a normal
  // observation model that lap does two kinds of damage -- it drags the latent
  // state and inflates sdo during fitting, and it is punished enormously by CRPS
  // at scoring time.
  //
  // The benchmark's own ablation is the evidence this matters: their skew-t
  // model scored 0.202 against their base model's 0.230, a 12% gain, and their
  // paper attributes it to robustness rather than tyre physics. We take the
  // finding, not the exact distribution: Student-t with estimated nu, which is
  // symmetric and has one fewer parameter than their skewed generalised t.
  nu ~ gamma(2, 0.1);
  {
    vector[N] mu;
    for (i in 1:N) {
      mu[i] = z[driver[i], lap[i]]
            + v[compound[i]] * tyre_life[i]
            + gamma * fuel_mass[i];
    }
    y ~ student_t(nu, mu, sdo);
  }
}

generated quantities {
  // One-step-ahead predictive draws for the held-out lap. The latent state
  // walks one lap forward, then the observation is drawn. Returning draws
  // rather than a mean is deliberate: CRPS is computed from the ensemble, by
  // the estimator already checked against R's scoringRules.
  // The observation draw uses the SAME Student-t the likelihood used. Their
  // code does not do this: full_race_1driver_and_fuel_base_t.stan fits with
  // skew_t and then forecasts with normal_rng, so its predictive intervals are
  // thinner-tailed than the model it fitted. Matching them here is the whole
  // reason heavy tails help the score rather than only the fit.
  vector[n_pred] y_pred;
  for (k in 1:n_pred) {
    real z_next = normal_rng(z[pred_driver[k], T], sdp);
    y_pred[k] = student_t_rng(
      nu,
      z_next + v[pred_compound[k]] * pred_tyre_life[k] + gamma * pred_fuel_mass[k],
      sdo
    );
  }
}
