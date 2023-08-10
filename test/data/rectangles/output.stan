parameters {
  real mu;
}
model {
  mu ~ normal(0, 1);
}
generated quantities {
  vector[2] v = [mu, mu * 2]';
  row_vector[3] r = mu + [mu, mu * 2, mu * 3];
  matrix[2, 3] m = mu * to_matrix(linspaced_vector(6, 5, 11), 2, 3);
  print(mu);
  print(m);
  print( m ./ mu);
  array[4] matrix[2, 3] threeD;
  for (i in 1 : 4) {
    threeD[i] = i * mu * to_matrix(linspaced_vector(6, 5, 11), 2, 3);
  }

  real nu = normal_rng(mu, 1);
  complex z = nu + nu * 2.0i;
  complex_vector[2] zv = to_complex([3 * nu, 5 * nu]', [nu * 4, nu * 6]');
  print(nu);
  print(zv);
  complex_matrix[2, 3] zm = to_complex(m, m + 1);
  array[4] complex_matrix[2, 3] z3D;
  for (i in 1 : 4) {
    z3D[i] = to_complex(threeD[i], threeD[i] + 1);
  }
}
