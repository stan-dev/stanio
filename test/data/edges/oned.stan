generated quantities {
  real base = normal_rng(0, 1);
  array[2, 1, 3] real A = {{{base, base * 1, base * 2}},
                           {{base * 3, base * 4, base * 5}}};

  array[1,1,1] real dummy = {{{base*10}}};

  array[1] tuple(real, array[1] real) dummy_tuple = {(base*11, {base*12})};
  }
