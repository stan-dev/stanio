generated quantities {
  real base = normal_rng(0, 1);
  int base_i = to_int(normal_rng(10, 10));

  tuple(real, real) pair = (base, base * 2);

  tuple(real, tuple(int, complex)) nested = (base * 3, (base_i, base * 4.0i));
  array[2] tuple(real, real) arr_pair = {pair, (base * 5, base * 6)};

  array[3] tuple(tuple(real, tuple(int, complex)), real) arr_very_nested
    = {(nested, base*7), ((base*8, (base_i*2, base*9.0i)), base * 10), (nested, base*11)};

  array[3,2] tuple(real, real) arr_2d_pair = {{(base * 12, base * 13), (base * 14, base * 15)},
                                              {(base * 16, base * 17), (base * 18, base * 19)},
                                              {(base * 20, base * 21), (base * 22, base * 23)}};

  real basep1 = base + 1, basep2 = base + 2;
  real basep3 = base + 3, basep4 = base + 4, basep5 = base + 5;
  array[2,3] tuple(array[2] tuple(real, vector[2]), matrix[4,5]) ultimate =
    {
      {(
        {(base, [base *2, base *3]'), (base *4, [base*5, base*6]')},
        to_matrix(linspaced_vector(20, 7, 11), 4, 5) * base
        ),
       (
        {(basep1, [basep1 *2, basep1 *3]'), (basep1 *4, [basep1*5, basep1*6]')},
        to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep1
        ),
        (
          {(basep2, [basep2 *2, basep2 *3]'), (basep2 *4, [basep2*5, basep2*6]')},
          to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep2
       )
     },
     {(
        {(basep3, [basep3 *2, basep3 *3]'), (basep3 *4, [basep3*5, basep3*6]')},
        to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep3
        ),
       (
        {(basep4, [basep4 *2, basep4 *3]'), (basep4 *4, [basep4*5, basep4*6]')},
        to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep4
        ),
        (
          {(basep5, [basep5 *2, basep5 *3]'), (basep5 *4, [basep5*5, basep5*6]')},
          to_matrix(linspaced_vector(20, 7, 11), 4, 5) * basep5
       )
     }};

 }
