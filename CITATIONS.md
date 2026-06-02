# Citations

Upstream work, papers, and libraries the deke project build on.

Vendored third-party source carries its own license; copies live under
[`vendored_license/`](vendored_license/) and are noted inline below.

## deke-kin

EAIK — Automatic Geometric Decomposition for Analytical Inverse Kinematics
(<https://github.com/OstermD/EAIK>, BSD-3-Clause).

```bibtex
@ARTICLE{EAIK,
  author={Ostermeier, Daniel and K{\"u}lz, Jonathan and Althoff, Matthias},
  journal={IEEE Robotics and Automation Letters},
  title={Automatic Geometric Decomposition for Analytical Inverse Kinematics},
  year={2025},
  volume={10},
  number={10},
  pages={9964-9971},
  doi={10.1109/LRA.2025.3597897}
}
```

ik-geo — Unified robot inverse kinematics using subproblem decomposition
(<https://github.com/rpiRobotics/ik-geo>, BSD-3-Clause). A fork (v0.1.2), slimmed
to the Paden–Kahan subproblem kernels, is **vendored** under
`deke-kin/src/ik_geo/`. License:
[`vendored_license/deke-kin/ik-geo-LICENSE`](vendored_license/deke-kin/ik-geo-LICENSE)
(BSD-3-Clause, © 2024 Verdant Evolution, LLC).

Subproblem decomposition (Elias / Wen) — the paper behind ik-geo:

```bibtex
@article{ik-geo,
  author  = {Elias, Alexander J. and Wen, John T.},
  title   = {IK-Geo: Unified Robot Inverse Kinematics Using Subproblem Decomposition},
  journal = {Mechanism and Machine Theory},
  volume  = {209},
  pages   = {105971},
  year    = {2025},
  note    = {arXiv:2211.05737},
  url     = {https://arxiv.org/abs/2211.05737}
}
```

### Improved Generic-6R solver (`src/rr_ik.rs`)

The fallback solver for 6R chains with no closed-form decomposition uses
Raghavan–Roth resultant elimination solved as a Manocha–Canny eigenvalue problem.

Raghavan–Roth elimination:

```bibtex
@article{raghavan1993inverse,
  author  = {Raghavan, Madhusudan and Roth, Bernard},
  title   = {Inverse Kinematics of the General 6R Manipulator and Related Linkages},
  journal = {ASME Journal of Mechanical Design},
  volume  = {115},
  number  = {3},
  pages   = {502--508},
  year    = {1993},
  doi     = {10.1115/1.2919218}
}
```

Tsai, *Robot Analysis*, Appendix C ("Raghavan and Roth's Solution", Eqs. C.1–C.15)
— the RR equations used for the matrix assembly.

```bibtex
@book{tsai1999robot,
  author    = {Tsai, Lung-Wen},
  title     = {Robot Analysis: The Mechanics of Serial and Parallel Manipulators},
  publisher = {Wiley},
  year      = {1999}
}
```

Manocha–Canny eigenvalue formulation — the reduced system as a 12×12 matrix
quadratic `Σ(x3) = A·x3² + B·x3 + C`, linearised to a 24×24 generalized
eigenproblem. Solved with faer 0.24.0 `Mat::generalized_eigen`.

```bibtex
@article{manocha1994efficient,
  author  = {Manocha, Dinesh and Canny, John F.},
  title   = {Efficient Inverse Kinematics for General 6R Manipulators},
  journal = {IEEE Transactions on Robotics and Automation},
  volume  = {10},
  number  = {5},
  pages   = {648--657},
  year    = {1994},
  url     = {https://people.eecs.berkeley.edu/~jfc/papers/94/MCtra94.pdf}
}
```

## deke-topp3tcp-nlp, deke-topp3tcp-spline

Time-optimal path tracking as a convex optimization over a reparameterized fixed path:

```bibtex
@article{verscheure2009timeoptimal,
  author  = {Verscheure, Diederik and Demeulenaere, Bram and Swevers, Jan and De Schutter, Joris and Diehl, Moritz},
  title   = {Time-Optimal Path Tracking for Robots: A Convex Optimization Approach},
  journal = {IEEE Transactions on Automatic Control},
  volume  = {54},
  number  = {10},
  pages   = {2318--2327},
  year    = {2009},
  doi     = {10.1109/TAC.2009.2028959}
}
```

Path derivatives use a chord-length PCHIP interpolant:

```bibtex
@article{fritsch1980monotone,
  author  = {Fritsch, F. N. and Carlson, R. E.},
  title   = {Monotone Piecewise Cubic Interpolation},
  journal = {SIAM Journal on Numerical Analysis},
  volume  = {17},
  number  = {2},
  pages   = {238--246},
  year    = {1980},
  doi     = {10.1137/0717021}
}
```

Time-optimal path parameterization:

```bibtex
@article{pham2018toppra,
  author  = {Pham, Hung and Pham, Quang-Cuong},
  title   = {A New Approach to Time-Optimal Path Parameterization based on Reachability Analysis},
  journal = {IEEE Transactions on Robotics},
  volume  = {34},
  number  = {3},
  pages   = {645--659},
  year    = {2018},
  note    = {arXiv:1707.07239},
  url     = {https://arxiv.org/abs/1707.07239}
}
```

Third-order (jerk-limited) TOPP structure and the TOPP3 algorithm (Pham & Pham,
ICRA 2017):

```bibtex
@inproceedings{pham2017topp3,
  author    = {Pham, Hung and Pham, Quang-Cuong},
  title     = {On the Structure of the Time-Optimal Path Parameterization Problem with Third-Order Constraints},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2017},
  note      = {arXiv:1609.05307},
  url       = {https://arxiv.org/abs/1609.05307}
}
```

## deke-topp-speed

Jerk-limited (S-curve) motion profiling — Berscheid & Kröger, "Jerk-limited
Real-time Trajectory Generation with Arbitrary Target States" (arXiv:2105.04830,
2021).

```bibtex
@article{berscheid2021jerk,
  author  = {Berscheid, Lars and Kr{\"o}ger, Torsten},
  title   = {Jerk-limited Real-time Trajectory Generation with Arbitrary Target States},
  year    = {2021},
  note    = {arXiv:2105.04830},
  url     = {https://arxiv.org/abs/2105.04830}
}
```

## deke-rrt

RRT-Connect:

```
J. J. Kuffner and S. M. LaValle. "RRT-Connect: An efficient approach to
single-query path planning". In: IEEE International Conference on Robotics and
Automation. Vol. 2. IEEE. 2000, pp. 995–1001.
```

VAMP — Motions in Microseconds via Vectorized Sampling-Based Planning
(<https://github.com/KavrakiLab/vamp>, Apache-2.0). arXiv:2309.14545.

```bibtex
@InProceedings{vamp_2024,
  author = {Thomason, Wil and Kingston, Zachary and Kavraki, Lydia E.},
  title = {Motions in Microseconds via Vectorized Sampling-Based Planning},
  booktitle = {IEEE International Conference on Robotics and Automation},
  pages = {8749--8756},
  url = {http://arxiv.org/abs/2309.14545},
  doi = {10.1109/ICRA57147.2024.10611190},
  date = {2024}
}
```

AORRTC — Almost-Surely Asymptotically Optimal Planning with RRT-Connect
(<https://robotic-esp.com/papers/wilson_ral25>, arXiv:2505.10542).

```bibtex
@article{aorrtc_2025,
  author = {Wilson, Tyler S. and Thomason, Wil and Kingston, Zachary and Gammell, Jonathan D.},
  title = {AORRTC: Almost-surely asymptotically optimal planning with RRT-Connect},
  journal = {IEEE Robotics and Automation Letters},
  url = {https://arxiv.org/abs/2505.10542},
  year = {2025},
  note = {Under Review}
}
```
