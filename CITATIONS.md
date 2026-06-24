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

### Manipulability (`src/kinematics/fk_chain.rs`)

`FkChain::manipulability` returns the Yoshikawa manipulability measure
`√det(J·Jᵀ)` (for a square 6-DOF arm, `|det J|`):

```bibtex
@article{yoshikawa1985manipulability,
  author  = {Yoshikawa, Tsuneo},
  title   = {Manipulability of Robotic Mechanisms},
  journal = {The International Journal of Robotics Research},
  volume  = {4},
  number  = {2},
  pages   = {3--9},
  year    = {1985},
  doi     = {10.1177/027836498500400201}
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

```text
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

## deke-multipath

Ordering required paths is an asymmetric generalized TSP (one option per
cluster, choosing direction/variant) — the robotic task sequencing problem with
sampled configurations.

Exact solver: the bitmask dynamic program keyed on `(visited set, last vertex)`
is the Held–Karp sequencing DP, applied per cluster.

```bibtex
@article{held1962dynamic,
  author  = {Held, Michael and Karp, Richard M.},
  title   = {A Dynamic Programming Approach to Sequencing Problems},
  journal = {Journal of the Society for Industrial and Applied Mathematics},
  volume  = {10},
  number  = {1},
  pages   = {196--210},
  year    = {1962},
  doi     = {10.1137/0110015}
}
```

Heuristic construction — decouple task-space ordering from configuration
selection (`src/agtsp.rs::solve_heuristic` orders clusters with a cheap
surrogate, then picks options):

```bibtex
@inproceedings{suarezruiz2018robotsp,
  author    = {Su{\'a}rez-Ruiz, Francisco and Lembono, Teguh Santoso and Pham, Quang-Cuong},
  title     = {RoboTSP -- A Fast Solution to the Robotic Task Sequencing Problem},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2018},
  note      = {arXiv:1709.09343},
  url       = {https://arxiv.org/abs/1709.09343}
}
```

Cluster optimization — given a fixed cluster order, the optimal one-vertex-per
-cluster selection is a layered-DAG shortest path
(`src/agtsp.rs::cluster_optimize`):

```bibtex
@article{karapetyan2011lk,
  author  = {Karapetyan, Daniel and Gutin, Gregory},
  title   = {Lin--Kernighan Heuristic Adaptations for the Generalized Traveling Salesman Problem},
  journal = {European Journal of Operational Research},
  volume  = {208},
  number  = {3},
  pages   = {221--232},
  year    = {2011},
  note    = {arXiv:1003.5330},
  url     = {https://arxiv.org/abs/1003.5330}
}
```

Related GTSP solvers consulted for the design (heavy-instance paths not yet
implemented — Noon–Bean reduction to ATSP, GLKH, and the GLNS large-neighbourhood
search):

```bibtex
@article{noon1993transformation,
  author  = {Noon, Charles E. and Bean, James C.},
  title   = {An Efficient Transformation of the Generalized Traveling Salesman Problem},
  journal = {INFOR: Information Systems and Operational Research},
  volume  = {31},
  number  = {1},
  pages   = {39--44},
  year    = {1993},
  doi     = {10.1080/03155986.1993.11732212}
}

@article{helsgaun2015gtsp,
  author  = {Helsgaun, Keld},
  title   = {Solving the Equality Generalized Traveling Salesman Problem Using the Lin--Kernighan--Helsgaun Algorithm},
  journal = {Mathematical Programming Computation},
  volume  = {7},
  pages   = {269--287},
  year    = {2015},
  doi     = {10.1007/s12532-015-0080-8}
}

@article{smith2017glns,
  author  = {Smith, Stephen L. and Imeson, Frank},
  title   = {GLNS: An Effective Large Neighborhood Search Heuristic for the Generalized Traveling Salesman Problem},
  journal = {Computers \& Operations Research},
  volume  = {87},
  pages   = {1--19},
  year    = {2017},
  doi     = {10.1016/j.cor.2017.05.010}
}
```

## deke-linear

Constant-TCP-speed Cartesian polyline following. The planner discretizes the
Cartesian path, inverts every sample, and routes a continuous, validator-checked
joint track through the per-sample IK candidates with a dynamic program — the
ladder-graph pattern of **ROS-Industrial Descartes**
(<https://github.com/ros-industrial-consortium/descartes>, Apache-2.0; see also
"Cartesian path planning for arc welding robots: Evaluation of the descartes
algorithm", IEEE ETFA 2017, <https://ieeexplore.ieee.org/document/8247616>).

The redundant planner extends that DP over the tool's free spin axis, resolving
the kinematic redundancy of a path-constrained arm in one global pass:

```bibtex
@article{ferrentino2023dp,
  author  = {Ferrentino, Enrico and Savino, Heitor J. and Franchi, Antonio and Chiacchio, Pasquale},
  title   = {A Dynamic Programming Framework for Optimal Planning of Redundant Robots Along Prescribed Paths With Kineto-Dynamic Constraints},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2023},
  note    = {arXiv:2207.05622},
  url     = {https://arxiv.org/abs/2207.05622}
}
```

Treating the welding torch's rotation about its own axis as a free DOF (functional
redundancy of the symmetric tool) follows the twist-decomposition welding work:

```bibtex
@article{huo2008jointlimits,
  author  = {Huo, Liguo and Baron, Luc},
  title   = {The joint-limits and singularity avoidance in robotic welding},
  journal = {Industrial Robot: An International Journal},
  volume  = {35},
  number  = {5},
  pages   = {456--464},
  year    = {2008},
  doi     = {10.1108/01439910810893626}
}
```

The DP node cost is the Yoshikawa manipulability measure (`√det(J·Jᵀ)`, computed
by `deke-kin`'s `FkChain::manipulability` — cited under [deke-kin](#deke-kin)).

The constant-feedrate retimer reuses the phase-plane time-optimal machinery — the
maximum-velocity curve `min_j v_max,j / |q'_j(s)|` and a backward/forward
acceleration-bounded pass (`v² ± 2 a Δs`) — but clamps the feedrate to the
commanded speed instead of maximizing it:

```bibtex
@article{bobrow1985timeoptimal,
  author  = {Bobrow, J. E. and Dubowsky, S. and Gibson, J. S.},
  title   = {Time-Optimal Control of Robotic Manipulators Along Specified Paths},
  journal = {The International Journal of Robotics Research},
  volume  = {4},
  number  = {3},
  pages   = {3--17},
  year    = {1985},
  doi     = {10.1177/027836498500400301}
}
```

Cartesian path conditioning (corner smoothing, arc-length parameterization) is
built on the first-party `squiggle` geometry crate, whose interpolating curve is the
Catmull–Rom spline:

```bibtex
@incollection{catmull1974splines,
  author    = {Catmull, Edwin and Rom, Raphael},
  title     = {A Class of Local Interpolating Splines},
  booktitle = {Computer Aided Geometric Design},
  publisher = {Academic Press},
  pages     = {317--326},
  year      = {1974},
  doi       = {10.1016/B978-0-12-079050-0.50020-5}
}
```

### Rail-axis (7th external axis) redundancy

A 6-DOF arm on a prismatic linear rail (`src/rail.rs`) resolves the rail position
as an *additional* smooth redundant DOF in the same global ladder DP, emitting a
rail-first `q = [x_rail, q1..q6]`. The rail enters inverse kinematics as a
base-frame shift — the world target is translated by `−x·â` and solved by the
unchanged 6-DOF analytic IK — the standard coordinated external-axis / linear-track
model. Resolving the rail (like the tool yaw) as a redundant DOF in the
path-constrained DP follows the same Ferrentino et al. framework cited above.
Treating a redundant external axis as a smooth function of the weld arc length,
decoupled from the inner IK, follows recent welding work:

> "An effective path planning approach for robot welding considering redundant
> kinematics", *Robotics and Computer-Integrated Manufacturing* (2024),
> <https://www.sciencedirect.com/science/article/abs/pii/S0141635924002356>.

The rail position schedule `x(s)` is smoothed with the Fritsch–Carlson monotone
PCHIP (cited under [deke-topp3tcp-nlp, deke-topp3tcp-spline](#deke-topp3tcp-nlp-deke-topp3tcp-spline))
— monotone so the slow heavy axis never overshoots a sampled value. Resolving the
rail *upstream* of the constant-feedrate retimer (rather than as a variable inside
it) is required because inverse kinematics is nonlinear in the rail position, which
would break the convexity the timing step relies on — the decoupling principle of
the Verscheure et al. convex path-tracking formulation (cited under
[deke-topp3tcp-nlp, deke-topp3tcp-spline](#deke-topp3tcp-nlp-deke-topp3tcp-spline)).

The DP node cost scores the **arm's** 6-DOF Yoshikawa manipulability, not the
augmented 7-DOF chain's: the rail's prismatic Jacobian column keeps `det(J·Jᵀ)`
high even when the arm itself is singular, so an augmented measure masks arm
singularities and the rail is never recruited to escape them. (Yoshikawa measure
cited under [deke-kin](#deke-kin).)
