# Recommended metrics

HullProd organizes the surface signature as **intensity**, **sign**, and
**areal extent**. Let `K` be Gaussian curvature, `H` mean curvature, `L_ref`
the declared or automatic reference length, and `S_K` the represented domain
where Gaussian curvature is valid and integrable, with area `A_K`.

## Total double-curvature intensity

The local dimensionless developability-deviation density is

```math
q_D(x) = |K(x)| L_{ref}^2.
```

Its represented-valid-area average is

```math
I_D = \frac{1}{A_K}\int_{S_K}|K|L_{ref}^2\,dA.
```

`I_D = 0` on an exactly developable surface. Increasing `I_D` means that the
valid represented surface has greater average double-curvature intensity at
the chosen reference scale.

## Signed intensity

Positive and negative Gaussian curvature are kept separate:

```math
q_D^+ = \max(K,0)L_{ref}^2, \qquad
q_D^- = \max(-K,0)L_{ref}^2,
```

```math
I_D^\pm = \frac{1}{A_K}\int_{S_K}q_D^\pm\,dA,
\qquad I_D = I_D^+ + I_D^-.
```

`I_D_plus` measures elliptic/synclastic intensity. `I_D_minus` measures
saddle/reverse/anticlastic intensity. These are nonnegative contributions, not
opposing signed numbers.

## Curvature-class areal extent

HullProd partitions represented valid area into four classes. With

```math
\tau_H = h_f/L_{ref}, \qquad \tau_K = k_f/L_{ref}^2,
```

the classes are:

| Component | Numerical condition | Plain-English interpretation |
|---|---|---|
| `a_C_flat` | `|K| <= tau_K` and `|H| <= tau_H` | both principal curvatures are quasi-zero |
| `a_C_single` | `|K| <= tau_K` and `|H| > tau_H` | one principal curvature is quasi-zero |
| `a_C_elliptic` | `K > tau_K` | elliptic/synclastic double curvature |
| `a_C_saddle` | `K < -tau_K` | saddle/reverse/anticlastic double curvature |

Each component is its class area divided by the total valid class area, so the
four fractions sum to one when the classification is valid.

The defaults `h_f = k_f = 1e-4` are dimensionless quasi-zero numerical
thresholds. They state what is treated as numerically indistinguishable from
zero at the chosen reference scale; they are not plate-forming or manufacturing
acceptance limits.

Together, `[I_D, I_D_plus, I_D_minus, a_C]` describes curvature intensity,
sign, and areal extent. These are geometry-based screening descriptors. They
do not directly predict cost, labor, forming route, or process feasibility, and
HullProd does not combine them into a universal scalar score.

`L_ref` is a length expressed in the input geometry's working unit. An explicit
`--lref` uses that unit exactly; it is not assumed to be metres. The automatic
mode uses a geometric principal-axis span and is never described as `L_pp`.
