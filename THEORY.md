# Estimand, properties, and proofs

This note describes the statistical target of `structnpe.fit` and the
properties that follow under stated assumptions. These are standard results
or elementary consequences of them. They are not new theorems claimed by this
package. A trained neural network is an approximation; the empirical evidence
for this implementation is reported separately in [BENCHMARKS.md](BENCHMARKS.md).

## 1. Setup and implementation

Let $\Theta\sim\pi$, $Y\mid\Theta=\theta\sim P_\theta$, and $X=S(Y)$.
For this note, fix the representation and preprocessing before evaluating
population risks on fresh prior-predictive draws. Assume regular conditional
distributions exist. Where densities or entropies are used, assume appropriate
dominating measures and finite expectations; the same identities apply to
probability mass functions.

Write $p_x=p(\theta\mid X=x)$ for the representation posterior and $Q_x$ for
an approximation with density $q(\theta\mid x)$. This posterior is defined
under the declared simulator and prior. It need not describe the real data if
that model is misspecified. It is also different from a maximum-likelihood
estimate.

The current public estimator:

1. Draws parameters from the supplied prior and simulates one dataset per draw.
2. Splits simulated datasets, fits the observation adapter on the training
   split, and freezes it; standardization also uses that split.
3. Transforms each parameter using its declared bounds and standardizes it.
4. Fits a neural mixture of $K$ diagonal Gaussians by conditional log loss,
   using AdamW and validation-based early stopping.
5. Samples the fitted joint mixture and maps draws back to parameter units.

In standardized coordinates $z=A(\theta)$ and $u=B(x)$, the density is

$$
q_\phi^Z(z\mid u)=\sum_{k=1}^K w_k(u)
\mathcal N\!\left(z;m_k(u),\operatorname{diag}(\exp\ell_k(u))\right).
$$

The default $K$ is five; the implementation clips log variances to $[-7,5]$.
Diagonal components do not imply an independent joint posterior: the shared
mixture component can induce dependence. A fixed finite mixture can still
miss important dependence, tails, or modes.

## 2. Population log loss targets the representation posterior

**Proposition 1 (conditional log-score identity).** For fixed $S$ and a
conditional density $q$ with finite risk,

$$
R(q):=\mathbb E[-\log q(\Theta\mid X)]
=H(\Theta\mid X)+\mathbb E_X
\operatorname{KL}\!\left(p_X\,\Vert\,Q_X\right).
$$

Consequently, unrestricted population minimization recovers $p_x$ for
prior-predictive-almost-every $x$. Minimization over a restricted family
instead gives its best attainable average forward-KL approximation, if a
minimizer exists.

**Proof.** Add and subtract $-\log p(\Theta\mid X)$ inside the expectation.
The first term is conditional entropy. Conditioning the remaining log ratio
on $X$ gives the displayed KL divergence. Nonnegativity of KL gives the
minimization claim, with equality only when the conditional laws agree almost
everywhere.

This is the conditional version of the proper logarithmic score
([Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437))
and the population basis for neural posterior estimation
([Papamakarios and Murray, 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf)).

### What a convergence claim would require

**Corollary 1 (conditional risk bound).** Let $\mathcal Q_n$ be a candidate
family, $\widehat R_n$ its empirical log loss, and $R_*=H(\Theta\mid X)$.
Suppose

$$
\sup_{q\in\mathcal Q_n}|\widehat R_n(q)-R(q)|\leq\delta_n,
\qquad
\widehat R_n(\widehat q_n)\leq
\inf_{q\in\mathcal Q_n}\widehat R_n(q)+\varepsilon_n.
$$

With $a_n=\inf_{q\in\mathcal Q_n}R(q)-R_*$,

$$
\mathbb E_X\operatorname{KL}(p_X\Vert\widehat Q_{n,X})
\leq a_n+2\delta_n+\varepsilon_n.
$$

**Proof.** Bound $R(\widehat q_n)$ by
$\widehat R_n(\widehat q_n)+\delta_n$, apply approximate minimization,
then bound the empirical infimum by the population infimum plus $\delta_n$.
Subtract $R_*$ and apply Proposition 1.

Vanishing approximation, generalization, and optimization errors therefore
imply average-KL convergence, in the mode in which the bound vanishes.
This is a conditional theorem, not an established convergence guarantee for
the package: the current fixed architecture, variance clipping, optimizer,
and early stopping do not establish these assumptions. In particular,
increasing the simulation count alone does not prove that $a_n$ vanishes.

## 3. What summaries discard

**Proposition 2 (representation and estimation error decomposition).**
For deterministic $X=S(Y)$ and finite terms,

$$
\mathbb E_Y\operatorname{KL}\!\left(p(\Theta\mid Y)\Vert Q_{S(Y)}\right)
=I(\Theta;Y\mid X)
+\mathbb E_X\operatorname{KL}(p_X\Vert Q_X).
$$

**Proof.** Insert $p(\Theta\mid X)$ into the density ratio. The expectation
of $\log[p(\Theta\mid Y)/p(\Theta\mid X)]$ is conditional mutual
information because $X$ is a function of $Y$. The expectation of
$\log[p(\Theta\mid X)/q(\Theta\mid X)]$, after conditioning on $X$,
is the second term.

Thus a perfect density estimator removes the second term but leaves the
information lost by $S$. The representation posterior equals the full-data
posterior almost surely exactly when $\Theta$ and $Y$ are conditionally
independent given $X$ under the declared prior-predictive law. This is a
Bayesian sufficiency statement for that law, not a guarantee for every prior.

For example, aggregate purchase means need not preserve the information in
the timing of purchases. A more expressive network cannot reconstruct
histories that the representation has discarded.

## 4. Parameter transformations preserve the statistical target

**Proposition 3 (change of coordinates).** Let $A$ be a differentiable
bijection from the interior of the parameter domain to transformed coordinates,
with nonzero Jacobian determinant. If the fitted transformed density is
$q^Z(z\mid B(x))$, its induced parameter density is

$$
q^\Theta(\theta\mid x)
=q^Z(A(\theta)\mid B(x))\,|\det DA(\theta)|.
$$

Conditional KL divergence is unchanged when both parameter laws are
transformed by $A$.

**Proof.** The density expression is the change-of-variables formula. In
the KL ratio the two Jacobians cancel; changing the integration variable
gives the invariance result. Since the Jacobian is independent of network
weights, training in transformed coordinates changes the original-coordinate
log loss only by a term independent of those weights.

The observation standardization $B$ is invertible on the represented array
and adds no information loss. This statement concerns the already chosen
representation $S$. The package's log and logit transforms respect declared
coordinate bounds mathematically; floating-point overflow or boundary
rounding still needs numerical handling. Coordinate bounds also need not
describe every restriction in a user's prior support.

## 5. Economic quantities from joint posterior draws

Let $h(\theta)$ be a measurable scalar or vector quantity defined by the
structural model, such as an elasticity or a promotion's consumption effect.
Applying $h$ to each joint posterior draw estimates the induced distribution
$h_\#Q_x$ (the distribution of $h(\Theta)$ when $\Theta\sim Q_x$).

**Proposition 4 (distributional and bounded-mean error).** With
$\operatorname{TV}(P,Q)=\sup_A|P(A)-Q(A)|$,

$$
\operatorname{TV}(h_\#p_x,h_\#Q_x)
\leq\operatorname{TV}(p_x,Q_x)
\leq\sqrt{\tfrac12\operatorname{KL}(p_x\Vert Q_x)}.
$$

For scalar $h\in[a,b]$,

$$
|\mathbb E_{p_x}h-\mathbb E_{Q_x}h|
\leq(b-a)\operatorname{TV}(p_x,Q_x).
$$

**Proof.** Every event in the transformed space has a preimage in parameter
space, proving the first inequality. The second is Pinsker's inequality.
For the expectation bound, write $h-a$ as the integral of its level-set
indicators over $[0,b-a]$, then bound each probability difference by TV.

These bounds apply if posterior error is controlled; the software does not
know its KL error merely from training loss. Unbounded profit or welfare
functions require additional tail or moment control. Numerical error in
solving $h$ is also separate from posterior approximation error.

For independent draws from a fixed fitted $Q_x$, the sample mean of an
integrable $h$ converges to $\mathbb E_{Q_x}h$ by the law of large numbers.
With finite variance its Monte Carlo standard error is
$\sqrt{\operatorname{Var}_{Q_x}(h)/R}$. More draws reduce that sampling error,
not the error in $Q_x$. Computing a nonlinear $h$ at the posterior mean
generally gives a different object from the posterior mean of $h$.

## 6. What Bayesian calibration does and does not establish

**Proposition 5 (ideal prior-predictive coverage).** If a measurable set
$C(X)$ has exact posterior probability $p_X(C(X))=1-\alpha$ almost surely,
then, under the declared prior-predictive law,

$$
\Pr\{\Theta\in C(X)\}=1-\alpha.
$$

**Proof.** Apply iterated expectation to
$\mathbf 1\{\Theta\in C(X)\}$, conditioning on $X$.

If instead $Q_X(C(X))=1-\alpha$, the same argument gives

$$
\left|\Pr\{\Theta\in C(X)\}-(1-\alpha)\right|
\leq\mathbb E_X\operatorname{TV}(p_X,Q_X).
$$

For a continuous scalar parameter or functional, the true simulated value
and independent exact-posterior draws are exchangeable conditional on $X$;
its rank is therefore discrete uniform. Ties require appropriate randomization.
This underlies simulation-based calibration
([Talts et al., 2018](https://arxiv.org/abs/1804.06788)).

Neither identity promises frequentist coverage at every fixed parameter
value. Passing an aggregate coverage or rank diagnostic is also not a
sufficient condition for posterior accuracy: an algorithm returning prior
draws irrespective of data can pass marginal SBC ranks. Local checks,
informative posterior comparisons, and model checks serve different purposes.
An exact representation posterior can also pass SBC despite losing
information relative to the full data.

## 7. Prior sampling and sequential extensions

The current public `fit` samples from `model.prior`. If training parameters
are instead drawn from $g(\theta)$, unweighted population log loss targets

$$
p_g(\theta\mid x)\propto g(\theta)p_S(x\mid\theta),
$$

not the posterior under the original prior $\pi$. With suitable support and
integrability, weighting each loss by $\pi(\theta)/g(\theta)$ restores the
original population criterion. This follows directly by changing measure in
the training expectation; its finite-sample variance can be large.

Sequential NPE requires a justified proposal correction, such as those
studied by [Greenberg et al. (2019)](https://proceedings.mlr.press/v97/greenberg19a.html).
It is not implemented by merely replacing the prior callable in this package.

## What is established for this release?

| Statement | Status |
| --- | --- |
| Log-loss target, representation decomposition, and transformation identities | Mathematical properties under the assumptions above |
| Convergence as simulations increase | Conditional corollary; assumptions not established for the fixed implementation |
| Accurate Eight Schools hyperposterior on the reported dataset | Five-seed numerical comparison through the public API |
| Accurate generic MDN on the reported Rust panel | Not established; the documented run fails its accuracy limits |
| Rust-specific simulation-trained grid posterior | Separate successful benchmark; different estimator |
| Uniform local calibration, arbitrary-model identification, or general speed superiority | Not established |

NPE can make posterior approximation practical without evaluating the
likelihood. It does not supply identification, establish a simulator's
empirical validity, or eliminate the cost of solving its economic model.
