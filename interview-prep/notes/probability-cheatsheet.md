# Probability Cheatsheet - Complete Reference

Based on: [Probability Cheatsheet by William Chen & Joe Blitzstein](https://github.com/wzchen/probability_cheatsheet)

**Compiled from**: Harvard's Stat 110 (Probability), licensed CC BY-NC-SA 4.0

---

## Notation Conventions

- **Events**: $A, B, C$ (subsets of sample space)
- **Complement**: $A'$ (not $A$, also written $A^c$ or $\bar{A}$)
- **Random variables**: $X, Y, Z$ (uppercase)
- **Realizations**: $x, y, z$ (lowercase)
- **Probability**: $P(\cdot)$ for events
- **PMF**: $p_X(x) = P(X=x)$ for discrete RVs
- **PDF**: $f_X(x)$ for continuous RVs
- **CDF**: $F_X(x) = P(X \leq x)$ for all RVs
- **Expectation**: $E(X)$ or $\mu$
- **Variance**: $\text{Var}(X)$ or $\sigma^2$
- **Standard deviation**: $\text{SD}(X)$ or $\sigma$
- **Independence**: $X \perp Y$
- **Convergence in distribution**: $\xrightarrow{D}$

---

## 1. Counting Principles

**Multiplication Rule**: If experiment has $r$ components with $n_1, n_2, \ldots, n_r$ possible outcomes respectively, then total outcomes = $n_1 \times n_2 \times \cdots \times n_r$

**Sampling Table** (selecting $k$ samples from population of $n$ objects):

| Order matters? | Replacement? | Formula |
|---|---|---|
| Yes | Yes | $n^k$ |
| Yes | No | $\frac{n!}{(n-k)!}$ |
| No | Yes | $\binom{n+k-1}{k}$ |
| No | No | $\binom{n}{k}$ |

**Naive Definition of Probability**:
$$P(A) = \frac{|\text{favorable outcomes}|}{|\text{total outcomes}|}$$
(Only applies when outcomes are equally likely!)

---

## 2. Conditional Probability & Independence

### Definitions

**Conditional Probability**:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Key insight**: "Conditional probability is probability" — any theorem for $P$ also holds for $P(\cdot|B)$

**Independence**: Events $A$ and $B$ are independent if any of these equivalent conditions hold:
- $P(A \cap B) = P(A)P(B)$
- $P(A|B) = P(A)$ (when $P(B) > 0$)
- $P(B|A) = P(B)$ (when $P(A) > 0$)

### Basic Rules

**De Morgan's Laws**:
- $(A \cup B)' = A' \cap B'$
- $(A \cap B)' = A' \cup B'$

**Intersection via Conditioning**:
- $P(A \cap B) = P(A)P(B|A) = P(B)P(A|B)$
- $P(A \cap B \cap C) = P(A) \cdot P(B|A) \cdot P(C|A,B)$

**Union via Inclusion-Exclusion**:
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- $P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$

### Law of Total Probability (LOTP)

For partition $\{B_1, B_2, \ldots, B_n\}$ (mutually exclusive, collectively exhaustive):
$$P(A) = \sum_{i=1}^n P(A|B_i)P(B_i) = \sum_{i=1}^n P(A \cap B_i)$$

**Special case** (partition into $B$ and $B'$):
$$P(A) = P(A|B)P(B) + P(A|B')P(B')$$

### Bayes' Rule ⭐⭐⭐

**Standard Form**:
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**With LOTP** (most useful form):
$$P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A')P(A')}$$

**Odds Form**:
$$\frac{P(A|B)}{P(A'|B)} = \frac{P(B|A)}{P(B|A')} \times \frac{P(A)}{P(A')}$$

where $\frac{P(A)}{P(A')}$ is prior odds, $\frac{P(B|A)}{P(B|A')}$ is likelihood ratio, and $\frac{P(A|B)}{P(A'|B)}$ is posterior odds.

---

## 3. Random Variables

### Probability Mass Function (PMF)

For discrete RV $X$:
$$p_X(x) = P(X = x)$$

**Properties**:
- $p_X(x) \geq 0$ for all $x$
- $\sum_{x} p_X(x) = 1$

### Cumulative Distribution Function (CDF)

For any RV $X$:
$$F_X(x) = P(X \leq x)$$

**Properties**:
- Right-continuous
- Non-decreasing
- $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$

### Probability Density Function (PDF)

For continuous RV $X$:
$$f_X(x) = \frac{dF_X}{dx}(x)$$

**Properties**:
- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x)\,dx = 1$
- $P(a \leq X \leq b) = \int_a^b f_X(x)\,dx = F_X(b) - F_X(a)$

**Key insight**: For continuous RV, $P(X = c) = 0$ for any $c$

### Universality of Uniform (UoU)

If $X$ is a continuous RV with CDF $F_X$, then:
$$F_X(X) \sim \text{Unif}(0,1)$$

**Converse**: If $U \sim \text{Unif}(0,1)$, then $F_X^{-1}(U)$ has CDF $F_X$

(Useful for simulation: generate uniform, transform to any distribution)

---

## 4. Expected Value

### Definition

**Discrete**:
$$E(X) = \sum_{x} x \cdot P(X=x) = \sum_{x} x \cdot p_X(x)$$

**Continuous**:
$$E(X) = \int_{-\infty}^{\infty} x \cdot f_X(x)\,dx$$

### Linearity of Expectation ⭐⭐⭐

**Most important property**:
$$E(aX + bY + c) = aE(X) + bE(Y) + c$$

**Works for ANY random variables** (even if dependent)!

### LOTUS (Law of the Unconscious Statistician)

To find $E(g(X))$, you don't need the distribution of $g(X)$:

**Discrete**:
$$E(g(X)) = \sum_{x} g(x) \cdot P(X=x)$$

**Continuous**:
$$E(g(X)) = \int_{-\infty}^{\infty} g(x) \cdot f_X(x)\,dx$$

**Multivariate**:
$$E(g(X,Y)) = \sum_x \sum_y g(x,y) \cdot P(X=x, Y=y)$$
$$E(g(X,Y)) = \int \int g(x,y) \cdot f_{X,Y}(x,y)\,dx\,dy$$

### Indicator Random Variables

$I_A$ is the **indicator** of event $A$:
$$I_A = \begin{cases} 1 & \text{if } A \text{ occurs} \\ 0 & \text{if } A \text{ does not occur} \end{cases}$$

**Properties**:
- $I_A^2 = I_A$
- $I_A \cdot I_B = I_{A \cap B}$
- $I_{A \cup B} = I_A + I_B - I_A \cdot I_B$

**Fundamental Bridge**:
$$E(I_A) = P(A)$$

*Extremely powerful for counting problems and linearity of expectation!*

### Conditional Expectation (on Events)

$$E(Y|A) = \sum_y y \cdot P(Y=y|A) \quad \text{or} \quad \int y \cdot f_{Y|A}(y)\,dy$$

This is a **number** (expectation given an event occurred).

### Adam's Law (Law of Total Expectation)

$$E(Y) = E(E(Y|X))$$

For discrete $X$:
$$E(Y) = \sum_x E(Y|X=x) \cdot P(X=x)$$

---

## 5. Variance & Covariance

### Variance

$$\text{Var}(X) = E[(X - \mu)^2] = E(X^2) - [E(X)]^2$$

where $\mu = E(X)$

**Properties**:
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X) \geq 0$ always
- $\text{SD}(X) = \sqrt{\text{Var}(X)}$

### Covariance

$$\text{Cov}(X,Y) = E[(X - E(X))(Y - E(Y))] = E(XY) - E(X)E(Y)$$

**Properties**:
- $\text{Cov}(X,X) = \text{Var}(X)$
- $\text{Cov}(X,Y) = \text{Cov}(Y,X)$
- $\text{Cov}(aX, bY) = ab \cdot \text{Cov}(X,Y)$
- If $X \perp Y$, then $\text{Cov}(X,Y) = 0$ (converse not always true)

### Correlation

$$\rho(X,Y) = \text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\text{SD}(X) \cdot \text{SD}(Y)}$$

**Properties**:
- $-1 \leq \rho(X,Y) \leq 1$
- $|\rho| = 1 \iff Y = aX + b$ for some constants $a \neq 0, b$
- $\rho = 0$ means uncorrelated (weaker than independence)

### Variance of Sums

**General formula**:
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$$

**If independent** ($X \perp Y$):
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

**For multiple variables**:
$$\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i<j} \text{Cov}(X_i, X_j)$$

### Conditional Expectation (on Random Variables)

$E(Y|X)$ is a **random variable** (function of $X$), not a number.

**Properties**:
1. If $X \perp Y$, then $E(Y|X) = E(Y)$
2. **"Taking out what's known"**: $E(h(X) \cdot W|X) = h(X) \cdot E(W|X)$
3. **Adam's Law**: $E(E(Y|X)) = E(Y)$

### Eve's Law (Law of Total Variance)

$$\text{Var}(Y) = E(\text{Var}(Y|X)) + \text{Var}(E(Y|X))$$

Interpretation: Total variance = Expected conditional variance + Variance of conditional expectation

---

## 6. Moment Generating Functions (MGF)

### Definition

$$M_X(t) = E(e^{tX})$$

(if this expectation exists for $t$ in some interval around 0)

### Properties

**Moment extraction**:
$$E(X^k) = M_X^{(k)}(0)$$

where $M_X^{(k)}$ denotes the $k$-th derivative.

**Taylor expansion**:
$$M_X(t) = \sum_{k=0}^{\infty} \frac{E(X^k)}{k!} t^k$$

**Linear transformation**: If $Y = aX + b$, then:
$$M_Y(t) = e^{bt} M_X(at)$$

**Sum of independent RVs**: If $X \perp Y$, then:
$$M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$$

**Uniqueness**: MGF uniquely determines the distribution

(If two RVs have the same MGF, they have the same distribution)

---

## 7. Joint Distributions

### Joint PMF/PDF

**Discrete**:
$$p_{X,Y}(x,y) = P(X=x, Y=y)$$

**Continuous**:
$$f_{X,Y}(x,y) = \frac{\partial^2 F_{X,Y}}{\partial x \partial y}$$

### Marginal Distributions

**From joint PMF**:
$$p_X(x) = \sum_y p_{X,Y}(x,y)$$

**From joint PDF**:
$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)\,dy$$

### Conditional Distributions

**Discrete**:
$$P(Y=y|X=x) = \frac{P(X=x, Y=y)}{P(X=x)} = \frac{p_{X,Y}(x,y)}{p_X(x)}$$

**Continuous**:
$$f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

### Independence

$X$ and $Y$ are independent if and only if:
$$p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y) \quad \text{for all } x,y$$
or
$$f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y) \quad \text{for all } x,y$$

---

## 8. Transformations

### One Variable

If $Y = g(X)$ where $g$ is differentiable and strictly monotonic:
$$f_Y(y) = f_X(g^{-1}(y)) \left|\frac{d}{dy} g^{-1}(y)\right|$$

### Two Variables (Jacobian)

If $(U,V) = g(X,Y)$ is a one-to-one transformation with inverse $(X,Y) = h(U,V)$:
$$f_{X,Y}(x,y) = f_{U,V}(u,v) \left|\det J\right|$$

where $J = \begin{pmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\ \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{pmatrix}$ is the Jacobian matrix.

### Convolution

For independent continuous RVs $X$ and $Y$, the PDF of $Z = X + Y$ is:
$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z-x)\,dx$$

---

## 9. Order Statistics

Given $n$ i.i.d. random variables $X_1, \ldots, X_n$, the **order statistics** are:
$$X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$$

where $X_{(1)} = \min$, $X_{(n)} = \max$.

### CDF of $k$-th Order Statistic

$$F_{X_{(k)}}(x) = \sum_{j=k}^n \binom{n}{j} [F_X(x)]^j [1-F_X(x)]^{n-j}$$

### PDF of $k$-th Order Statistic

$$f_{X_{(k)}}(x) = n \binom{n-1}{k-1} [F_X(x)]^{k-1} [1-F_X(x)]^{n-k} f_X(x)$$

### Special Case: Uniform Order Statistics

If $U_1, \ldots, U_n \sim \text{Unif}(0,1)$ i.i.d., then:
$$U_{(j)} \sim \text{Beta}(j, n-j+1)$$

---

## 10. Discrete Distributions

### Bernoulli: $\text{Bern}(p)$

**Story**: Single trial with success probability $p$

**PMF**: $P(X=1) = p$, $P(X=0) = 1-p$

**Parameters**: $p \in [0,1]$

**Moments**:
- $E(X) = p$
- $\text{Var}(X) = p(1-p)$

**MGF**: $M(t) = 1-p + pe^t$

### Binomial: $\text{Bin}(n,p)$

**Story**: Number of successes in $n$ independent Bernoulli$(p)$ trials

**PMF**: $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$, $k \in \{0,1,\ldots,n\}$

**Parameters**: $n \in \{1,2,\ldots\}$, $p \in [0,1]$

**Moments**:
- $E(X) = np$
- $\text{Var}(X) = np(1-p)$

**MGF**: $M(t) = (1-p + pe^t)^n$

**Properties**:
- $n - X \sim \text{Bin}(n, 1-p)$ (successes ↔ failures)
- Sum: $\text{Bin}(n_1,p) + \text{Bin}(n_2,p) = \text{Bin}(n_1+n_2, p)$
- Poisson approximation: $\text{Bin}(n,p) \approx \text{Pois}(np)$ when $n$ large, $p$ small
- Normal approximation: $\text{Bin}(n,p) \approx N(np, np(1-p))$ when $n$ large

### Geometric: $\text{Geom}(p)$

**Story**: Number of failures before first success

**PMF**: $P(X=k) = (1-p)^k p$, $k \in \{0,1,2,\ldots\}$

**Parameters**: $p \in (0,1]$

**Moments**:
- $E(X) = \frac{1-p}{p}$
- $\text{Var}(X) = \frac{1-p}{p^2}$

**Note**: Some definitions count trials until first success (starting from 1)

### Negative Binomial: $\text{NBin}(r,p)$

**Story**: Number of failures before $r$-th success

**PMF**: $P(X=n) = \binom{n+r-1}{r-1} p^r (1-p)^n$, $n \in \{0,1,2,\ldots\}$

**Parameters**: $r \in \{1,2,\ldots\}$, $p \in (0,1]$

**Moments**:
- $E(X) = \frac{r(1-p)}{p}$
- $\text{Var}(X) = \frac{r(1-p)}{p^2}$

**Properties**:
- $\text{NBin}(1,p) = \text{Geom}(p)$
- Sum: $\text{NBin}(r_1,p) + \text{NBin}(r_2,p) = \text{NBin}(r_1+r_2, p)$

### Hypergeometric: $\text{HGeom}(w,b,n)$

**Story**: Sampling without replacement. In population of $w$ "success" objects and $b$ "failure" objects, draw $n$ objects. $X$ = number of successes in sample.

**PMF**: $P(X=k) = \frac{\binom{w}{k}\binom{b}{n-k}}{\binom{w+b}{n}}$

**Parameters**: $w,b \in \{0,1,\ldots\}$, $n \in \{1,\ldots,w+b\}$

**Moments**:
- $E(X) = \frac{nw}{w+b}$

**Capture-recapture**: Tag $n$ animals, return to population of size $N$. Later capture $m$ animals. Number of tagged animals in second sample $\sim \text{HGeom}(n, N-n, m)$.

### Poisson: $\text{Pois}(\lambda)$

**Story**: Number of rare events occurring in a fixed time/space interval at rate $\lambda$

**PMF**: $P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!}$, $k \in \{0,1,2,\ldots\}$

**Parameters**: $\lambda > 0$

**Moments**:
- $E(X) = \lambda$
- $\text{Var}(X) = \lambda$

**MGF**: $M(t) = e^{\lambda(e^t - 1)}$

**Properties**:
- Sum: $\text{Pois}(\lambda_1) + \text{Pois}(\lambda_2) = \text{Pois}(\lambda_1 + \lambda_2)$
- Conditioning: If $X \sim \text{Pois}(\lambda_1)$, $Y \sim \text{Pois}(\lambda_2)$ independent, then $X|(X+Y=n) \sim \text{Bin}(n, \frac{\lambda_1}{\lambda_1+\lambda_2})$
- **Chicken-egg**: Pois$(\lambda)$ trials, each success w.p. $p$ independently. Then successes $\sim$ Pois$(\lambda p)$, failures $\sim$ Pois$(\lambda(1-p))$, and they're independent.

---

## 11. Continuous Distributions

### Uniform: $\text{Unif}(a,b)$

**Story**: Equally likely to be anywhere in $[a,b]$

**PDF**: $f(x) = \frac{1}{b-a}$ for $x \in [a,b]$, else 0

**Parameters**: $a < b$

**Moments**:
- $E(X) = \frac{a+b}{2}$
- $\text{Var}(X) = \frac{(b-a)^2}{12}$

**Property**: Probability of interval proportional to its length

### Normal/Gaussian: $N(\mu, \sigma^2)$ ⭐⭐⭐

**Story**: Natural continuous variation; arises from CLT

**PDF**: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

**Parameters**: $\mu \in \mathbb{R}$, $\sigma^2 > 0$

**Moments**:
- $E(X) = \mu$
- $\text{Var}(X) = \sigma^2$

**MGF**: $M(t) = \exp(\mu t + \frac{\sigma^2 t^2}{2})$

**Standard Normal**: $Z \sim N(0,1)$ has CDF denoted $\Phi(z)$

**Standardization**: If $X \sim N(\mu, \sigma^2)$, then $\frac{X-\mu}{\sigma} \sim N(0,1)$

**Properties**:
- Linear transformation: $aX + b \sim N(a\mu + b, a^2\sigma^2)$
- Sum: $N(\mu_1, \sigma_1^2) + N(\mu_2, \sigma_2^2) = N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$
- 68-95-99.7 rule: $P(\mu-\sigma < X < \mu+\sigma) \approx 0.68$, etc.

### Exponential: $\text{Expo}(\lambda)$

**Story**: Waiting time for first event in Poisson process with rate $\lambda$

**PDF**: $f(x) = \lambda e^{-\lambda x}$ for $x > 0$, else 0

**Parameters**: $\lambda > 0$

**Moments**:
- $E(X) = \frac{1}{\lambda}$
- $\text{Var}(X) = \frac{1}{\lambda^2}$

**MGF**: $M(t) = \frac{\lambda}{\lambda - t}$ for $t < \lambda$

**Memoryless property**: $P(X > s+t | X > s) = P(X > t)$

(Only continuous memoryless distribution)

**Properties**:
- Min: $\min(X_1, \ldots, X_k) \sim \text{Expo}(\lambda_1 + \cdots + \lambda_k)$ if $X_i \sim \text{Expo}(\lambda_i)$ independent

### Gamma: $\text{Gamma}(a, \lambda)$

**Story**: Waiting time for $a$-th event in Poisson process with rate $\lambda$

**PDF**: $f(x) = \frac{1}{\Gamma(a)} (\lambda x)^a e^{-\lambda x} \frac{1}{x}$ for $x > 0$

where $\Gamma(a) = \int_0^{\infty} x^{a-1} e^{-x}\,dx$ and $\Gamma(n) = (n-1)!$ for integer $n$

**Parameters**: $a > 0$ (shape), $\lambda > 0$ (rate)

**Moments**:
- $E(X) = \frac{a}{\lambda}$
- $\text{Var}(X) = \frac{a}{\lambda^2}$

**MGF**: $M(t) = \left(\frac{\lambda}{\lambda-t}\right)^a$ for $t < \lambda$

**Properties**:
- $\text{Gamma}(1, \lambda) = \text{Expo}(\lambda)$
- Sum: $\text{Gamma}(a_1, \lambda) + \text{Gamma}(a_2, \lambda) = \text{Gamma}(a_1+a_2, \lambda)$

### Beta: $\text{Beta}(a,b)$

**Story**: Conjugate prior for Binomial; models probabilities/proportions

**PDF**: $f(x) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} x^{a-1}(1-x)^{b-1}$ for $x \in (0,1)$

**Parameters**: $a > 0$, $b > 0$

**Moments**:
- $E(X) = \frac{a}{a+b}$
- $\text{Var}(X) = \frac{ab}{(a+b)^2(a+b+1)}$

**Properties**:
- $\text{Beta}(1,1) = \text{Unif}(0,1)$
- **Bank-post office**: If $X \sim \text{Gamma}(a,\lambda)$, $Y \sim \text{Gamma}(b,\lambda)$ independent, then:
  - $\frac{X}{X+Y} \sim \text{Beta}(a,b)$
  - $X+Y \perp \frac{X}{X+Y}$
- Uniform order statistics: $U_{(j)} \sim \text{Beta}(j, n-j+1)$ for $U_i \sim \text{Unif}(0,1)$

### Chi-Square: $\chi^2_n$

**Story**: Sum of squares of $n$ independent $N(0,1)$ random variables

**Definition**: $\chi^2_n = \text{Gamma}(n/2, 1/2)$

**Parameters**: $n \in \{1,2,\ldots\}$ (degrees of freedom)

**Moments**:
- $E(X) = n$
- $\text{Var}(X) = 2n$

**Property**: Sum of $\chi^2_{n_1} + \chi^2_{n_2} = \chi^2_{n_1+n_2}$

---

## 12. Multivariate Distributions

### Multinomial: $\text{Mult}_k(n, \mathbf{p})$

**Story**: Extension of Binomial to $k$ categories. Toss $n$ items into $k$ bins with probabilities $\mathbf{p} = (p_1, \ldots, p_k)$ where $\sum p_i = 1$.

**PMF**: $P(\mathbf{X} = \mathbf{n}) = \frac{n!}{n_1! \cdots n_k!} p_1^{n_1} \cdots p_k^{n_k}$

where $\mathbf{X} = (X_1, \ldots, X_k)$, $\mathbf{n} = (n_1, \ldots, n_k)$, and $\sum n_i = n$

**Marginals**: $X_i \sim \text{Bin}(n, p_i)$

**Covariances**:
- $\text{Var}(X_i) = np_i(1-p_i)$
- $\text{Cov}(X_i, X_j) = -np_i p_j$ for $i \neq j$

**Lumping property**: Combining categories preserves Multinomial distribution

### Multivariate Normal: $\text{MVN}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$

**Definition**: Random vector $\mathbf{X}$ is MVN if every linear combination $\mathbf{t}^T\mathbf{X}$ is Normally distributed

**Parameters**:
- Mean vector $\boldsymbol{\mu}$
- Covariance matrix $\boldsymbol{\Sigma}$ (positive semidefinite)

**Properties**:
- Any subvector is MVN
- Uncorrelated components in MVN are independent
- Linear transformation: $A\mathbf{X} + \mathbf{b} \sim \text{MVN}(A\boldsymbol{\mu} + \mathbf{b}, A\boldsymbol{\Sigma}A^T)$

**Bivariate Normal** with $N(0,1)$ marginals and correlation $\rho$:
$$f(x,y) = \frac{1}{2\pi\tau} \exp\left(-\frac{x^2 + y^2 - 2\rho xy}{2\tau^2}\right)$$
where $\tau = \sqrt{1-\rho^2}$

---

## 13. Poisson Process

**Definition**: Random process with arrivals at rate $\lambda$ per unit time satisfying:
1. Number of arrivals in $[0,t]$ is $N(t) \sim \text{Pois}(\lambda t)$
2. Numbers of arrivals in disjoint time intervals are independent

**Count-Time Duality**:
- $N(t) \sim \text{Pois}(\lambda t)$ (number of arrivals by time $t$)
- $T_1 \sim \text{Expo}(\lambda)$ (time until first arrival)
- Interarrival times are i.i.d. $\text{Expo}(\lambda)$

---

## 14. Markov Chains

**Markov Property**: Future independent of past given present
$$P(X_{n+1} = j | X_0, X_1, \ldots, X_n = i) = P(X_{n+1} = j | X_n = i)$$

**Transition Matrix** $Q$: $q_{ij} = P(X_{n+1} = j | X_n = i)$
- Rows sum to 1
- $(Q^m)_{ij} = P(X_m = j | X_0 = i)$ (m-step transition probability)

**State Classification**:
- **Recurrent**: Will return with probability 1
- **Transient**: Positive probability of never returning
- **Periodic**: State has period $k > 1$
- **Aperiodic**: State has period 1

**Chain Properties**:
- **Irreducible**: Can reach any state from any other state
- **Reversible**: $\pi_i q_{ij} = \pi_j q_{ji}$ for some $\pi$

**Stationary Distribution** $\boldsymbol{\pi}$:
$$\boldsymbol{\pi} Q = \boldsymbol{\pi}$$

Interpretation: Long-run proportion of time in each state

**Finding Stationary Distribution**:
1. Solve $(Q^T - I)\boldsymbol{s} = \boldsymbol{0}$ with $\sum s_i = 1$
2. Use reversibility: $\pi_i q_{ij} = \pi_j q_{ji}$ for all $i,j$
3. For random walk on graph: $\pi_i \propto \text{degree}(i)$

**Convergence**: For irreducible, aperiodic chain, $Q^n \to$ matrix with all rows equal to $\boldsymbol{\pi}$

---

## 15. Limit Theorems

### Law of Large Numbers (LLN)

For i.i.d. $X_1, X_2, \ldots$ with $E(X_i) = \mu$:
$$\bar{X}_n = \frac{X_1 + \cdots + X_n}{n} \xrightarrow{P} \mu \quad \text{as } n \to \infty$$

**Strong LLN**: Convergence with probability 1

### Central Limit Theorem (CLT) ⭐⭐⭐

For i.i.d. $X_1, X_2, \ldots$ with $E(X_i) = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$:
$$\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{D} N(0,1) \quad \text{as } n \to \infty$$

**Equivalently** (for sample mean):
$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{D} N(0,1)$$

**Practical form**: For large $n$,
$$\sum_{i=1}^n X_i \approx N(n\mu, n\sigma^2)$$
$$\bar{X}_n \approx N\left(\mu, \frac{\sigma^2}{n}\right)$$

---

## 16. Inequalities

### Cauchy-Schwarz Inequality
$$|E(XY)| \leq \sqrt{E(X^2)E(Y^2)}$$

**Special case**: $|E(XY)| \leq \sqrt{\text{Var}(X)\text{Var}(Y)} + |E(X)E(Y)|$

### Markov's Inequality
For non-negative $X$ and $a > 0$:
$$P(X \geq a) \leq \frac{E(X)}{a}$$

### Chebyshev's Inequality
For $X$ with $E(X) = \mu$, $\text{Var}(X) = \sigma^2$, and $a > 0$:
$$P(|X - \mu| \geq a) \leq \frac{\sigma^2}{a^2}$$

**Equivalent form** (in terms of standard deviations):
$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

### Jensen's Inequality
If $g$ is **convex**:
$$E(g(X)) \geq g(E(X))$$

If $g$ is **concave**:
$$E(g(X)) \leq g(E(X))$$

**Examples**:
- $E(X^2) \geq [E(X)]^2$ (since $g(x)=x^2$ is convex)
- $E(\log X) \leq \log E(X)$ (since $\log$ is concave)

---

## 17. Distribution Relationships & Convolutions

### Sum of Independent Random Variables

| Distribution 1 | Distribution 2 | Sum |
|---|---|---|
| $\text{Pois}(\lambda_1)$ | $\text{Pois}(\lambda_2)$ | $\text{Pois}(\lambda_1+\lambda_2)$ |
| $\text{Bin}(n_1, p)$ | $\text{Bin}(n_2, p)$ | $\text{Bin}(n_1+n_2, p)$ |
| $\text{Gamma}(a_1, \lambda)$ | $\text{Gamma}(a_2, \lambda)$ | $\text{Gamma}(a_1+a_2, \lambda)$ |
| $\text{NBin}(r_1, p)$ | $\text{NBin}(r_2, p)$ | $\text{NBin}(r_1+r_2, p)$ |
| $N(\mu_1, \sigma_1^2)$ | $N(\mu_2, \sigma_2^2)$ | $N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$ |
| $\chi^2_{n_1}$ | $\chi^2_{n_2}$ | $\chi^2_{n_1+n_2}$ |

### Special Cases & Relationships

- $\text{Bern}(p) = \text{Bin}(1, p)$
- $\text{Expo}(\lambda) = \text{Gamma}(1, \lambda)$
- $\text{Geom}(p) = \text{NBin}(1, p)$
- $\text{Unif}(0,1) = \text{Beta}(1,1)$
- $\chi^2_n = \text{Gamma}(n/2, 1/2)$

---

## 18. Key Mathematical Formulas

**Geometric Series**:
$$\sum_{k=0}^{n-1} r^k = \frac{1-r^n}{1-r}$$

If $|r| < 1$:
$$\sum_{k=0}^{\infty} r^k = \frac{1}{1-r}$$

**Exponential**:
$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = \lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n$$

**Gamma Function**:
$$\Gamma(t) = \int_0^{\infty} x^{t-1} e^{-x}\,dx$$
$$\Gamma(n) = (n-1)! \quad \text{for integer } n \geq 1$$

**Beta Integral**:
$$\int_0^1 x^{a-1}(1-x)^{b-1}\,dx = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$$

**Stirling's Approximation**:
$$n! \approx \sqrt{2\pi n}\left(\frac{n}{e}\right)^n$$

---

## 19. Problem-Solving Strategies ⭐⭐⭐

### General Approach
1. **Define clearly**: Write down events and RVs with notation
2. **Sanity checks**: Try simple/extreme cases
3. **Draw pictures**: Venn diagrams, probability trees, timelines
4. **Use symmetry**: Especially powerful for i.i.d. variables

### For Probability Problems
- Try **counting**, **complements**, **conditioning**, or **LOTP**
- Check if events are **independent**
- Look for patterns matching **Bayes' Rule**

### For Distribution Problems
- Check **support** (where is PDF/PMF nonzero?)
- Look for **story** matching named distributions
- Use **properties** (memoryless → Exponential/Geometric)

### For Expectation
- Use **named distributions** if possible
- Apply **LOTUS** for functions of RVs
- Use **indicator variables** for counting
- Apply **linearity** (works even with dependence!)
- Use **Adam's Law** for conditional expectation

### For Variance
- Use **Var$(X) = E(X^2) - [E(X)]^2$**
- Apply **covariance** properties for sums
- Use **indicators** for complex counting
- Use **Eve's Law** for conditional variance

### Special Tricks
- **All orderings equally likely**: For i.i.d. continuous RVs, all orderings have equal probability $1/n!$
- **Symmetry**: $E(X_1) = E(X_2) = \cdots = E(X_n)$ for i.i.d., so $E(X_i) = E(\bar{X}) = \mu$

---

## 20. Common Pitfalls ("Biohazards") ⚠️

1. **Don't misuse naive definition**: Only applies when outcomes are equally likely

2. **Don't confuse probability types**:
   - $P(A|B)$ vs $P(B|A)$ (prosecutor's fallacy)
   - Joint vs conditional vs marginal

3. **Don't assume independence**: Need justification!
   - Uncorrelated $\not\Rightarrow$ independent (except for MVN)
   - Pairwise independence $\not\Rightarrow$ mutual independence

4. **Don't forget sanity checks**:
   - Probabilities must be in $[0,1]$
   - Variances must be $\geq 0$
   - PMF/PDF must sum/integrate to 1

5. **Don't confuse RVs, values, and events**:
   - $X$ is a random variable
   - $x$ is a realization/value
   - $\{X = x\}$ is an event

6. **Don't confuse distribution with RV**:
   - "$X \sim N(0,1)$" doesn't mean $X = N(0,1)$

7. **Don't pull nonlinear functions out of $E$**:
   - $E(g(X)) \neq g(E(X))$ in general (unless $g$ is linear)
   - $E(XY) \neq E(X)E(Y)$ unless independent
   - $\text{Var}(X+Y) \neq \text{Var}(X) + \text{Var}(Y)$ unless uncorrelated

8. **Don't forget conditioning changes the sample space**:
   - $P(\cdot|B)$ is a valid probability measure on reduced space

---

## Interview Tips by Topic ⭐⭐⭐

### Most Frequently Tested
1. **Bayes' Rule** - Medical tests, false positives, A/B testing
2. **Conditional Probability** - Dependency scenarios, real-world applications
3. **Named Distributions** - Recognizing stories, choosing correct distribution
4. **CLT** - Why means are normal, sample size calculations
5. **Independence vs Correlation** - Critical distinction

### Quick Distribution Selection
- **Binomial**: Fixed trials, counting successes
- **Geometric**: Trials until first success
- **Poisson**: Rare events, arrivals per time
- **Exponential**: Time until event, memoryless
- **Normal**: Natural variation, CLT applies
- **Uniform**: Equally likely over interval
- **Beta**: Modeling probabilities (0 to 1)

### Red Flags in Problems
- "Independent" → Can multiply probabilities, variances add
- "Without replacement" → Hypergeometric, not Binomial
- "Memoryless" → Exponential (continuous) or Geometric (discrete)
- "Rate per time" → Poisson process
- "Large $n$" → CLT might apply
- "Given that" → Conditional probability

### Common Interview Patterns
- Medical test → Bayes' Rule with sensitivity/specificity
- Coin flips → Binomial or Geometric
- Customer arrivals → Poisson process
- A/B test → Normal approximation to Binomial, hypothesis testing
- Lifetime/duration → Exponential
- Success probability estimation → Beta-Binomial conjugacy

---

**Complete Reference**: This cheatsheet covers all major topics from Harvard's Stat 110. For full derivations and deeper intuition, see the original course materials.
