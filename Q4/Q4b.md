1. Unimodal vs Multimodal

From KDE + violin:

ρ = 2, 4 → clearly unimodal
Single sharp peak around ~ -120 to -110
Very consistent learning
ρ = 1 → slightly wider, mild secondary bump
Indicates some variability across runs
Possibly weak multimodality (few underperforming runs)
ρ = 8 → mostly unimodal but distorted
One main peak + long tail (outlier visible)

👉 Conclusion:

Mostly unimodal distributions
Slight multimodality only at low ρ (ρ=1)
🔹 2. Skewness
All distributions show left skew (long tail toward worse returns)

Especially:

ρ = 8 → strongly left-skewed
Big outlier (~ -600)
ρ = 1 → mild skew
ρ = 2, 4 → least skewed

👉 Interpretation:

Most runs perform well
Few runs fail badly → creates left tail
🔹 3. Normal-like or not?
ρ = 2, 4 → closest to normal
Symmetric-ish
Narrow spread
ρ = 1, 8 → clearly non-normal
Skewed
Heavy tails

👉 Overall:

Not perfectly Gaussian (expected in RL)
Only mid-range ρ approximates normal
🔹 4. Spread (Stability)

From box + violin:

ρ = 4 → tightest distribution
Most stable
ρ = 2 → also tight
ρ = 1 → wider
ρ = 8 → widest (due to outlier)

👉 Stability ranking:

ρ=4 ≈ ρ=2  >  ρ=1  >  ρ=8
🔹 5. Key Insight (VERY IMPORTANT)

Even though mean plots might look similar:

👉 Distribution shows:

ρ = 4 is most reliable
ρ = 8 is risky (occasional failure)
ρ = 1 is inconsistent


The performance distributions for different replay factors are predominantly unimodal, with a single peak representing consistent learning behavior across seeds. However, for lower replay factors (ρ = 1), the distribution is slightly wider and exhibits mild multimodality, indicating variability in learning outcomes across runs. All distributions exhibit left skewness, with a long tail toward lower returns, suggesting that while most runs achieve good performance, a few runs fail significantly. The distributions are not perfectly normal due to this skewness and the presence of outliers, particularly for higher replay factors such as ρ = 8. Among the tested values, ρ = 4 produces the most concentrated and symmetric distribution, indicating the most stable and reliable performance. Overall, moderate replay factors lead to more consistent learning, while very low or very high replay factors introduce variability and instability.

The presence of occasional extreme outliers (e.g., for ρ = 8) suggests that higher replay factors may lead to instability or overfitting to replay buffer samples, resulting in degraded performance in some runs.