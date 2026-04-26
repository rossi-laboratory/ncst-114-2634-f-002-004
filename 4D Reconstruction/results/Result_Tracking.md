# Tracking Experimental Results

## 1. Evaluation Setup

To comprehensively evaluate the effectiveness of the proposed 4D reconstruction framework, we conduct experiments on a diverse set of dynamic scenes that include varying motion dynamics, occlusion levels, and environmental complexity. The dataset spans both controlled and unconstrained scenarios, ensuring that the evaluation reflects real-world deployment conditions.

We adopt **tracking success rate** as the primary evaluation metric. This metric is defined as the percentage of frames in which the system successfully maintains consistent tracking without catastrophic failure, drift, or identity loss. A frame is considered successful if the estimated state remains within a predefined tolerance threshold with respect to the ground truth trajectory.

To ensure statistical significance, each method is evaluated across multiple sequences and repeated trials with different initial conditions. The reported results thus reflect a distribution over diverse scenarios rather than a single deterministic outcome.

We compare our method against representative baselines:

- **3D Gaussian Splatting (3DGS)**: a recent high-fidelity 3D representation method with strong rendering capabilities but limited temporal modeling.
- **Neural Radiance Fields (NeRF)**: a widely adopted implicit representation that excels in static scene reconstruction.
- **Classical SLAM-based methods**: traditional geometric approaches that rely on feature matching and optimization.

All methods are evaluated under identical settings, including input data, initialization, and evaluation protocol, to ensure fair comparison.

---

## 2. Tracking Performance Distribution

<p align="center">
  <img src="./Tracking1.png" width="70%">
</p>

Figure 1 presents the distribution of tracking success rates using a **violin + box plot visualization**, which provides a comprehensive view of both central tendency and distribution characteristics.

Our method achieves a **median tracking success rate of 92.4%**, outperforming all baselines by a substantial margin. More importantly, the distribution of our method is significantly more concentrated, indicating lower variance and higher consistency across different scenes.

In contrast, baseline methods exhibit:

- Wider interquartile ranges, suggesting unstable performance
- Heavy-tailed distributions, indicating frequent failure cases
- Increased sensitivity to challenging conditions such as occlusion and fast motion

The violin plot further reveals that baseline methods often have multimodal or skewed distributions, implying that their performance varies significantly depending on scene characteristics. On the other hand, our method demonstrates a unimodal and compact distribution, suggesting robust generalization across different environments.

This result highlights a critical advantage of our approach:

> **The improvement is not limited to average performance but extends to reliability and stability across diverse conditions.**

---

## 3. Temporal Tracking Robustness

<p align="center">
  <img src="./Tracking2.png" width="70%">
</p>

Figure 2 illustrates the temporal evolution of tracking performance over extended sequences. This evaluation is particularly important, as many methods perform well in short sequences but degrade over time due to accumulated errors.

Our method maintains a consistently high tracking success rate throughout the entire sequence, with minimal fluctuations. The shaded region represents the standard deviation, showing that the performance remains stable even under long-horizon tracking.

In contrast, baseline methods exhibit several characteristic failure patterns:

- **Gradual performance decay**, indicating drift accumulation
- **High-frequency fluctuations**, reflecting sensitivity to transient disturbances
- **Abrupt drops**, corresponding to tracking failure or reinitialization

These observations suggest that baseline methods lack effective temporal modeling, leading to instability in dynamic environments.

The superior temporal stability of our method can be attributed to the explicit modeling of scene dynamics in the 4D representation. By incorporating temporal information directly into the scene representation, our approach enables consistent state estimation and reduces error accumulation over time.

---

## 4. Analysis and Insights

The experimental results reveal several important insights into the behavior of dynamic scene reconstruction methods:

### Robustness Across Scenes

The compact distribution observed in Figure 1 indicates that our method generalizes well across different scenarios. Unlike baseline methods that are sensitive to specific conditions, our approach maintains stable performance even under challenging dynamics.

### Temporal Consistency

Figure 2 demonstrates that temporal consistency is a key factor in long-horizon tracking. Methods that do not explicitly model temporal evolution tend to accumulate errors, leading to drift and eventual failure.

### Failure Mode Reduction

The reduced variance and absence of heavy tails in our method suggest that failure cases are significantly mitigated. This is particularly important for real-world applications, where occasional failures can lead to system-level breakdown.

### Role of 4D Representation

The results strongly support the hypothesis that incorporating time as a first-class dimension in scene representation is essential for dynamic environments. The 4D formulation enables:

- Continuous modeling of scene evolution
- Improved correspondence across frames
- Reduced ambiguity in motion estimation

---

## 5. Implications for Downstream Tasks

Beyond tracking performance, the improved stability and robustness of our method have important implications for downstream applications:

- **Robotic manipulation**: Stable tracking enables precise interaction with dynamic objects
- **Navigation**: Reduced drift improves long-horizon planning and localization
- **Simulation and digital twins**: Reliable temporal modeling supports physically consistent simulation

These results suggest that our approach is not only effective for tracking but also provides a strong foundation for broader embodied intelligence tasks.

---

## 6. Summary

In summary, the proposed 4D reconstruction framework achieves state-of-the-art tracking performance with a median success rate of 92.4%. More importantly, it significantly improves robustness and temporal stability compared to existing methods.

The results demonstrate that explicit temporal modeling is critical for reliable tracking in dynamic environments, and validate the effectiveness of our approach in addressing this challenge.
