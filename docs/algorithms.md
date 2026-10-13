# Memo-exip Algorithms and Mathematical Formulations

## 1. Temporal Intent Classification

Queries are classified into discrete temporal modes $I \in \{\text{CURRENT}, \text{HISTORICAL}, \text{AT\_TIME}, \text{TIMELINE}, \text{GENERAL}\}$ using deterministic regular expression matching:

$$\text{Intent}(q) = \begin{cases}
\text{TIMELINE} & \text{if } q \text{ matches history, progression, timeline patterns} \\
\text{AT\_TIME} & \text{if } q \text{ contains explicit date or day patterns} \\
\text{HISTORICAL} & \text{if } q \text{ matches previously, earlier, past, was without present cues} \\
\text{CURRENT} & \text{if } q \text{ matches currently, now, latest, present, or present tense} \\
\text{GENERAL} & \text{otherwise}
\end{cases}$$

---

## 2. Hybrid Retrieval and Temporal Intent Scoring

Given candidate record $m = (r, v, s)$ retrieved via vector similarity $S_{\text{sim}}(q, m) \in [0, 1]$, the temporal alignment score $S_{\text{temp}}(m, I, t_{\text{ref}})$ is computed:

### For Current Queries ($I = \text{CURRENT}$):
$$S_{\text{temp}}(m, \text{CURRENT}) = \begin{cases}
1.00 & \text{if } status(m) = \text{CURRENT} \\
0.40 & \text{if } status(m) = \text{DISPUTED} \\
0.05 & \text{if } status(m) = \text{SUPERSEDED} \quad \text{(severe penalty against obsolete facts)} \\
0.10 & \text{otherwise}
\end{cases}$$

### For Historical Queries ($I = \text{HISTORICAL}$):
$$S_{\text{temp}}(m, \text{HISTORICAL}) = \begin{cases}
1.00 & \text{if } status(m) = \text{SUPERSEDED} \\
0.30 & \text{if } status(m) = \text{CURRENT} \\
0.50 & \text{otherwise}
\end{cases}$$

### For Point-in-Time Queries ($I = \text{AT\_TIME}$ with reference time $t_{\text{ref}}$):
$$S_{\text{temp}}(m, \text{AT\_TIME}, t_{\text{ref}}) = \begin{cases}
1.00 & \text{if } valid\_from(m) \le t_{\text{ref}} < valid\_until(m) \\
0.05 & \text{otherwise}
\end{cases}$$

---

## 3. Utility-Based Context-Budget Knapsack Selection

Given candidate set $\mathcal{C} = \{c_1, \dots, c_n\}$, token budget $B$, and prompt overhead $B_{\text{overhead}}$:

### 3.1 Base Utility Formulation
$$U(c_i, q) = w_{\text{sim}} \cdot S_{\text{sim}}(c_i, q) + w_{\text{temp}} \cdot S_{\text{temp}}(c_i) + w_{\text{imp}} \cdot \text{imp}(c_i) + w_{\text{conf}} \cdot \text{conf}(c_i) - \text{Penalty}_{\text{redundancy}}$$

Default weights: $w_{\text{sim}} = 0.35, w_{\text{temp}} = 0.35, w_{\text{imp}} = 0.15, w_{\text{conf}} = 0.15$.

### 3.2 Redundancy Penalty
If an entity $(subject, attribute)$ has already been selected into the accepted candidate set $\mathcal{A}_{\text{sel}}$:
$$\text{Penalty}_{\text{redundancy}} = 0.50$$

### 3.3 Greedy Knapsack Packing
1. Estimate token consumption $t_i = \text{estimate\_tokens}(c_i)$.
2. Sort candidates by utility-per-token density:
   $$\rho_i = \frac{U(c_i, q)}{\max(1, t_i)}$$
3. Iterate in descending order of $\rho_i$:
   - If $\sum_{j \in \mathcal{A}_{\text{sel}}} t_j + t_i \le (B - B_{\text{overhead}})$, admit $c_i$.
   - Else, reject $c_i$ with reason `budget_exceeded`.

---

## 4. Adaptive Tabular Bandit Controller (UCB)

Inspired by MEMCON (Jiang et al., 2026), the optional controller maps coarse state tuples $\phi(s) = (\text{intent}, \text{mem\_size\_bin}, \text{budget\_pressure})$ to retrieval parameter presets $a \in \mathcal{A}$:

$$a_t = \arg\max_{a \in \mathcal{A}} \left[ Q(\phi(s_t), a) + c \sqrt{\frac{\ln \sum_{a'} N(\phi(s_t), a')}{N(\phi(s_t), a)}} \right]$$

Online sample updates are performed with learning rate $\alpha = 0.1$:
$$Q(\phi(s_t), a_t) \leftarrow Q(\phi(s_t), a_t) + \alpha \left( R_t - Q(\phi(s_t), a_t) \right)$$
