---
title: Interpreting the Diffusion Policy Loss Function
---
[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137) ("Original Diffusion Policy Paper")

> "As shown in Ho et al. (2020), minimizing the loss function in Eq 3 also  minimizes the variational lower bound of the KL-divergence between the data distribution $p(x_0)$ and the distribution of samples drawn from the DDPM $q(x_0)$ using Eq 1."

What does this mean...? I wanted to understand intuitively, so I derived this claim...

[The actual diffusion policy paper might be helpful background]

- Formulation: $x_t^{k-1} = \alpha(x_t^k - \gamma \epsilon_\theta(x_t^k, \text{obs}_t, k) + \mathcal{N}(0, \sigma^2 I))$ (Eq 1)
	- ${}_t$ represents timestep
	- $\epsilon_\theta$ is the noise prediction network, learned params $\theta$
		- $\text{obs}_t$ is observation at timestep $t$
	- $\mathcal{N}(0, \sigma^2 I)$ is the DDPM noise added at each sampling step
	- $\alpha$ empirically set slightly smaller than 1, improving stability
	- $\gamma$ set by noise schedule 
	- Note: $k$ = denoising step

- Training
	- $\epsilon_\theta$ trained with loss: $\mathcal{L} = MSE(\epsilon^k, \epsilon_\theta(x_k + \epsilon^k, k))$ (Eq 3)
		- Mathematical Interpretation:
			- $p_\theta(x_{0:K} | \text{obs}) = p_\theta(x_K | \text{obs}) \Pi_{k=1}^K p_\theta(x_{k-1} | x_k, \text{obs})$
				- i.e. the denoising process is a Markov Chain; next state; $p(x_K | \text{obs})$ is a prior; each $x_{k-1}$ then is independent of all other $x_{i}$ given $x_k$ and $\text{obs}$.
				- $p_\theta(x_{0:K} | \text{obs})$ is the joint probability density of predicting the whole chain of latent semi-denoised "images"
			- Take the marginal to get the probability density of predicting the unnoisy image:
				- $p_\theta(x_0| \text{obs}) = \int p_\theta(x_{0:K} | \text{obs}) dx_{1:K}$
					- This is an integral over all possible denoising paths $x_{1:K}$
			- We want to minimize the KL divergence between the data distribution and DDPM distribution of sampled clean images (in other words, maximize log-likelihood; this is ML estimation):
			$$
			\begin{align*}& \quad\; \arg\min_\theta D_{KL}(p_{\text{data}}(x_0| \text{obs}) \,\|\, p_\theta(x_0 | \text{obs})) \\ &= \arg\min_\theta \sum_{i=1}^N \big[ \log p_{\text{data}}(x_0^{(i)} | \text{obs}^{(i)}) - \log p_\theta(x_0^{(i)} | \text{obs}^{(i)}) \big] \\
			&= \arg\min_{\theta} \sum_{i=1}^N- \log p_\theta(x_0^{(i)} | \text{obs}^{(i)}) \end{align*}
			$$
			- Minimizing the KL Divergence directly is intractable -- 
				- Even computing the log likehood for a given $\theta$ is intractable because computing the integral in $p_\theta(x_0)$ is intractable; firstly, we are integrating over a huge neural net, not some clean math function; there's no analytical soln. But, even if you discretized and approximated the integral as a sum, you'd have to sum over all possible (discretized) denoising paths $x_{1:K}$, of which there are still tons.
			- Therefore, we minimize the (negative) *variational lower bound* (VLB) on the true log-likelihood; a computational proxy for the actual true log-likelihood:
			$$
			\begin{align*}\log p_\theta(x_0^{(i)} | \text{obs}^{(i)}) &= \log \int p_\theta(x_{0:K} | \text{obs}) dx_{1:K} \\ 
			&= \log \int q(x_{1:K}|x_0) \frac{p_\theta(x_{0:K} | \text{obs})}{q(x_{1:K}|x_0)} dx_{1:K} \\ 
			&= \log \mathbb{E}_{q(x_{1:K}|x_0)}\big[ \frac{p_\theta(x_{0:K} | \text{obs})}{q(x_{1:K}|x_0)} \big] \\
			&\ge \mathbb{E}_{q(x_{1:K}|x_0)}\big[ \log \frac{p_\theta(x_{0:K} | \text{obs})}{q(x_{1:K}|x_0)} \big]\quad\quad\quad\text{(Jensen's Inequality)}\\
			&= \mathbb{E}_{q(x_{1:K}|x_0)}\big[ \log \frac{p_\theta(x_K | \text{obs}) \Pi_{k=1}^K p_\theta(x_{k-1} | x_k, \text{obs})}{\Pi_{k=1}^K q(x_k | x_{k-1})} \big] \\
			&= \mathbb{E}_{q(x_{1:K}|x_0)}\big[ \log p_\theta(x_K | \text{obs}) + \sum_{k=1}^K \log p_\theta(x_{k-1} | x_k,\text{obs}) - \sum_{k=1}^K \log q(x_k | x_{k-1})\big] \quad\quad\quad\text{(log rules)} \\
			&= \mathbb{E}_{q(x_{1:K}|x_0)}\big[ \log p_\theta(x_K | \text{obs}) + \sum_{k=1}^K (\log p_\theta(x_{k-1} | x_k,\text{obs}) -\log q(x_k | x_{k-1}))\big] \quad\quad\quad\text{(log rules)}
			\end{align*}
			$$
			- Note that we introduce $q$ as the forward-noising Markov Chain we used to create the training data
			- At this step, we sub in (using Bayes Rule): $\log q(x_k | x_{k-1}) = \log q(x_{k-1} | x_k, x_0) + \log q(x_k | x_0) - \log q(x_{k-1} | x_0)$
				- Bayes Rule derives the analytically ideal reverse step $q(x_{k-1} | x_k, x_0)$ from the known forward-noising process. We can then reformulate the VLB in a much more interpretable way, as matching $p_\theta(x_{k-1} | x_k, \text{obs})$ with $q(x_{k-1} | x_k, x_0)$. 
				$$
				\begin{align*} &= \mathbb{E}_{q(x_{1:K}|x_0)}\big[ \log p_\theta(x_K | \text{obs}) + \sum_{k=1}^K (\log p_\theta(x_{k-1} | x_k,\text{obs}) -\big(\log q(x_{k-1} | x_k, x_0) + \log q(x_k | x_0) - \log q(x_{k-1} | x_0)\big))\big] \\
				&= \mathbb{E}_{q(x_{1:K}|x_0)}\big[ \log p_\theta(x_K | \text{obs}) + \sum_{k=1}^K (\log p_\theta(x_{k-1} | x_k, \text{obs}) - \log q(x_{k-1} | x_k, x_0) - \log q(x_k | x_0) + \log q(x_{k-1} | x_0))\big] \\ &\text{\quad (Distribute the negative sign)}\\\\ 
				&= \mathbb{E}_{q(x_{1:K}|x_0)}\bigg[ \log p_\theta(x_K | \text{obs}) + \sum_{k=1}^K \bigg(\log \frac{p_\theta(x_{k-1}|x_k, \text{obs})}{q(x_{k-1}|x_k,x_0)}\bigg) - \sum_{k=1}^K \bigg(\log q(x_k|x_0) - \log q(x_{k-1}|x_0)\bigg) \bigg] \\ 
				&\text{\quad (Group terms into log ratios and separate the sums)}\\\\ 
				&= \mathbb{E}_{q(x_{1:K}|x_0)}\bigg[ \log p_\theta(x_K | \text{obs}) + \log \frac{p_\theta(x_0|x_1, \text{obs})}{q(x_0|x_1,x_0)} + \sum_{k=2}^K \log \frac{p_\theta(x_{k-1}|x_k, \text{obs})}{q(x_{k-1}|x_k,x_0)} - \big(\log q(x_K|x_0) - \log q(x_0|x_0)\big) \bigg] \\ 
				&\text{\quad (Split the first sum to separate the k=1 reconstruction term; the second sum is a telescoping series which simplifies)}\\\\ 
				&= \mathbb{E}_{q(x_{1:K}|x_0)}\bigg[ \log p_\theta(x_K | \text{obs}) + \log p_\theta(x_0|x_1, \text{obs}) - \log q(x_0|x_1,x_0) + \sum_{k=2}^K \log \frac{p_\theta(x_{k-1}|x_k, \text{obs})}{q(x_{k-1}|x_k,x_0)} - \log q(x_K|x_0) + \log q(x_0|x_0) \bigg] \\ 
				&\text{\quad (Expand first log; distribute minus sign on the telescoped terms)}\\\\ 
				&= \mathbb{E}_{q(x_{1:K}|x_0)}\bigg[ \log p_\theta(x_0|x_1, \text{obs}) + \sum_{k=2}^K \log \frac{p_\theta(x_{k-1}|x_k, \text{obs})}{q(x_{k-1}|x_k,x_0)} + \log p_\theta(x_K | \text{obs}) - \log q(x_K|x_0) \bigg] \\ 
				&\text{\quad (Regroup/reorder the remaining terms, canceling the } \log q(x_0|x_0) \text{ terms)}\\\\ 
				&= \mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1, \text{obs})] + \sum_{k=2}^K \mathbb{E}_{q(x_0,x_{k-1},x_k|x_0)}\bigg[\log \frac{p_\theta(x_{k-1}|x_k, \text{obs})}{q(x_{k-1}|x_k,x_0)}\bigg] + \mathbb{E}_{q(x_K|x_0)}\bigg[\log \frac{p_\theta(x_K | \text{obs})}{q(x_K|x_0)}\bigg] \\ 
				&\text{\quad (Distribute the expectation. Note that each term only depends on a subset of } x_{1:K} \text{, so the expectation marginalizes out the irrelevant variables.)}\\\\ 
				&= \mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1, \text{obs})] - \sum_{k=2}^K \mathbb{E}_{q(x_k, x_0)}\bigg[ \mathbb{E}_{q(x_{k-1}|x_k, x_0)}\bigg[\log \frac{q(x_{k-1}|x_k,x_0)}{p_\theta(x_{k-1}|x_k, \text{obs})}\bigg] \bigg] - \mathbb{E}_{q(x_K|x_0)}\bigg[\log \frac{q(x_K|x_0)}{p_\theta(x_K | \text{obs})}\bigg] \\ 
				&\text{\quad (Using Law of Total Expectation, the expectation in the sum is broken into an inner expectation over } x_{k-1} \text{ and an outer one over } x_k, x_0)\\\\ 
				&= \mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1, \text{obs})] - \sum_{k=2}^K \mathbb{E}_{q(x_k,x_0)}[D_{KL}(q(x_{k-1}|x_k, x_0) \ || \ p_\theta(x_{k-1}|x_k, \text{obs}))] - D_{KL}(q(x_K|x_0) \ || \ p_\theta(x_K | \text{obs})) \\ 
				&\text{\quad (Identify the inner expectations as the definitions of KL Divergence)} \\\\
				&= - \sum_{k=1}^K \mathbb{E}_{q(x_k,x_0)}[D_{KL}(q(x_{k-1}|x_k, x_0) \ || \ p_\theta(x_{k-1}|x_k, \text{obs}))] - D_{KL}(q(x_K|x_0) \ || \ p_\theta(x_K | \text{obs})) \\
				&\text{\quad ("Fold" the first term into the } k=1 \text{ case of the sum)}
				\end{align*}
				$$
			- The last equality is non-obvious, but if we expand the $k=1$ case of the sum, we can see this:
			$$
			\begin{align*} \mathbb{E}_{q(x_k,x_0)}[D_{KL}(q(x_0|x_1, x_0) \ || \ p_\theta(x_0|x_1, \text{obs}))] &= \mathbb{E}_{q(x_k,x_0)}[D_{KL}(\delta_{x_0} \ || \ p_\theta(x_0|x_1, \text{obs}))] \\ &= \mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1, \text{obs})]\end{align*}
			$$
			- The last term of VRB, $D_{KL}(q(x_K|x_0) \ || \ p_\theta(x_K | \text{obs}))$ is also trivial and so it's dropped
				- Both are just a standard Gaussian prior; there are no parameters to learn here.
			- This yields the final objective (there are a few more steps to simplify this to the MSE loss described in the paper): $$\arg\min_\theta- \sum_{k=1}^K \mathbb{E}_{q(x_k,x_0)}[D_{KL}(q(x_{k-1}|x_k, x_0) \ || \ p_\theta(x_{k-1}|x_k, \text{obs}))]$$
			- The next key realization is that, during training: "we randomly select denoising iteration $k$ and then sample a random noise $\epsilon_k$ with appropriate variance for iteration $k$"
				- Basically, we never compute the loss for the whole denoising sequence $\sum_{1:K}$; we monte-carlo sample a single denoising step $k$ and compute the loss just for that.
				- (With enough monte-carlo samples, optimizing using the sampled per-denoising step loss should be equivalent to optimizing using the sum of losses from all denoising steps).
				- Thus, we basically optimize, for a single $k$ per step of training: 
				$$
				\arg\min_\theta- \mathbb{E}_{q(x_k,x_0)}[D_{KL}(q(x_{k-1}|x_k, x_0) \ || \ p_\theta(x_{k-1}|x_k, \text{obs}))]
				$$
			- The last key realization is that both distributions: $q(x_{k-1} | x_k, x_0)$ and $p_\theta(x_{k-1} | x_k, \text{obs})$ are Gaussian. Note that the mean here represents the semi-noised image $x_{k-1}$ and the variance is fixed by the noise scheduler.
				- $q(x_{k-1} | x_k, x_0)$ is by design of the diffusion noising process as adding Gaussian noise.
				- $p_\theta(x_{k-1} | x_k \text{obs})$ also is by design of the denoising process. 
			- The KL-Divergence between 2 Gaussians (w/ fixed, diagonal covariance) has closed-form:
			$$
			D_{KL}\!\left(\mathcal{N}(\tilde{\mu}_k, \tilde{\beta}_k I) \,\|\, \mathcal{N}(\mu_\theta, \sigma_k^2 I)\right) = \text{const} + \frac{1}{2\sigma_k^2} \left\| \tilde{\mu}_k(x_k, x_0) - \mu_\theta(x_k, k) \right\|^2
			$$ 
			- Typically, we simply drop the $\text{const} + \frac{1}{2\sigma_k^2}$ terms and minimize $\left\| \tilde{\mu}_k(x_k, x_0) - \mu_\theta(x_k, k) \right\|^2$. This works in practice, though we are no longer minimizing exactly the VLB.
				- The $\frac{1}{2\sigma_k^2}$ multiplier is effectively a weighting depending on the current denoising step; Ho et al. (original DDPM paper) find empirically that dropping this weighting is actually better.
			- However, the model doesn't predict the "image" (i.e. $\mu_\theta$) directly -- it predicts the noise vector between $x_k$ and $x_{k-1}$. Thus, we must take one final step to convert the MSE error of $\mu$ to an MSE error of the noise vectors $\epsilon$ predicted by the network.
				- In particular, Diffusion Policy uses a "Variance-Preserving" Noise Schedule that is applied like so: $$x_k = \sqrt{\bar{\alpha}_k} x_0 + \sqrt{1 - \bar{\alpha}_k}\epsilon$$
					- This implies: 
					$$
					\mu_\theta(x_k, k) = \frac{1}{\sqrt{\alpha_k}} \left(x_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \, \epsilon_\theta(x_k, \text{obs}, k)\right)
					$$
					- Then, $\arg\min_\theta \left\| \tilde{\mu}_k(x_k, x_0) - \mu_\theta(x_k, k) \right\|^2$ is equivalent to: 
					$$
					\begin{align*} 
					& \;\;\;\;\; \arg\min_\theta \left\| \frac{1}{\sqrt{\alpha_k}}\left(x_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}}\epsilon\right) - \frac{1}{\sqrt{\alpha_k}}\left(x_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}}\epsilon_\theta(x_k, \text{obs}, k)\right) \right\|^2 \\
					&= \arg\min_\theta \frac{1}{\alpha_k} \left\| \left( x_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}}\epsilon \right) - \left( x_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}}\epsilon_\theta(x_k, \text{obs}, k) \right) \right\|^2 \\
					&= \arg\min_\theta \frac{1}{\alpha_k} \left\| -\frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}}\epsilon + \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}}\epsilon_\theta(x_k, \text{obs}, k) \right\|^2 \\
					&= \arg\min_\theta \frac{1}{\alpha_k} \left( \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \right)^2 \left\| \epsilon_\theta(x_k, \text{obs}, k) - \epsilon \right\|^2 \\
					&= \arg\min_\theta \left\| \epsilon_\theta(x_k, \text{obs}, k) - \epsilon \right\|^2
					\end{align*}
					$$
					- This is exactly the MSE loss proposed in the paper.
	- Notes:
		- This is a skewed way of looking at it, but: diffusion-policy training is kind like the EM algorithm with a trivial E-step. We already know the ground-truth $q(\cdot | x_0)$ used to generate the training data; so there is no need to optimize $q$ as we do in EM. Rather, we just perform the M-step of optimizing $\theta$.
		- The VLB becomes tighter as the model better learns to approximate $q(\cdot | x_0)$ using $p_\theta(\cdot | x_0, \text{obs})$; this is not obvious from the math shown above; but see my 6.7810 notes on EM; $D_{KL}(q(\cdot | x_0) \| p_\theta(\cdot | x_i, \text{obs}))$ is exactly the gap between VBL and the true log likelihood.
			- This is not true of EM in general; only true here because the ground-truth $q(\cdot | x_0)$ is known. Also, in practice, in EM, we often restrict $q(\cdot | x_0)$ to something simple like a Gaussian, even though the ground-truth $q(\cdot | x_0)$ may not be a Gaussian. Thus the VLB gap is not closable.
		- The underlying graphical model is a directed Markov Chain: 
		$$
		x_K \sim p_\theta(x_K \mid \text{obs}) \; (\text{often } \mathcal{N}(0, I)), \quad x_{t-1} \sim p_\theta(x_{t-1} \mid x_t, \text{obs}), \quad t = K, \ldots, 1
		$$
		- and the forward noising process can also be seen as a Markov Chain (though nothing to learn in this case): 
		$$
		x_t \sim q(x_t \mid x_{t-1}) \quad \text{with known Gaussians set by noise schedule}, \quad x_0 \sim \text{data}
		$$
		- The parameters $\theta$ live in the neural network, but represent (using a neural net) the probability distributions $p(x_{t-1} | x_t, \text{obs})$
