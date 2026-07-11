# 一. DDPM: Denoising Diffusion Probabilistic Model.

## 1.1 Forward Process

取数据样本$x_0 \sim q_{data}(x_0)$, 进行前向加噪, 即每步添加一定的噪声, 设当前步为$x_{t-1}$, 下一步为$x_t$, 一步前向转移分布为:
$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I), \beta_t \in (0, 1).
$$

定义 $\alpha_t = 1-\beta_t$, $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$, 则从$x_0$到$x_t$的$t$步前向加噪分布为
$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I), 
$$
即
$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{(1- \bar{\alpha}_t)}\epsilon, \epsilon \sim \mathcal{N}(0, I).
$$

## 1.2 Backward Process

设 $T$ 足够大，则噪声调度满足 $\bar{\alpha}_T \approx 0$，前向过程的终点分布近似为标准高斯分布：

$$
q(x_T) \approx \mathcal{N}(0,I).
$$

因此，在生成阶段，可以从标准高斯噪声开始采样：

$$
x_T \sim p(x_T) = \mathcal{N}(0,I).
$$

然后希望通过一个可学习的反向马尔可夫过程，逐步从 $x_T$ 生成 $x_{T-1},x_{T-2},\dots,x_0$。该反向生成模型的联合分布定义为

$$
p_{\theta}(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_{\theta}(x_{t-1}\mid x_t).
$$

其中每一步反向转移分布通常被参数化为高斯分布：

$$
p_{\theta}(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_{\theta}(x_t,t),
\sigma_t^2 I
\right).
$$

这里定义联合分布 $p_\theta(x_{0:T})$ 的目的是构造一个完整的反向生成过程。真正关心的是其关于 $x_0$ 的边缘分布：

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\, dx_1\cdots dx_T.
$$

训练的目标是使这个边缘分布接近真实数据分布：

$$
p_\theta(x_0) \approx q_{\mathrm{data}}(x_0).
$$

因此，扩散模型可以理解为：通过学习每一步的反向去噪转移分布 $p_\theta(x_{t-1}\mid x_t)$，最终让从高斯噪声出发的反向链生成真实数据样本。

## 1.3 DDPM Loss

DDPM 的训练目标是最大化数据样本的对数似然：

$$
\log p_\theta(x_0).
$$

但是由于

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}
$$

需要对所有隐变量 $x_1,\dots,x_T$ 积分，因此难以直接计算。于是引入前向扩散过程

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^T q(x_t\mid x_{t-1})
$$

作为变分分布。由 Jensen 不等式可得

$$
\log p_\theta(x_0)
=
\log
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right]
\geq
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

因此，负对数似然有如下上界：

$$
-\log p_\theta(x_0)
\leq
\mathcal{L}_{\mathrm{VLB}},
$$

其中

$$
\mathcal{L}_{\mathrm{VLB}}
=
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\Vert
p(x_T)
\right)
+
\sum_{t=2}^T
\mathbb{E}_{q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\Vert
p_\theta(x_{t-1}\mid x_t)
\right)
\right]
-
\mathbb{E}_{q(x_1\mid x_0)}
\left[
\log p_\theta(x_0\mid x_1)
\right].
$$

其中第一项要求前向过程终点接近标准高斯先验，第二项要求模型学习每一步真实的反向去噪分布，第三项是从 $x_1$ 重建 $x_0$ 的重建项。

由于前向过程为高斯分布, 有

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta}_t I
\right),
$$

且

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2 I
\right),
$$

因此中间的 KL 项可以转化为高斯分布之间的 KL。实际训练中常使用噪声预测参数化，将训练目标简化为

$$
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_\theta
\left(
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
t
\right)
\right\|^2
\right],
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$
