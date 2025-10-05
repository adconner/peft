# Parameter Effficient Fine Tuning Ideas

We consider the problem of parameter efficient fine tuning. Taking inspiritaion
from Lora, if desired, all of the ideas presented here are capable of being
baked into the model weights, which allows for no overhead during inference for the
fine tuned model.

In the below, losses are cross entropy per token, measured in bits. Take as a
power of $2$ to obtain perplexity per token. My hyperparameter optimization
target is the train loss after three epochs. I do not address the overfitting
that therefore results, and test losses can be mostly regarded as noise.

## Observation: Half of Lora parameters are trained very slowly

Lora adapts a pretrained weight matrix $W$ corresponding to dense linear layer
$x\mapsto xW$ as $x \mapsto xW + xAB$, where $A$ and $B$ are selected with
small inner multiplication dimension $r$.
In the literature, Lora parameters are typically initialized as $A = 0$, $B =
\mathcal{N}(0, \sigma^2)$, with $\sigma$ typically taken to be 1 (as in
huggingface/peft and microsoft/LoRA). 
Interestingly, the original Lora article
initializes $B = 0$ and $A = \mathcal{N}(0,\sigma^2)$, but this detail is not
relevant here. This causes a problem: the relative learning rate of the nonzero
initialized factor $B$ will be essentially zero throughout training, and the
effect will be that only $A$ gets meaningfully learned. More precisely, suppose
hyperparameters are chosen so that training is not diverging, so in particular
in all iterations, $\nabla_A \mathcal{L} = O(1)$ and the
entries of $B$ remain $O(1)$. Then, on average the entries of $A$ grow like $O(
 \alpha \sqrt{\text{iteration number}})$, where $\alpha$ is the learning rate.
But this implies $\nabla_B \mathcal{L} = O(\alpha \sqrt{\text{iteration
number}})$, so that over the entire course of training, we expect $B$ to be
replaced with $B+\Delta B$, where $\Delta B = O(\alpha N)$, where $N$ is the
total number of iterations. As a consequence, we need $\Omega(\frac{1}{\alpha})$
iterations before we expect $B$ to be meaningfully different from a random
matrix, tens of thousands of iterations for commonly used learning rates.

### Solution: Boost effective learning rate of $B$

This is unnecessarily slow. We can address this issue by boosting the
relative learning rate of $B$ to be on the same order as that of $A$ without
changing anything else. Introduce a hyperparameter $\gamma$, and "multiply and
divide" by $\gamma$ to use $B = \mathcal{N}(0, \frac{1}{\gamma^2} )$ and
$x\mapsto xW + \gamma x AB$. For gradient descent with no learning rate
adaptation like in Adam, the effective learning rate for $B$ increases by
$\gamma^2$ and for learning rate adaptation based optimizers like Adam, the
effective learning rate increases by a factor of $\gamma$, both without
affecting the learning for $A$. In order to equalize the effective learning
rates between $A$ and $B$, we can therefore select $\gamma$ somewhere between
$\frac{1}{\sqrt{\alpha}}$ and $\frac{1}{\alpha}$.

Plot here on only this

## Other methods of parameter efficient fine tuning

Lets explore some additional ideas for parameter efficient fine tuning. Our
motivation is to construct schemes which potentially improve over existing
approaches (Lora, Dora, Tied-Lora) in dimensions such as tunable parameters
needed and capacity per parameter. Other possible impovement metrics include 
generalization , avoiding forgetfulness, qualitative assesment of the tuned
model output quality, and performance on benchmarks. For simplicity, we
currently focus on the metrics we can observe through train loss and
tunable parameters, and to a lesser extent test loss.

#### Previous work: Tied-Lora

In an effort to further reduce the number of parameters required by Lora,
Nvidia introduced Tied-Lora, where the layer adaptation is replaced with $x\mapsto
xW + x A v B u$, where $u$ and $v$ are vectors whose action is pointwise
multiplication, and $A$ and $B$ are tied between all model layers. More
precisely, let $W_i$ be the weight matrix for corresponding to the same element
as $i$ ranges through the $\ell$ model layers. In layer $i$, Tied-Lora adapts
$x\mapsto x W_i$ as $x\mapsto x W_i + x A v_i B u_i$.

### New idea: Partial weight tying

One general class of ideas to generate different parameter efficient fine
tuning schemes can be considered different schemes for tying or partially tying
parameters between different model layers, just as in Tied-Lora. The main new
idea is that of partial weight tying, intended to be intermediate between
fully tied and fully independent weights per model layer. For $i$ running over
model layers, an $m\times n$ matrix $A_i$ is *partially tied* if it is selected
from a tied low dimensional (dimension $k$, say) space of possible $A$'s.
Explicitly, there is globally tied tensor $L$ of shape $(k,m,n)$ representing
the linear space, and each layer has a vector $w_i$ of shape $(k,)$ determining
its selection of $A_i$ from $L$, obtained as $$A_i = L(w_i) =
\operatorname{einsum}(\text{`kmn,k->mn'},L,w_i).$$

#### Partially Tied Lora

Lets apply the partial tying to both the $A$ and $B$ factors of Lora. Let $L_A$
and $L_B$ be $d_A$ and $d_B$ dimensional subpsaces of matrices of the same
shape as $A$ and $B$ respectively, represented as tensors of shapes
$(\ell_A,\text{in dimension},r)$, and $(\ell_B,r,\text{out dimension})$,
respectively. Then $W_i$ is adapted as $x\to x W_i + x L_A(w_i^A) L_B(w_i^B)$,
where $w_i^A$ and $w_i^B$ are vectors of shape $(\ell_A,)$ and $(\ell_B,)$
respectively. The idea is to obtain parameter efficiency by picking $\ell_A, \ell_B <
\ell$, the number of layers. Optionally, we can introduce vectors $u_i$ of shape $(r,)$, and
$v_i$ of shape $(\text{out dimension},)$ for each layer as in Tied-Lora to use
the adaptation $x\to x W_i + x L_A(w_i^A) v_i L_B(w_i^B) u_i$. Then when
$\ell_A=\ell_B=1$, we recover a scheme as expressive as Tied-Lora, and when
$\ell_A=\ell_B=\ell$, we recover a scheme as expressive as Lora.

#### Tied Lora Extra

In Tied Lora, replace the vector $v_i$ of shape $(r,)$ with a matrix $M_i$ of
shape $(r_1,r_2)$, and optionally drop $u_i$. Since $r$ is small, this represents a
small increase in number of parameters. Explicitly, $W_i$ as adapted to
$x\mapsto xW_i + x A M_i B$, with $M_i$ of shape $(r_1,r_2)$. Typically, we
will take $r_1 = r_2$.

#### Tensor Embedding Adaptation

In Tied Lora Extra, partially tie the $M_i$'s between the layers, say $M_i$ is
selected from linear space $L$, represented as a tensor of shape $(\ell_1,
r_1,r_2)$, with the selection given by vector $w_i$ of shape $(\ell_1,)$.
Explicitly, $x\mapsto x W_i + x A L(w_i) B$.

As a remark, this was actually my first idea, inspired by thinking of the $W_i$
stacked into an $(\ell,\text{in dimension},\text{out dimension})$ tensor, and
adapting this with a smaller $(\ell_1,r_2,r_2)$ tensor embedded into the space.
Tied-Lora-Extra is the special case where $\ell_1 = \ell$.

#### Previous Work: Dora

Observe that $W = \frac{W} {\lVert W \rVert_r} \lVert W
\rVert_r$, where $\lVert W\rVert_r$ is the vector of row norms of $W$, and
division and multiplication by vectors is pointwise. Dora introduces a
parameter $w_i = \lVert W \rVert_r$ and ordinary lora parameters $A$ and $B$,
and adapts $W_i$ as 
$$x\mapsto \frac{xW+xAB}{\lVert W + AB \rVert} w_i.$$
In practice, the denominator is detached to save memory during the backward pass.

#### Simple Dora

Similarly factor $W = W^0 w$, where $W^0 = \frac{W}{\lVert W
\rVert_r}$ has rows of unit length and $w = \lVert W \rVert_r$ and the
multiplication is pointwise. Let $W^0$ be untrainable, $w$ be trainable, and
introduce trainable $A$ and $B$ and adapt $W$ as $x\mapsto (xW + xAB) w$.
We aim to retain some power of Dora while avoiding the step of materializing
the matrix $W+AB$ during the forward pass.

### Results

{{ model_select }} {{ best_select }} 

{{ peft_select }}

{{ main_plots }}

