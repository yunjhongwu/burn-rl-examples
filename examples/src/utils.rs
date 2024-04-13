use burn::module::{Param, ParamId};
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_rl::base::{Agent, ElemType, Environment};

pub fn demo_model<E: Environment>(agent: impl Agent<E>) {
    let mut env = E::new(true);
    let mut state = env.state();
    let mut done = false;
    while !done {
        if let Some(action) = agent.react(&state) {
            let snapshot = env.step(action);
            state = *snapshot.state();
            done = snapshot.done();
        }
    }
}

fn soft_update_tensor<const N: usize, B: Backend>(
    this: &Param<Tensor<B, N>>,
    that: &Param<Tensor<B, N>>,
    tau: ElemType,
    tag: impl Into<ParamId>,
) -> Param<Tensor<B, N>> {
    let that_weight = that.val();
    let this_weight = this.val();
    let new_weight = this_weight * (1.0 - tau) + that_weight * tau;

    Param::initialized(tag.into(), new_weight)
}

pub fn soft_update_linear<B: Backend>(
    this: Linear<B>,
    that: &Linear<B>,
    tau: ElemType,
    tag: &str,
) -> Linear<B> {
    let weight = soft_update_tensor(&this.weight, &that.weight, tau, format!("{}.weight", tag));
    let bias = match (&this.bias, &that.bias) {
        (Some(this_bias), Some(that_bias)) => Some(soft_update_tensor(this_bias, that_bias, tau, format!("{}.bias", tag))),
        _ => None,
    };

    Linear::<B> { weight, bias }
}
