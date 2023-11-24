use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_rl::base::{Agent, ElemType, Environment};

pub fn demo_model<E: Environment>(agent: impl Agent<E>) {
    let mut env = E::new(true);
    let mut state = env.state();
    let mut done = false;
    while !done {
        let action = agent.react(&state);
        let snapshot = env.step(action);
        state = *snapshot.state();
        done = snapshot.done();
    }
}

fn soft_update_tensor<const N: usize, B: Backend>(
    this: &Param<Tensor<B, N>>,
    that: &Param<Tensor<B, N>>,
    tau: ElemType,
) -> Param<Tensor<B, N>> {
    let that_weight = that.val();
    let this_weight = this.val();
    let new_this_weight = this_weight * (1.0 - tau) + that_weight * tau;

    Param::from(new_this_weight.no_grad())
}

pub fn soft_update_linear<B: Backend>(this: &mut Linear<B>, that: &Linear<B>, tau: ElemType) {
    this.weight = soft_update_tensor(&this.weight, &that.weight, tau);
    if let (Some(this_bias), Some(that_bias)) = (&mut this.bias, &that.bias) {
        this.bias = Some(soft_update_tensor(this_bias, that_bias, tau));
    }
}
