use crate::base::Snapshot;
use crate::base::{Action, State};
use crate::components::env::Environment;
use std::f32::consts::PI;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

#[derive(Debug, Copy, Clone)]
pub struct CartPoleState {
    data: [f32; 4],
}

impl Index<usize> for CartPoleState {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for CartPoleState {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl CartPoleState {
    pub fn new(state: [f32; 4]) -> Self {
        Self { data: state }
    }
}

impl State for CartPoleState {
    type Data = [f32; 4];
    fn data(&self) -> &Self::Data {
        &self.data
    }
}

#[allow(unused)]
#[derive(Debug, Copy, Clone)]
pub enum CartPoleAction {
    Left,
    Right,
}

impl Action for CartPoleAction {
    fn random() -> Self {
        if rand::random::<f32>() < 0.5 {
            Self::Left
        } else {
            Self::Right
        }
    }

    fn enumerate() -> Vec<Self> {
        vec![Self::Left, Self::Right]
    }
}

#[derive(Debug)]
pub struct CartPole {
    state_space: [[f32; 2]; 4],
    gravity: f32,
    pole_mass: f32,
    total_mass: f32,
    length: f32,
    polemass_length: f32,
    force_mag: f32,
    tau: f32,
    x_threshold: f32,
    theta_threshold: f32,
    state: CartPoleState,
    count: i32,
}

impl CartPole {
    #[allow(unused)]
    pub fn new() -> Self {
        let cart_mass = 1.0;
        let pole_mass = 0.1;
        Self {
            state_space: [
                [-4.8, 4.8],
                [f32::NEG_INFINITY, f32::INFINITY],
                [-0.418, 0.418],
                [f32::NEG_INFINITY, f32::INFINITY],
            ],
            gravity: 9.8,
            pole_mass,
            total_mass: cart_mass + pole_mass,
            length: 0.5,
            polemass_length: 0.1 * 0.5,
            force_mag: 10.0,
            tau: 0.02,
            x_threshold: 2.4,
            theta_threshold: 12.0 * 2.0 * PI / 360.0,
            state: CartPoleState::new([0.0, 0.0, 0.0, 0.0]),
            count: 0,
        }
    }
}

impl Environment for CartPole {
    type State = CartPoleState;
    type Action = CartPoleAction;

    fn render(&mut self) {}

    fn reset(&mut self) -> Snapshot<CartPoleState> {
        for i in 0..4 {
            self.state[i] = (rand::random::<f32>() * 0.1) - 0.05;
        }
        self.count = 0;
        Snapshot::<CartPoleState>::new(self.state, 1.0, false)
    }

    fn step(&mut self, action: CartPoleAction) -> Snapshot<CartPoleState> {
        let force = match action {
            CartPoleAction::Left => -self.force_mag,
            CartPoleAction::Right => self.force_mag,
        };
        let cos_theta = f32::cos(self.state[2]);
        let sin_theta = f32::sin(self.state[2]);
        let temp = (force + self.polemass_length * f32::powi(self.state[3], 2) * sin_theta)
            / self.total_mass;

        let pole_force = self.pole_mass * f32::powi(cos_theta, 2) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - temp * cos_theta)
            / (self.length * (4.0 / 3.0 - pole_force));
        let x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass;

        self.state[0] += self.tau * self.state[1];
        self.state[1] += self.tau * x_acc;
        self.state[2] += self.tau * self.state[3];
        if self.state[2] > PI {
            self.state[2] -= 2.0 * PI;
        } else if self.state[2] < -PI {
            self.state[2] += 2.0 * PI;
        }
        self.state[3] += self.tau * theta_acc;

        for i in 0..self.state_space.len() {
            if self.state[i] < self.state_space[i][0] {
                self.state[i] = self.state_space[i][0];
            } else if self.state[i] > self.state_space[i][1] {
                self.state[i] = self.state_space[i][1];
            }
        }
        let done = self.state[0] < -self.x_threshold
            || self.state[0] > self.x_threshold
            || self.state[2] < -self.theta_threshold
            || self.state[2] > self.theta_threshold;

        Snapshot::new(self.state, 1.0, self.count > 500 || done)
    }
}

#[cfg(test)]
mod tests {
    use crate::components::env::Environment;
    use crate::env::cart_pole::CartPole;
    use ndarray::Array1;
    use ndarray_linalg::assert_close_max;

    #[test]
    fn test_cart_pole() {
        let mut env = CartPole::new();
        let snapshot1 = env.step(<CartPole as Environment>::Action::Left);

        assert!(!snapshot1.mask());
        assert_eq!(snapshot1.reward(), 1.0);
        assert_close_max!(
            &Array1::from(snapshot1.state().data.to_vec()),
            &Array1::from(vec![0.0, -0.19512196, 0.0, 0.29268293]),
            1e-8
        );

        env.step(<CartPole as Environment>::Action::Right);
        env.step(<CartPole as Environment>::Action::Right);
        let snapshot2 = env.step(<CartPole as Environment>::Action::Left);
        assert_close_max!(
            &Array1::from(snapshot2.state().data.to_vec()),
            &Array1::from(vec![-0.00000169, -0.00016741, 0.00003705, 0.00369301]),
            1e-8
        );

        for _i in 0..79 {
            env.step(<CartPole as Environment>::Action::Left);
        }
        assert!(env.step(<CartPole as Environment>::Action::Left).mask());
    }
}
