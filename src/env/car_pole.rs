use crate::components::env::Environment;
use crate::utils::Snapshot;
use std::f64::consts::PI;
use std::fmt::Debug;

type State = [f64; 4];

#[allow(unused)]
#[derive(Debug)]
enum Action {
    Left,
    Right,
    None,
}

#[derive(Debug)]
pub struct CartPole {
    state_space: [[f64; 2]; 4],
    gravity: f64,
    pole_mass: f64,
    total_mass: f64,
    length: f64,
    polemass_length: f64,
    force_mag: f64,
    tau: f64,
    x_threshold: f64,
    theta_threshold: f64,
    state: State,
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
                [f64::NEG_INFINITY, f64::INFINITY],
                [-0.418, 0.418],
                [f64::NEG_INFINITY, f64::INFINITY],
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
            state: [0.0, 0.0, 0.0, 0.0],
            count: 0,
        }
    }
}

impl Environment<State, Action> for CartPole {
    fn render(&mut self) {}

    fn reset(&mut self) -> Snapshot<State> {
        for i in 0..4 {
            self.state[i] = (rand::random::<f64>() * 0.1) - 0.05;
        }
        self.count = 0;
        Snapshot::<State>::new(self.state, 1.0, false)
    }

    fn step(&mut self, action: Action) -> Snapshot<State> {
        let force = match action {
            Action::Left => -self.force_mag,
            Action::Right => self.force_mag,
            Action::None => 0.0,
        };
        let cos_theta = self.state[2].cos();
        let sin_theta = self.state[2].sin();
        let temp =
            (force + self.polemass_length * self.state[3].powi(2) * sin_theta) / self.total_mass;

        let theta_acc = (self.gravity * sin_theta - temp * cos_theta)
            / (self.length * (4.0 / 3.0 - self.pole_mass * cos_theta.powi(2) / self.total_mass));
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
    use crate::env::car_pole::{Action, CartPole};
    use ndarray::Array1;
    use ndarray_linalg::assert_close_max;

    #[test]
    fn test_cart_pole() {
        let mut env = CartPole::new();
        let snapshot1 = env.step(Action::Left);

        assert!(!snapshot1.mask());
        assert_eq!(snapshot1.reward(), 1.0);
        assert_close_max!(
            &Array1::from(snapshot1.state().to_vec()),
            &Array1::from(vec![0.0, -0.19512195, 0.0, 0.29268293]),
            1e-8
        );

        env.step(Action::Right);
        env.step(Action::Right);
        let snapshot2 = env.step(Action::Left);
        assert_close_max!(
            &Array1::from(snapshot2.state().to_vec()),
            &Array1::from(vec![-0.00000169, -0.00016741, 0.00003705, 0.00369305]),
            1e-8
        );

        for _i in 0..79 {
            env.step(Action::None);
        }
        assert!(env.step(Action::None).mask());
    }
}
