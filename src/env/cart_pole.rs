use crate::base::Snapshot;
use crate::base::{Action, State};
use crate::components::agent::Agent;
use crate::components::env::Environment;
use nannou::prelude::*;
use nannou::text::Align;
use nannou::window::Id;
use std::f32::consts::PI;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

#[derive(Debug, Copy, Clone, Default)]
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
    #[allow(unused)]
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

impl Default for CartPoleAction {
    fn default() -> Self {
        Self::Left
    }
}

impl Action for CartPoleAction {
    fn random() -> Self {
        if random::<f32>() < 0.5 {
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
            state: CartPoleState::default(),
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
            self.state[i] = (random::<f32>() * 0.1) - 0.05;
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

pub struct Visualizer<A: Agent> {
    id: Id,
    env: CartPole,
    agent: A,
    snapshot: Snapshot<<CartPole as Environment>::State>,
    episode: u32,
}

impl<A: Agent<State = CartPoleState, Action = CartPoleAction> + 'static> Visualizer<A> {
    #[allow(unused)]
    const CART_HEIGHT: f32 = 30.0;
    #[allow(unused)]
    const CART_WIDTH: f32 = 50.0;

    #[allow(unused)]
    fn new(id: Id) -> Self {
        let mut env = CartPole::new();
        let state = env.reset();
        Self {
            id,
            env,
            agent: A::default(),
            snapshot: state,
            episode: 0,
        }
    }

    #[allow(unused)]
    pub fn run() {
        let model: app::ModelFn<Self> = move |app: &App| {
            let id = app.new_window().view(Self::view).build().unwrap();
            Self::new(id)
        };

        nannou::app(model)
            .loop_mode(LoopMode::RefreshSync)
            .update(Self::update)
            .run();
    }

    #[allow(unused)]
    fn step(&mut self) {
        if self.snapshot.mask() {
            self.snapshot = self.env.reset();
            self.episode += 1;
            return;
        }

        self.snapshot = self.env.step(self.agent.react(self.snapshot.state()));
    }

    #[allow(unused)]
    fn state(&self) -> &<CartPole as Environment>::State {
        self.snapshot.state()
    }

    #[allow(unused)]
    fn id(&self) -> Id {
        self.id
    }

    #[allow(unused)]
    fn update(_app: &App, model: &mut Self, _update: Update) {
        model.step();
    }

    #[allow(unused)]
    fn view(app: &App, model: &Self, frame: Frame) {
        let draw = app.draw();
        let (window_width, window_height) = app.window(model.id()).unwrap().inner_size_points();
        let state = model.state();
        let position = state[0];
        let cart_x = window_width * (position / (2.0 * 4.8));
        let cart_y = window_height * -0.2;
        let angle = state[2] + 0.5 * PI;
        let pole_len = 0.25 * window_height;

        draw.background().color(WHITE);
        draw.line()
            .start(Point2::new(-0.5 * window_width, cart_y))
            .end(Point2::new(0.5 * window_width, cart_y))
            .weight(2.0)
            .color(BLACK);
        draw.rect()
            .x_y(cart_x, cart_y)
            .w_h(Self::CART_WIDTH, Self::CART_HEIGHT)
            .color(BLACK);
        draw.line()
            .start(Point2::new(cart_x, cart_y))
            .end(Point2::new(
                cart_x + f32::cos(angle) * pole_len,
                cart_y + f32::sin(angle) * pole_len,
            ))
            .weight(8.0)
            .color(ORANGE);

        draw.text(&format!("Episode: {}", model.episode))
            .x_y(window_width * -0.4, window_height * 0.6)
            .y_align_text(Align::Start)
            .left_justify()
            .font_size(24)
            .color(BLACK);

        draw.to_frame(app, &frame).unwrap();
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
