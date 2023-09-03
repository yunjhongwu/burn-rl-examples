use crate::base::Snapshot;
use crate::components::agent::Agent;
use crate::components::env::Environment;
use crate::env::cart_pole::CartPole;
use nannou::prelude::*;
use nannou::text::Align;
use nannou::window::Id;

struct RuleBasedCartPole {}

impl RuleBasedCartPole {
    fn new() -> Self {
        Self {}
    }
}

impl Agent for RuleBasedCartPole {
    type State = <CartPole as Environment>::State;
    type Action = <CartPole as Environment>::Action;
    fn react(&mut self, state: &Self::State) -> Self::Action {
        if state[2] < 0.0 {
            Self::Action::Left
        } else {
            Self::Action::Right
        }
    }

    fn collect(&mut self, _reward: f32, _done: bool) {}

    fn reset(&mut self) {}

    fn is_eval(&self) -> bool {
        false
    }
}

pub struct Runner {
    id: Id,
    env: CartPole,
    agent: RuleBasedCartPole,
    snapshot: Snapshot<<CartPole as Environment>::State>,
    episode: u32,
}

impl Runner {
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
            agent: RuleBasedCartPole::new(),
            snapshot: state,
            episode: 0,
        }
    }

    #[allow(unused)]
    pub fn run() {
        nannou::app(Self::model)
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
    fn model(app: &App) -> Self {
        let id = app.new_window().view(Self::view).build().unwrap();
        Runner::new(id)
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
