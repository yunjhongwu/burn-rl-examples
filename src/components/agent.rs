pub trait Agent<State, Action> {
    fn react(&mut self, state: &State) -> Action;
    fn collect(&mut self, reward: f32, done: bool);
    fn reset(&mut self);
}
