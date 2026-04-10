use deke_types::{DekeError, DekeResult, SRobotQ, Validator};
use glam::Affine3A;
use wreck::{Collider, ColliderComponent, soa::SpheresSoA};

pub trait InlinedRobot<const N: usize>: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn clone_box(&self) -> Box<dyn InlinedRobot<N>>;
    fn validate(
        &mut self,
        q: SRobotQ<N>,
        env: &Collider,
        ee_attachments: &[Collider],
    ) -> DekeResult<()>;
    fn spheres(&self, q: SRobotQ<N>) -> SpheresSoA;
    fn debug_self_collisions(&self, q: SRobotQ<N>) -> Vec<(usize, usize)>;
    fn eefk(&self, q: SRobotQ<N>) -> Affine3A;
    fn dof(&self) -> usize {
        N
    }
}

pub struct InlinedWreckValidator<const N: usize> {
    robot: Box<dyn InlinedRobot<N>>,
    ee_attachments: Vec<Collider>,
    environment: Collider,
}

impl<const N: usize> std::fmt::Debug for InlinedWreckValidator<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InlinedWreckValidator")
            .field("robot", &self.robot.name())
            .field("ee_attachments", &self.ee_attachments)
            .field("environment", &self.environment)
            .finish()
    }
}

impl<const N: usize> Clone for InlinedWreckValidator<N> {
    fn clone(&self) -> Self {
        Self {
            robot: self.robot.clone_box(),
            ee_attachments: self.ee_attachments.clone(),
            environment: self.environment.clone(),
        }
    }
}

impl<const N: usize> InlinedWreckValidator<N> {
    pub fn new(robot: Box<dyn InlinedRobot<N>>, env: Collider) -> Self {
        Self {
            robot,
            environment: env,
            ee_attachments: Vec::new(),
        }
    }

    pub fn spheres(&self, q: SRobotQ<N>) -> SpheresSoA {
        self.robot.spheres(q)
    }

    pub fn debug_self_collisions(&self, q: SRobotQ<N>) -> Vec<(usize, usize)> {
        self.robot.debug_self_collisions(q)
    }

    pub fn eefk(&self, q: SRobotQ<N>) -> Affine3A {
        self.robot.eefk(q)
    }

    pub fn add_ee_attachment(&mut self, collider: Collider) {
        self.ee_attachments.push(collider);
    }

    pub fn dof(&self) -> usize {
        self.robot.dof()
    }

    pub fn environment(&self) -> &Collider {
        &self.environment
    }

    pub fn with_environment(&mut self, environment: Collider) {
        self.environment = environment;
    }

    pub fn with_environment_component(mut self, component: impl ColliderComponent) -> Self {
        self.environment.add(component);
        self
    }

    pub fn with_environment_collider(mut self, collider: Collider) -> Self {
        self.environment.include(collider);
        self
    }
}

impl<const N: usize> Validator<N> for InlinedWreckValidator<N> {
    fn validate<E: Into<DekeError>, A: TryInto<SRobotQ<N>, Error = E>>(
        &mut self,
        q: A,
    ) -> DekeResult<()> {
        let q = q.try_into().map_err(|e| e.into())?;
        self.robot
            .validate(q, &self.environment, &self.ee_attachments)
    }

    fn validate_motion(&mut self, qs: &[SRobotQ<N>]) -> DekeResult<()> {
        for q in qs {
            self.validate(q.as_slice())?;
        }
        Ok(())
    }
}
