use crate::environment::Environment;
use crate::light::Light;
use crate::medium::Medium;
use crate::object::Object;
use crate::{Cube, Material, Mesh, Transformed};

/// Type for adding light and object at the same time
pub type LightAndMeshObject = (Mesh, Material);

/// Object representing a scene that can be rendered
#[derive(Default)]
pub struct Scene {
    /// Collection of objects in the scene
    pub objects: Vec<Object>,

    /// Collection of lights in the scene
    pub lights: Vec<Light>,

    /// Collection of lights in the scene
    pub media: Vec<Medium>,

    /// Environment map used for scene lighting
    pub environment: Environment,
}

impl Scene {
    /// Construct a new, empty scene
    pub fn new() -> Self {
        Default::default()
    }
}

/// Trait that allows adding an object or light to a scene
pub trait SceneAdd<T> {
    /// Add an object or light to the scene
    fn add(&mut self, node: T);
}

impl SceneAdd<Object> for Scene {
    fn add(&mut self, object: Object) {
        self.objects.push(object);
    }
}

impl SceneAdd<Light> for Scene {
    fn add(&mut self, light: Light) {
        self.lights.push(light);
    }
}

/// Implements adding object lights as both objects and lights at the same time
/// This is needed because Light::Object cannot implement clone, so we cannot clone the underlying
/// object, only Mesh can be cloned. Thus, we take in a Mesh and a Material
///
/// option 1 -> Light::Object should be refactored, and Light should be at trait
/// option 2 -> add them separately, conform to existing abstraction
impl SceneAdd<LightAndMeshObject> for Scene {
    fn add(&mut self, m: LightAndMeshObject) {
        let (mesh, material) = m;
        let obj = Object::new(mesh.clone()).material(material);
        let light = Light::Object(Object::new(mesh.clone()).material(material));
        self.add(obj);
        self.add(light);
    }
}

impl SceneAdd<(Transformed<Cube>, Material)> for Scene {
    fn add(&mut self, m: (Transformed<Cube>, Material)) {
        let (mesh, material) = m;
        let obj = Object::new(mesh.clone()).material(material);
        let light = Light::Object(Object::new(mesh.clone()).material(material));
        self.add(obj);
        self.add(light);
    }
}

impl SceneAdd<Medium> for Scene {
    fn add(&mut self, medium: Medium) {
        self.media.push(medium);
    }
}
