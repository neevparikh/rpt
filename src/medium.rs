use rand::rngs::StdRng;
use rand::Rng;

use crate::color::Color;
use crate::{hex_color, HitRecord, Ray};

/// Represents volumetric media To represent the media in a heterogenous way, the absorption
/// scattering parameters are dependent on the position.
pub struct Medium {
    /// absorption
    absorption: Box<dyn Fn(&glm::DVec3) -> f64 + Send + Sync>,

    /// scattering
    scattering: Box<dyn Fn(&glm::DVec3) -> f64 + Send + Sync>,

    /// emission
    emission: Box<dyn Fn(&glm::DVec3) -> Color + Send + Sync>,

    /// phase fn
    phase: Box<dyn Fn(&glm::DVec3, &glm::DVec3) -> f64 + Send + Sync>,

    /// sample phase fn
    sample_ph: Box<dyn Fn(&glm::DVec3, &mut StdRng) -> (glm::DVec3, f64) + Send + Sync>,
}

/// Getters
impl Medium {
    /// get absorption factor
    pub fn absorption(&self, pos: &glm::DVec3) -> f64 {
        let absorption = &self.absorption;
        absorption(pos)
    }

    /// get emission factor
    pub fn emission(&self, pos: &glm::DVec3) -> glm::DVec3 {
        let emission = &self.emission;
        emission(pos)
    }

    /// get scattering factor
    pub fn scattering(&self, pos: &glm::DVec3) -> f64 {
        let scattering = &self.scattering;
        scattering(pos)
    }

    /// get extinction factor
    pub fn extinction(&self, pos: &glm::DVec3) -> f64 {
        let absorption = &self.absorption;
        let scattering = &self.scattering;
        absorption(pos) + scattering(pos)
    }

    /// compute phase func
    /// - `wo` - unit direction vector toward the viewer
    /// - `wi` - unit direction vector toward the incident ray
    ///   - i.e. both point away from each other/the point
    pub fn phase(&self, wo: &glm::DVec3, wi: &glm::DVec3) -> f64 {
        let phase = &self.phase;
        phase(wo, wi)
    }

    /// Sample phase distribution to get direction
    pub fn sample_ph(&self, wo: &glm::DVec3, rng: &mut StdRng) -> (glm::DVec3, f64) {
        let sample_ph = &self.sample_ph;
        sample_ph(wo, rng)
    }
}

impl Medium {
    /// Create a default homogeneous isotropic medium, i.e. uniform fog
    pub fn homogeneous_isotropic(absorption: f64, scattering: f64) -> Medium {
        Medium {
            absorption: Box::new(move |_| absorption),
            scattering: Box::new(move |_| scattering),
            emission:   Box::new(move |_| hex_color(0x808080)),
            phase:      Box::new(move |_, _| 1.0 / (4.0 * glm::pi::<f64>())),
            sample_ph:  Box::new(move |_, rng| {
                let wo = glm::vec3(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                );
                (glm::normalize(&wo), 1.0 / (4.0 * glm::pi::<f64>()))
            }),
        }
    }

    /// Create a colored, emissive homogeneous isotropic medium
    pub fn colored_glowing_fog(absorption: f64, scattering: f64) -> Medium {
        Medium {
            absorption: Box::new(move |_| absorption),
            scattering: Box::new(move |_| scattering),
            emission:   Box::new(move |x| {
                if x[1] > 250.0 {
                    hex_color(0xFF0000) * 10.0
                } else {
                    hex_color(0x0000FF) * 10.0
                }
            }),
            phase:      Box::new(move |_, _| 1.0 / 4.0 * glm::pi::<f64>()),
            sample_ph:  Box::new(move |_, rng| {
                let wo = glm::vec3(
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                    rng.gen_range(0.0..1.0),
                );
                (glm::normalize(&wo), 1.0 / 4.0 * glm::pi::<f64>())
            }),
        }
    }
}

impl Medium {
    /// Returns transmittence along ray up to t_max
    pub fn transmittence(&self, ray: &Ray, t_max: f64, step: f64, rng: &mut StdRng) -> f64 {
        let extinction = self.extinction(&ray.origin);
        let optical_thickness = extinction * t_max;
        (-optical_thickness).exp()
    }

    /// Sample distance along a ray, return distance y and pdf(y)
    pub fn sample_d(&self, ray: &Ray, rng: &mut StdRng) -> (f64, f64, f64) {
        let random = rng.gen_range(0.0..1.0);

        // homogeneous
        let extinction = self.extinction(&ray.origin);

        // analytic
        let dist = -f64::ln(random) / extinction;
        let transmittence = self.transmittence(ray, dist, 0.0, rng);
        let pdf = extinction * transmittence;
        let cdf = 1.0 - transmittence;

        (dist, pdf, cdf)
    }
}
