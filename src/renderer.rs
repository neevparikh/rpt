use image::RgbImage;
use indicatif::{ParallelProgressIterator, ProgressBar};
use kd_tree::{ItemAndDistance, KdPoint, KdTree};
use nalgebra;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::buffer::{Buffer, Filter};
use crate::camera::Camera;
use crate::color::Color;
use crate::light::Light;
use crate::material::Material;
use crate::object::Object;
use crate::scene::Scene;
use crate::shape::{HitRecord, Ray};
use crate::Medium;

const EPSILON: f64 = 1e-12;
const FIREFLY_CLAMP: f64 = 100.0;

/// Path tracing

/// Builder object for rendering a scene
pub struct Renderer<'a> {
    /// The scene to be rendered
    pub scene: &'a Scene,

    /// The camera to use
    pub camera: Camera,

    /// The width of the output image
    pub width: u32,

    /// The height of the output image
    pub height: u32,

    /// Exposure value (EV)
    pub exposure_value: f64,

    /// Optional noise-reduction filter
    pub filter: Filter,

    /// Ray marching step size
    pub stepsize: f64,

    /// The maximum number of ray bounces
    pub max_bounces: u32,

    /// Number of random paths traced per pixel
    pub num_samples: u32,

    gather_size: usize,

    gather_size_volume: usize,
}

impl<'a> Renderer<'a> {
    /// Construct a new renderer for a scene
    pub fn new(scene: &'a Scene, camera: Camera) -> Self {
        Self {
            scene,
            camera,
            width: 800,
            height: 600,
            exposure_value: 0.0,
            filter: Filter::default(),
            stepsize: 0.0,
            max_bounces: 0,
            num_samples: 1,
            gather_size: 50,
            gather_size_volume: 50,
        }
    }

    /// Set the width of the rendered scene
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set the height of the rendered scene
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Set the exposure value of the rendered scene
    pub fn exposure_value(mut self, exposure_value: f64) -> Self {
        self.exposure_value = exposure_value;
        self
    }

    /// Set the stepsize for ray marching
    pub fn stepsize(mut self, stepsize: f64) -> Self {
        self.stepsize = stepsize;
        self
    }

    /// Set the noise reduction filter
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = filter;
        self
    }

    /// Set the maximum number of ray bounces when ray is traced
    pub fn max_bounces(mut self, max_bounces: u32) -> Self {
        self.max_bounces = max_bounces;
        self
    }

    /// Set the number of random paths traced per pixel
    pub fn num_samples(mut self, num_samples: u32) -> Self {
        self.num_samples = num_samples;
        self
    }

    /// Set the number of photons that are counted in the final gather step when on a surface
    pub fn gather_size(mut self, gather_size: usize) -> Self {
        self.gather_size = gather_size;
        self
    }

    /// Set the number of photons that are counted in the final gather step when in a volume
    pub fn gather_size_volume(mut self, gather_size_volume: usize) -> Self {
        self.gather_size_volume = gather_size_volume;
        self
    }

    /// Render the scene by path tracing
    pub fn render(&self) -> RgbImage {
        let mut buffer = Buffer::new(self.width, self.height, self.filter);
        self.sample(self.num_samples, &mut buffer);
        buffer.image()
    }

    /// Render the scene iteratively, calling a callback after every k samples
    pub fn iterative_render<F>(&self, callback_interval: u32, mut callback: F)
    where
        F: FnMut(u32, &Buffer),
    {
        let mut buffer = Buffer::new(self.width, self.height, self.filter);
        let mut iteration = 0;
        while iteration < self.num_samples {
            let steps = std::cmp::min(self.num_samples - iteration, callback_interval);
            self.sample(steps, &mut buffer);
            iteration += steps;
            callback(iteration, &buffer);
        }
    }

    fn sample(&self, iterations: u32, buffer: &mut Buffer) {
        let colors: Vec<_> = (0..self.height)
            .into_par_iter()
            .flat_map(|y| {
                let mut rng = StdRng::from_entropy();
                (0..self.width)
                    .into_iter()
                    .map(|x| self.get_color(x, y, iterations, &mut rng))
                    .collect::<Vec<_>>()
            })
            .collect();
        buffer.add_samples(&colors);
    }

    fn get_color(&self, x: u32, y: u32, iterations: u32, rng: &mut StdRng) -> Color {
        let dim = std::cmp::max(self.width, self.height) as f64;
        let xn = ((2 * x + 1) as f64 - self.width as f64) / dim;
        let yn = ((2 * (self.height - y) - 1) as f64 - self.height as f64) / dim;
        let mut color = glm::vec3(0.0, 0.0, 0.0);
        for _ in 0..iterations {
            let dx = rng.gen_range((-1.0 / dim)..(1.0 / dim));
            let dy = rng.gen_range((-1.0 / dim)..(1.0 / dim));
            color += self.trace_ray(self.camera.cast_ray(xn + dx, yn + dy, rng), 0, rng);
        }
        color / f64::from(iterations) * 2.0_f64.powf(self.exposure_value)
    }

    /// Trace a ray, obtaining a Monte Carlo estimate of the luminance
    fn trace_ray(&self, ray: Ray, _num_bounces: u32, rng: &mut StdRng) -> Color {
        if self.scene.media.len() > 0 {
            // TODO: this should be intersection tests with a bounded mesh of media
            let medium = &self.scene.media[0];

            // sample distance along ray:
            let (d, d_pdf, d_cdf) = medium.sample_d(&ray, rng);

            let wo = -glm::normalize(&ray.dir);
            // let rr_p: f64 = rng.gen_range(0.0..1.0);
            let (max_dist, surface_color) = match self.get_closest_hit(ray) {
                None => {
                    let background_dist = 600.0;
                    let color = if d >= background_dist {
                        medium.transmittence(&ray, background_dist, 0.0, rng)
                            * self.scene.environment.get_color(&ray.dir)
                            / d_cdf
                    } else {
                        glm::vec3(0.0, 0.0, 0.0)
                    };
                    (background_dist, color)
                }
                Some((h, object)) => {
                    let color = if d >= h.time {
                        let world_pos = ray.at(h.time);
                        let material = object.material;

                        let mut color = if _num_bounces == 0 {
                            material.emittance() * material.color()
                        } else {
                            glm::vec3(0.0, 0.0, 0.0)
                        };

                        color += self.sample_lights(&material, &world_pos, &h.normal, &wo, rng);
                        // maybe use num_bounces in the rr prob
                        if _num_bounces < self.max_bounces {
                            if let Some((wi, pdf)) = material.sample_f(&h.normal, &wo, rng) {
                                let f = material.bsdf(&h.normal, &wo, &wi);
                                let ray = Ray {
                                    origin: world_pos,
                                    dir:    wi,
                                };
                                let indirect = 1.0 / pdf
                                    * f.component_mul(&self.trace_ray(ray, _num_bounces + 1, rng))
                                    * wi.dot(&h.normal).abs();
                                color.x += indirect.x.min(FIREFLY_CLAMP);
                                color.y += indirect.y.min(FIREFLY_CLAMP);
                                color.z += indirect.z.min(FIREFLY_CLAMP);
                                // color /= rr_p;
                            }
                        }
                        medium.transmittence(&ray, h.time, 0.0, rng) * color / d_cdf
                    } else {
                        glm::vec3(0.0, 0.0, 0.0)
                    };
                    (h.time, color)
                }
            };

            if d < max_dist {
                let collision = ray.at(d);
                let abs = medium.absorption(&collision);
                let emm = medium.emission(&collision);
                let scat = medium.scattering(&collision);

                let mut color = if _num_bounces == 0 {
                    abs * emm
                } else {
                    glm::vec3(0.0, 0.0, 0.0)
                };

                // direct lighting for media particle
                color += self.sample_lights_for_media(&medium, &collision, &wo, rng);

                if _num_bounces < self.max_bounces {
                    let (wi, ph_p) = medium.sample_ph(&wo, rng);

                    let new_ray = Ray {
                        origin: collision,
                        dir:    wi,
                    };

                    // compute scattered light recursively
                    let mut indirect = scat * self.trace_ray(new_ray, _num_bounces + 1, rng);

                    // note that there is no cosine factor, because the media is a
                    // point-like sphere
                    indirect /= ph_p;
                    indirect *= medium.phase(&wo, &wi);

                    color += indirect;
                    // color /= rr_p;
                }

                color *= medium.transmittence(&ray, d, 0.0, rng);
                color /= d_pdf;
                color
            } else {
                surface_color
            }
        } else {
            match self.get_closest_hit(ray) {
                None => self.scene.environment.get_color(&ray.dir),
                Some((h, object)) => {
                    let world_pos = ray.at(h.time);
                    let material = object.material;
                    let wo = -glm::normalize(&ray.dir);

                    // let rr_p: f64 = rng.gen_range(0.0..1.0);
                    let mut color = if _num_bounces == 0 {
                        material.emittance() * material.color()
                    } else {
                        glm::vec3(0.0, 0.0, 0.0)
                    };
                    color += self.sample_lights(&material, &world_pos, &h.normal, &wo, rng);
                    if _num_bounces < self.max_bounces {
                        if let Some((wi, pdf)) = material.sample_f(&h.normal, &wo, rng) {
                            let f = material.bsdf(&h.normal, &wo, &wi);
                            let ray = Ray {
                                origin: world_pos,
                                dir:    wi,
                            };
                            let indirect = 1.0 / pdf
                                * f.component_mul(&self.trace_ray(ray, _num_bounces + 1, rng))
                                * wi.dot(&h.normal).abs();
                            color.x += indirect.x.min(FIREFLY_CLAMP);
                            color.y += indirect.y.min(FIREFLY_CLAMP);
                            color.z += indirect.z.min(FIREFLY_CLAMP);
                            // color /= rr_p;
                        }
                    }

                    color
                }
            }
        }
    }

    /// Explicitly sample from all the lights in the scene
    fn sample_lights_for_media(
        &self,
        medium: &Medium,
        pos: &glm::DVec3,
        wo: &glm::DVec3,
        rng: &mut StdRng,
    ) -> Color {
        let mut color = glm::vec3(0.0, 0.0, 0.0);
        let scat = medium.scattering(pos);
        let em = medium.emission(pos);
        for light in &self.scene.lights {
            if let Light::Ambient(ambient_color) = light {
                color += ambient_color.component_mul(&em);
            } else {
                let (intensity, wi, dist_to_light) = light.illuminate(pos, rng);
                let ray = Ray {
                    origin: *pos,
                    dir:    wi,
                };
                let closest_hit = self.get_closest_hit(ray.clone()).map(|(r, _)| r.time);

                if let Some(hit) = closest_hit {
                    if (hit - dist_to_light).abs() < EPSILON {
                        // analogue of bsdf
                        let ph = medium.phase(wo, &wi);
                        // radiance from the light is scattered and diminished by media
                        // note that there is no cosine factor
                        color += scat
                            * medium.transmittence(&ray, dist_to_light, 0.0, rng)
                            * &intensity
                            * scat
                            * ph;
                    }
                }
            }
        }
        color
    }

    /// Explicitly sample from all the lights in the scene
    fn sample_lights(
        &self,
        material: &Material,
        pos: &glm::DVec3,
        n: &glm::DVec3,
        wo: &glm::DVec3,
        rng: &mut StdRng,
    ) -> Color {
        let mut color = glm::vec3(0.0, 0.0, 0.0);
        let medium = if self.scene.media.len() > 0 {
            Some(&self.scene.media[0])
        } else {
            None
        };
        for light in &self.scene.lights {
            if let Light::Ambient(ambient_color) = light {
                color += ambient_color.component_mul(&material.color());
            } else {
                let (intensity, wi, dist_to_light) = light.illuminate(pos, rng);
                let ray = Ray {
                    origin: *pos,
                    dir:    wi,
                };
                let intensity = intensity;
                let wi = wi;
                let dist_to_light = dist_to_light;
                let closest_hit = self
                    .get_closest_hit(Ray {
                        origin: *pos,
                        dir:    wi,
                    })
                    .map(|(r, _)| r.time);

                if let Some(hit) = closest_hit {
                    if (hit - dist_to_light).abs() < EPSILON {
                        let f = material.bsdf(n, wo, &wi);
                        color += f.component_mul(&intensity) * wi.dot(n);
                        if medium.is_some() {
                            color *= medium.as_ref().unwrap().transmittence(
                                &ray,
                                dist_to_light,
                                0.0,
                                rng,
                            );
                        }
                    }
                }
            }
        }
        color
    }

    /// Loop through all objects in the scene to find the closest hit.
    ///
    /// Note that we intentionally do not use a `KdTree` to accelerate this computation.
    /// The reason is that some objects, like planes, have infinite extent, so it would
    /// not be appropriate to put them indiscriminately into a kd-tree.
    fn get_closest_hit(&self, ray: Ray) -> Option<(HitRecord, &'_ Object)> {
        let mut h = HitRecord::new();
        let mut hit = None;
        for object in &self.scene.objects {
            if object.shape.intersect(&ray, EPSILON, &mut h) {
                hit = Some(object);
            }
        }
        Some((h, hit?))
    }
}

/// Photon mapping

struct Photon {
    pub position:  glm::DVec3,
    pub direction: glm::DVec3,
    pub power:     glm::DVec3,
}

impl KdPoint for Photon {
    type Scalar = f64;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f64 {
        self.position[k]
    }
}

#[derive(Default)]
struct PhotonList(Vec<Photon>, Vec<Photon>);

impl PhotonList {
    fn new(surface: Vec<Photon>, volume: Vec<Photon>) -> Self {
        PhotonList(surface, volume)
    }
    fn merge(lists: Vec<Self>) -> Self {
        let mut out = PhotonList::default();
        for list in lists {
            out.0.extend(list.0);
            out.1.extend(list.1);
        }
        out
    }
    fn add_surface(&mut self, photon: Photon) {
        self.0.push(photon)
    }
    fn add_volume(&mut self, photon: Photon) {
        self.1.push(photon)
    }
}

impl std::fmt::Display for PhotonList {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        write!(
            f,
            "PhotonList(surface: {}, volume: {})",
            self.0.len(),
            self.1.len()
        )
    }
}

struct PhotonMap(KdTree<Photon>, KdTree<Photon>);

impl PhotonMap {
    fn new(list: PhotonList) -> Self {
        let surface_map = KdTree::build_by(list.0, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });
        let volume_map = KdTree::build_by(list.1, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });
        PhotonMap(surface_map, volume_map)
    }
    fn surface(&self) -> &KdTree<Photon> {
        &self.0
    }
    fn volume(&self) -> &KdTree<Photon> {
        &self.1
    }
}

impl<'a> Renderer<'a> {
    /// renders an image using photon mapping
    pub fn photon_map_render(&self, photon_count: usize) -> RgbImage {
        // ensure that scene only has object lights (may not be necessary)
        for light in self.scene.lights.iter() {
            match light {
                Light::Object(_) => {}
                _ => {
                    panic!("Only object lights are supported for photon mapping");
                }
            }
        }

        println!("Shooting photons");
        let watts = 100.;

        let pb = ProgressBar::new(photon_count as u64);
        pb.set_draw_rate(1);
        let mut photon_list = (0..photon_count)
            .collect::<Vec<_>>()
            .into_par_iter()
            .progress_with(pb)
            .map(|_|
            // shoot photon from a random light
            {
                let mut rng = StdRng::from_entropy();
                let power = watts / photon_count as f64;
                self.shoot_photon(power, &mut rng)
            })
            .collect::<Vec<_>>();
        let photon_list = PhotonList::merge(photon_list);

        println!("{}", photon_list);

        let mut avg = 0.;
        for p in photon_list.0.iter() {
            avg += p.power.norm();
        }
        avg /= photon_list.0.len() as f64;
        println!("surface avg: {}", avg);
        avg = 0.;
        for p in photon_list.1.iter() {
            avg += p.power.norm();
        }
        avg /= photon_list.1.len() as f64;
        println!("vol avg: {}", avg);

        println!("Building kdtree");
        let photon_map = PhotonMap::new(photon_list);

        println!("Tracing rays");
        let mut buffer = Buffer::new(self.width, self.height, self.filter);
        let colors: Vec<_> = (0..self.height)
            .into_par_iter()
            .progress()
            .flat_map(|y| {
                let mut rng = StdRng::from_entropy();
                (0..self.width)
                    .into_iter()
                    .map(|x| self.get_color_with_photon_map(x, y, &mut rng, &photon_map))
                    .collect::<Vec<_>>()
            })
            .collect();
        buffer.add_samples(&colors);

        buffer.image()
    }

    /// shoot a photon from a random light with power `power` and return a list of
    /// photons that have gathered in the scene
    fn shoot_photon(&self, power: f64, rng: &mut StdRng) -> PhotonList {
        // FIXME: sample random light based on area instead of choosing randomly
        let light_index: usize = rng.gen_range(0..self.scene.lights.len());
        let light = &self.scene.lights[light_index as usize];

        // sample a random point on the light and a random direction in the hemisphere
        if let Light::Object(object) = light {
            // the `target` arg isn't used when sampling a triangle, so it can be a dummy value
            // Sample a location on the light
            let target = glm::vec3(0., 0., 0.);
            let (pos, n, pdf) = object.shape.sample(&target, rng);

            // sample random hemisphere direction
            let phi = 2. * glm::pi::<f64>() * rng.gen::<f64>();
            let theta = (1. - rng.gen::<f64>()).acos();
            let pdf_of_sample = 0.5 * glm::one_over_pi::<f64>();
            let random_hemisphere_dir = glm::vec3(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );

            // rotate direction towards normal
            let rotation: nalgebra::Rotation3<f64> = nalgebra::Rotation3::rotation_between(
                &glm::vec3(0., 1., 0.),
                &n,
            )
            .unwrap_or_else(|| {
                nalgebra::Rotation3::rotation_between(&glm::vec3(0., 1., 0.00000001), &n).unwrap()
            });
            let direction = rotation * random_hemisphere_dir;

            // recursively gather photons
            let photons = self.trace_photon(
                Ray {
                    origin: pos,
                    dir:    direction,
                },
                power * object.material.color() / pdf / pdf_of_sample,
                rng,
                0,
            );

            photons
        } else {
            panic!("Found non-object light while photon mapping")
        }
    }

    /// trace a photon along ray `ray` with power `power` and check for intersections. Returns
    /// vec of photons that have been recursively traces and placed in the scene
    fn trace_photon(
        &self,
        ray: Ray,
        power: glm::DVec3,
        rng: &mut StdRng,
        num_bounces: i32,
    ) -> PhotonList {
        let surface_hit = self.get_closest_hit(ray);

        if self.scene.media.len() > 0 {
            let medium = &self.scene.media[0];
            let (d, d_pdf, _d_cdf) = medium.sample_d(&ray, rng);

            // if the sampled distance is less than the distance to the surface (or no surface
            // exists) then bounce a photon in the medium
            if surface_hit.as_ref().is_none() || d < surface_hit.as_ref().unwrap().0.time {
                let wo = -glm::normalize(&ray.dir);

                let collision = ray.at(d);
                let abs = medium.absorption(&collision);
                let emm = medium.emission(&collision);
                let scat = medium.scattering(&collision);
                let extinction = abs + scat;

                // TODO: add back d_pdf term
                let attenuated_power = power * medium.transmittence(&ray, d, 0.0, rng); // / d_pdf;

                let rr_prob = scat / extinction;
                let mut next_photons = if rng.gen::<f64>() < rr_prob {
                    let (wi, ph_p) = medium.sample_ph(&wo, rng);

                    let new_ray = Ray {
                        origin: collision,
                        dir:    wi,
                    };
                    // compute scattered light recursively
                    self.trace_photon(
                        new_ray,
                        attenuated_power * scat * medium.phase(&wo, &wi) / ph_p / rr_prob,
                        rng,
                        num_bounces + 1,
                    )
                } else {
                    PhotonList::default()
                };

                // add current photon
                next_photons.add_volume(Photon {
                    position:  collision,
                    direction: wo,
                    power:     attenuated_power,
                });

                return next_photons;
            }
        }

        match surface_hit {
            None => PhotonList::default(),
            Some((h, object)) => {
                let world_pos = ray.at(h.time);
                let material = object.material;
                let wo = -glm::normalize(&ray.dir);

                // page 16 of siggraph course on photon mapping
                // let specular = 1. - material.roughness;
                let specular = 0.1;
                let specular = material.get_specular();
                let diffuse = material.get_diffuse();
                let specular = glm::vec3(0.1, 0.1, 0.1);
                let diffuse = glm::vec3(0.5, 0.5, 0.5);
                let p_r = vec![
                    specular.x + diffuse.x,
                    specular.y + diffuse.y,
                    specular.z + diffuse.z,
                ]
                .into_iter()
                .fold(f64::NAN, f64::max);
                let diffuse_sum = diffuse.x + diffuse.y + diffuse.z;
                let specular_sum = specular.x + specular.y + specular.z;
                let p_d = diffuse_sum / (diffuse_sum + specular_sum) * p_r;
                let p_s = specular_sum / (diffuse_sum + specular_sum) * p_r;

                // only do diffuse russian rouletter for now (no specular)
                let russian_roulette: f64 = rng.gen();

                let p_d = 0.9;
                if russian_roulette < p_d {
                    // diffuse reflection
                    if let Some((wi, pdf)) = material.sample_f(&h.normal, &wo, rng) {
                        let f = material.bsdf(&h.normal, &wo, &wi);
                        let ray = Ray {
                            origin: world_pos,
                            dir:    wi,
                        };

                        // account for the chance of terminating
                        let russian_roulette_scale_factor =
                            glm::vec3(diffuse.x / p_d, diffuse.y / p_d, diffuse.z / p_d);
                        let russian_roulette_scale_factor = 0.5;

                        // gather recursive photons with scaled down power
                        let mut next_photons = self.trace_photon(
                            ray,
                            power.component_mul(&f)
                                * russian_roulette_scale_factor
                                * wi.dot(&h.normal)
                                / pdf,
                            rng,
                            num_bounces + 1,
                        );

                        // add photon from current step
                        if pdf != 1. {
                            // only add current photon if surface is not specular (pdf of 1)
                            if russian_roulette < p_s {
                                next_photons.add_surface(Photon {
                                    position: world_pos,
                                    direction: wo,
                                    power,
                                });
                            }
                        }

                        next_photons
                    } else {
                        // total internal reflection: no photons
                        PhotonList::default()
                    }
                } else {
                    // absorbed
                    PhotonList::default()
                }
            }
        }
    }

    /// traces rays for a given pixel in the image and uses photon map to calc
    /// indirect lighting
    fn get_color_with_photon_map(
        &self,
        x: u32,
        y: u32,
        rng: &mut StdRng,
        photon_map: &PhotonMap,
    ) -> Color {
        let dim = std::cmp::max(self.width, self.height) as f64;
        let xn = ((2 * x + 1) as f64 - self.width as f64) / dim;
        let yn = ((2 * (self.height - y) - 1) as f64 - self.height as f64) / dim;
        let mut color = glm::vec3(0.0, 0.0, 0.0);
        for _ in 0..self.num_samples {
            let dx = rng.gen_range((-1.0 / dim)..(1.0 / dim));
            let dy = rng.gen_range((-1.0 / dim)..(1.0 / dim));
            // trace ray
            color += self.trace_ray_with_photon_map(
                self.camera.cast_ray(xn + dx, yn + dy, rng),
                0,
                rng,
                photon_map,
            );
        }
        color / f64::from(self.num_samples) * 2.0_f64.powf(self.exposure_value)
    }

    /// trace ray `Ray` through the scene to calculate illumination. Uses photon map
    /// for indirect illumination
    fn trace_ray_with_photon_map(
        &self,
        ray: Ray,
        num_bounces: u32,
        rng: &mut StdRng,
        photon_map: &PhotonMap,
    ) -> Color {
        match self.get_closest_hit(ray) {
            None => self.scene.environment.get_color(&ray.dir),
            Some((h, object)) => {
                let world_pos = ray.at(h.time);
                let material = object.material;
                let wo = -glm::normalize(&ray.dir);

                if self.scene.media.len() > 0 {
                    // TODO: this should be intersection tests with a bounded mesh of media
                    let medium = &self.scene.media[0];
                    let (d, d_pdf, _d_cdf) = medium.sample_d(&ray, rng);
                    if d < h.time {
                        let collision = ray.at(d);
                        let abs = medium.absorption(&collision);
                        let emm = medium.emission(&collision);
                        let scat = medium.scattering(&collision);
                        let extinction = abs + scat;

                        let mut color = Color::new(0., 0., 0.);

                        // direct lighting for media particle
                        // color += self.sample_lights_for_media(&medium, &collision, &wo, rng);

                        // estimate indirect lighting via photon map
                        let near_photons = photon_map.volume().nearests(
                            &[collision.x, collision.y, collision.z],
                            self.gather_size_volume,
                        );

                        let max_dist_squared = near_photons
                            .iter()
                            .map(
                                |ItemAndDistance {
                                     squared_distance, ..
                                 }| { squared_distance },
                            )
                            .fold(0., |acc: f64, &p: &f64| acc.max(p));

                        for ItemAndDistance {
                            item: photon,
                            squared_distance: _,
                        } in near_photons
                        {
                            color += photon.power * medium.phase(&wo, &photon.direction);
                        }
                        color /= (4. / 3.) * glm::pi::<f64>() * max_dist_squared.powf(1.5);
                        // color /= extinction;

                        color *= medium.transmittence(&ray, d, 0.0, rng);
                        color /= d_pdf;

                        // if max_dist_squared > 0.1 {
                        //     color = Color::new(0., 0., 0.);
                        // }
                        return color;
                    }
                }

                if material.is_mirror() {
                    if num_bounces > 100 {
                        return Color::new(0., 0., 0.);
                    }
                    if let Some((wi, pdf)) = material.sample_f(&h.normal, &wo, rng) {
                        self.trace_ray_with_photon_map(
                            Ray {
                                origin: world_pos,
                                dir:    wi,
                            },
                            num_bounces + 1,
                            rng,
                            photon_map,
                        )
                    } else {
                        // total internal reflection
                        Color::new(0., 0., 0.)
                    }
                } else {
                    let near_photons = photon_map
                        .surface()
                        .nearests(&[world_pos.x, world_pos.y, world_pos.z], self.gather_size);

                    // of the nearest photons, how far away is the furthest one?
                    let max_dist_squared = near_photons
                        .iter()
                        .map(
                            |ItemAndDistance {
                                 squared_distance, ..
                             }| { squared_distance },
                        )
                        .fold(0., |acc: f64, &p: &f64| acc.max(p));

                    let mut color = Color::new(0.0, 0.0, 0.0);

                    // indirect lighting via photon map
                    for ItemAndDistance {
                        item: photon,
                        squared_distance: _,
                    } in near_photons
                    {
                        color += material
                            .bsdf(&h.normal, &wo, &photon.direction)
                            .component_mul(&photon.power)
                            * photon.direction.dot(&h.normal).clamp(0., 1.);
                    }

                    // normalize by (1/(pi * r^2))
                    color = color * (1. / (glm::pi::<f64>() * max_dist_squared));

                    // TODO: divide by probability of sampling behind surface

                    // if max_dist_squared > 0.01 {
                    //     color = Color::new(0., 0., 0.);
                    // }

                    // direct lighting via light sampling
                    // color += self.sample_lights(&material, &world_pos, &h.normal, &wo, rng);

                    // emitted lighting
                    // color += material.emittance() * material.color();

                    color
                }
            }
        }
    }
}
