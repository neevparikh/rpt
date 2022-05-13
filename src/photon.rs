use bvh::aabb::{Bounded, AABB};
use bvh::bounding_hierarchy::BHShape;
use bvh::bvh::BVH;
use bvh::ray::Ray as BvhRay;
use bvh::Point3;
use image::RgbImage;
use indicatif::{ParallelProgressIterator, ProgressBar};
use kd_tree::{ItemAndDistance, KdPoint, KdTree};
use nalgebra;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::buffer::Buffer;
use crate::color::Color;
use crate::light::Light;
use crate::material::Material;
use crate::shape::Ray;
use crate::{HitRecord, Medium, Object, Renderer};

/// Photon mapping

#[derive(Debug, Clone)]
struct Photon {
    pub position:          glm::DVec3,
    pub direction:         glm::DVec3,
    pub power:             glm::DVec3,
    pub starting_position: glm::DVec3,
}

impl KdPoint for Photon {
    type Scalar = f64;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f64 {
        self.position[k]
    }
}

struct PhotonBeam {
    pub start_position: glm::DVec3,
    pub end_position:   glm::DVec3,
    pub radius:         f64,
    pub node_index:     usize,
    pub ray:            Ray,
    pub power:          glm::DVec3,
}

struct PhotonSphere {
    pub position:   glm::DVec3,
    pub radius:     f64,
    pub node_index: usize,
    pub direction:  glm::DVec3,
    pub power:      glm::DVec3,
}

impl Bounded for PhotonBeam {
    fn aabb(&self) -> AABB {
        let ax = self.start_position.x;
        let ay = self.start_position.y;
        let az = self.start_position.z;

        let bx = self.end_position.x;
        let by = self.end_position.y;
        let bz = self.end_position.z;

        let cx_sq = (ax - bx).powi(2);
        let cy_sq = (ay - by).powi(2);
        let cz_sq = (az - bz).powi(2);

        let sq_sum = cx_sq + cy_sq + cz_sq;

        let kx = ((cy_sq + cz_sq) / sq_sum).sqrt();
        let ky = ((cx_sq + cz_sq) / sq_sum).sqrt();
        let kz = ((cx_sq + cy_sq) / sq_sum).sqrt();

        let adjustment = glm::vec3(kx * self.radius, ky * self.radius, kz * self.radius);

        let min = glm::vec3(ax.min(bx), ay.min(by), az.min(bz)) - adjustment;
        let max = glm::vec3(ax.max(bx), ay.max(by), az.max(bz)) + adjustment;

        AABB::with_bounds(
            Point3::new(min.x as f32, min.y as f32, min.z as f32),
            Point3::new(max.x as f32, max.y as f32, max.z as f32),
        )
    }
}
impl BHShape for PhotonBeam {
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
}

impl Bounded for PhotonSphere {
    fn aabb(&self) -> AABB {
        let half_size = glm::vec3(self.radius, self.radius, self.radius);
        let min = self.position - half_size;
        let max = self.position + half_size;
        AABB::with_bounds(
            Point3::new(min.x as f32, min.y as f32, min.z as f32),
            Point3::new(max.x as f32, max.y as f32, max.z as f32),
        )
    }
}
impl BHShape for PhotonSphere {
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
}

#[derive(Debug, Default, Clone)]
struct PhotonList(Vec<Photon>, Vec<Photon>);

impl PhotonList {
    pub fn merge(lists: Vec<Self>) -> Self {
        let mut out = PhotonList::default();
        for list in lists {
            out.0.extend(list.0);
            out.1.extend(list.1);
        }
        out
    }
    pub fn add_surface(&mut self, photon: Photon) {
        self.0.push(photon)
    }
    pub fn add_volume(&mut self, photon: Photon) {
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

enum PhotonMap {
    PointMapForPointEstimate(KdTree<Photon>, KdTree<Photon>),
    PointMapForBeamEstimate(KdTree<Photon>, BVH, Vec<PhotonSphere>),
    BeamMapForBeamEstimate(KdTree<Photon>, BVH, Vec<PhotonBeam>),
}

impl PhotonMap {
    fn new_point_map_for_point_estimate(list: PhotonList) -> Self {
        let surface_map = KdTree::build_by(list.0, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });
        let volume_map = KdTree::build_by(list.1, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });
        Self::PointMapForPointEstimate(surface_map, volume_map)
    }

    fn new_point_map_for_beam_estimate(list: PhotonList) -> Self {
        let surface_map = KdTree::build_by(list.clone().0, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });
        let volume_map = KdTree::build_by(list.clone().1, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });

        let mut spheres = list
            .1
            .into_iter()
            .map(|p| {
                let max_distance_to_neighbor = volume_map
                    .nearests(&p, 10)
                    .into_iter()
                    .map(
                        |ItemAndDistance {
                             squared_distance, ..
                         }| squared_distance,
                    )
                    .fold(-1., f64::max)
                    .sqrt();
                // let max_distance_to_neighbor = 4.;
                let sphere = PhotonSphere {
                    position:   p.position,
                    radius:     max_distance_to_neighbor,
                    direction:  p.direction,
                    power:      p.power,
                    node_index: 0,
                };
                return sphere;
            })
            .collect::<Vec<_>>();

        let avg = spheres.iter().map(|s| s.radius).sum::<f64>() / spheres.len() as f64;
        let max = spheres.iter().map(|s| s.radius).fold(0., f64::max);
        let min = spheres.iter().map(|s| s.radius).fold(0., f64::min);
        println!("Finished calculating Photon radiuses {:?}", (avg, max, min));

        println!("Building BVH");
        let bvh = BVH::build(spheres.as_mut_slice());

        Self::PointMapForBeamEstimate(surface_map, bvh, spheres)
    }

    fn new_beam_map_for_beam_estimate(list: PhotonList) -> Self {
        let surface_map = KdTree::build_by(list.clone().0, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });
        let volume_map = KdTree::build_by(list.clone().1, |a, b, k| {
            a.position[k].partial_cmp(&b.position[k]).unwrap()
        });

        let mut beams = list
            .1
            .into_iter()
            .map(|p| {
                let max_distance_to_neighbor = volume_map
                    .nearests(&p, 3)
                    .into_iter()
                    .map(
                        |ItemAndDistance {
                             squared_distance, ..
                         }| squared_distance,
                    )
                    .fold(-1., f64::max)
                    .sqrt();
                // let max_distance_to_neighbor = 4.;
                let beam = PhotonBeam {
                    start_position: p.starting_position.clone(),
                    end_position:   p.position,
                    radius:         max_distance_to_neighbor,
                    ray:            Ray {
                        origin: p.starting_position,
                        dir:    glm::normalize(&(p.position - p.starting_position)),
                    },
                    power:          p.power,
                    node_index:     0,
                };
                return beam;
            })
            .collect::<Vec<_>>();

        let avg = beams.iter().map(|s| s.radius).sum::<f64>() / beams.len() as f64;
        let max = beams.iter().map(|s| s.radius).fold(0., f64::max);
        let min = beams.iter().map(|s| s.radius).fold(0., f64::min);
        println!(
            "Finished calculating photon beam radiuses {:?}",
            (avg, max, min)
        );

        println!("Building BVH");
        let bvh = BVH::build(beams.as_mut_slice());

        Self::BeamMapForBeamEstimate(surface_map, bvh, beams)
    }

    fn surface(&self) -> &KdTree<Photon> {
        match self {
            Self::PointMapForPointEstimate(surface_map, _) => surface_map,
            Self::PointMapForBeamEstimate(surface_map, _, _) => surface_map,
            Self::BeamMapForBeamEstimate(surface_map, _, _) => surface_map,
            // _ => todo!(),
        }
    }

    fn estimate_indirect(
        &self,
        renderer: &Renderer,
        ray: &Ray,
        medium: Option<&Medium>,
        rng: &mut StdRng,
    ) -> Color {
        let wo = -glm::normalize(&ray.dir);

        let surface_estimate = |h: &HitRecord, material: &Material, rng: &mut StdRng| {
            let world_pos = ray.at(h.time);
            let near_photons = self.surface().nearests(
                &[world_pos.x, world_pos.y, world_pos.z],
                renderer.gather_size,
            );

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
                let disp = world_pos - photon.position;
                let r = Ray {
                    origin: photon.position,
                    dir:    glm::normalize(&disp),
                };
                if let Some((hit, _)) = renderer.get_closest_hit(r) {
                    if disp.norm() > hit.time {
                        continue;
                    }
                }
                color += material
                    .bsdf(&h.normal, &wo, &photon.direction)
                    .component_mul(&photon.power)
                    * photon.direction.dot(&h.normal).clamp(0., 1.);
            }

            // normalize by (1/(pi * r^2))
            color = color * (1. / (glm::pi::<f64>() * max_dist_squared));

            // TODO: divide by probability of sampling behind surface

            // direct lighting via light sampling
            // color += renderer.sample_lights(&material, &world_pos, &h.normal, &wo, rng);

            // emitted lighting
            // color += material.emittance() * material.color();

            color
        };

        let volume_estimate = |medium: &Medium,
                               hit: Option<&HitRecord>,
                               object: Option<&Object>,
                               rng: &mut StdRng| {
            match self {
                Self::PointMapForPointEstimate(_, volume_map) => {
                    let (d, d_pdf, _d_cdf) = medium.sample_d(&ray, rng);
                    if hit.is_none() || d < hit.unwrap().time {
                        let collision = ray.at(d);
                        let abs = medium.absorption(&collision);
                        let _emm = medium.emission(&collision);
                        let medium_color = medium.color(&collision);
                        let scat = medium.scattering(&collision);
                        let extinction = abs + scat;

                        let mut color = Color::new(0., 0., 0.);

                        // direct lighting for media particle
                        // color += self.sample_lights_for_media(&medium,
                        // &collision,&wo, rng);

                        // estimate indirect lighting via photon map
                        let near_photons = volume_map.nearests(
                            &[collision.x, collision.y, collision.z],
                            renderer.gather_size_volume,
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
                            color += photon.power.component_mul(&medium_color)
                                * medium.phase(&wo, &photon.direction);
                        }
                        color /= (4. / 3.) * glm::pi::<f64>() * max_dist_squared.powf(1.5);
                        // color /= scat;
                        color /= extinction;

                        color *= medium.transmittence(&ray, d, 0.0, rng);
                        color /= d_pdf;

                        color
                    } else {
                        let material = object.unwrap().material;
                        // TODO: do we need transmittance here?
                        surface_estimate(&hit.unwrap(), &material, rng)
                            * medium.transmittence(&ray, hit.unwrap().time, 0.0, rng)
                        //    / d_cdf
                    }
                }
                Self::PointMapForBeamEstimate(_, bvh, spheres) => {
                    let intersected_photons = bvh.traverse(
                        &BvhRay::new(
                            Point3::new(
                                ray.origin.x as f32,
                                ray.origin.y as f32,
                                ray.origin.z as f32,
                            ),
                            Point3::new(ray.dir.x as f32, ray.dir.y as f32, ray.dir.z as f32),
                        ),
                        spheres.as_slice(),
                    );

                    let dummy_pos = glm::vec3(0., 0., 0.);
                    let abs = medium.absorption(&dummy_pos);
                    let _emm = medium.emission(&dummy_pos);
                    let medium_color = medium.color(&dummy_pos);
                    let scat = medium.scattering(&dummy_pos);
                    let extinction = abs + scat;

                    let mut volume_color = Color::new(0., 0., 0.);

                    // direct lighting for media particle
                    // color += self.sample_lights_for_media(&medium, &collision, &wo,
                    // rng);

                    // the blur kernel
                    let k2 = |square_param: f64| {
                        let tmp = 1. - square_param;
                        return (3. / glm::pi::<f64>()) * tmp * tmp;
                    };

                    // estimate indirect lighting via photon map
                    for photon in intersected_photons.iter() {
                        let scale_factor = 1.;

                        let origin_to_center = photon.position - ray.origin;

                        if let Some(hit) = hit {
                            if origin_to_center.norm() > hit.time {
                                continue;
                            }
                        }
                        let radius_squared = photon.radius * photon.radius;
                        let disk_distance = glm::dot(&origin_to_center, &ray.dir);
                        let distance_squared =
                            (ray.at(disk_distance) - photon.position).norm().powi(2);

                        if disk_distance > 0. && distance_squared < radius_squared {
                            let weight = k2(distance_squared / radius_squared) / radius_squared;

                            let wi = -photon.direction;

                            let transmittance = (-extinction * disk_distance).exp();
                            volume_color += transmittance
                                * photon.power.component_mul(&medium_color)
                                * medium.phase(&wi, &(-ray.dir))
                                * weight
                                * scale_factor;
                        }
                    }

                    return volume_color;
                }
                Self::BeamMapForBeamEstimate(_, bvh, beams) => {
                    let intersected_beams = bvh.traverse(
                        &BvhRay::new(
                            Point3::new(
                                ray.origin.x as f32,
                                ray.origin.y as f32,
                                ray.origin.z as f32,
                            ),
                            Point3::new(ray.dir.x as f32, ray.dir.y as f32, ray.dir.z as f32),
                        ),
                        beams.as_slice(),
                    );

                    let dummy_pos = glm::vec3(0., 0., 0.);
                    let medium_color = medium.color(&dummy_pos);
                    let scat = medium.scattering(&dummy_pos);

                    let mut volume_color = Color::new(0., 0., 0.);

                    let mut c = 0;

                    let k2 = |square_param: f64| {
                        let tmp = 1. - square_param;
                        return (3. / glm::pi::<f64>()) * tmp * tmp;
                    };

                    for beam in intersected_beams.iter() {
                        let l = beam.start_position - ray.origin;
                        let u = glm::normalize(&l.cross(&beam.ray.dir));
                        let n = beam.ray.dir.cross(&u);
                        let t = n.dot(&l) / n.dot(&ray.dir);
                        let query_collision = ray.at(t);

                        if let Some(hit) = hit {
                            if t >= hit.time {
                                continue;
                            }
                        }

                        let inv_sin_theta =
                            1.0 / 0.0_f64.max(1.0 - ray.dir.dot(&beam.ray.dir).powi(2)).sqrt();

                        let beam_t = beam.ray.dir.dot(&(query_collision - beam.start_position));

                        let beam_len = (beam.end_position - beam.start_position).norm();
                        if beam_t < 0.0 || beam_t > beam_len {
                            continue;
                        }

                        let beam_collision = beam.ray.at(beam_t);

                        let dist = (query_collision - beam_collision).norm();
                        if dist >= beam.radius {
                            continue;
                        }

                        let r = Ray {
                            origin: beam_collision,
                            dir:    -beam.ray.dir.clone(),
                        };

                        c += 1;

                        let color = scat
                            * beam.power.component_mul(&medium_color)
                            * medium.phase(&(-beam.ray.dir), &(-ray.dir))
                            * inv_sin_theta
                            * medium.transmittence(&ray, t, 0.0, rng)
                            * medium.transmittence(&r, beam_t, 0.0, rng)
                            * k2(dist / beam.radius)
                            / beam.radius;

                        volume_color += color;
                    }

                    if rng.gen::<f64>() < 0.00001 {
                        dbg!(c);
                        dbg!(intersected_beams.len());
                    }
                    volume_color
                }
            }
        };

        match renderer.get_closest_hit(*ray) {
            None => match medium {
                None => renderer.scene.environment.get_color(&ray.dir), // TODO attenuate
                Some(m) => volume_estimate(m, None, None, rng),
            },
            Some((h, object)) => {
                let material = object.material;
                match medium {
                    None => surface_estimate(&h, &material, rng),
                    Some(medium) => match self {
                        Self::PointMapForPointEstimate(_, _) => {
                            // NOTE: surface color happens inside
                            volume_estimate(&medium, Some(&h), Some(object), rng)
                        }
                        Self::PointMapForBeamEstimate(_, _, _)
                        | Self::BeamMapForBeamEstimate(_, _, _) => {
                            let volume_color = volume_estimate(medium, Some(&h), Some(object), rng);
                            let surface_color = surface_estimate(&h, &material, rng)
                                * medium.transmittence(&ray, h.time, 0., rng);
                            if rng.gen::<f64>() < 0.00001 {
                                println!("Surface: {}, volume: {}", surface_color, volume_color);
                            }
                            surface_color + volume_color
                        }
                    },
                }
            }
        }
    }
}

pub enum PhotonRenderKind {
    PhotonMap,
    PhotonPointBeam,
    PhotonBeamBeam,
}

impl<'a> Renderer<'a> {
    pub fn photon_point_query_beam_render(&self, photon_count: usize) -> RgbImage {
        self.photon_render(photon_count, PhotonRenderKind::PhotonPointBeam)
    }

    pub fn photon_beam_query_beam_render(&self, photon_count: usize) -> RgbImage {
        self.photon_render(photon_count, PhotonRenderKind::PhotonBeamBeam)
    }

    pub fn photon_map_render(&self, photon_count: usize) -> RgbImage {
        self.photon_render(photon_count, PhotonRenderKind::PhotonMap)
    }

    /// renders an image using photon mapping
    pub fn photon_render(&self, photon_count: usize, kind: PhotonRenderKind) -> RgbImage {
        // ensure that scene only has object lights (may not be necessary)
        for light in self.scene.lights.iter() {
            match light {
                Light::Object(_) => {}
                _ => {
                    // panic!("Only object lights are supported for photon mapping");
                }
            }
        }

        println!("Shooting photons");

        let pb = ProgressBar::new(photon_count as u64);
        pb.set_draw_rate(1);
        let photon_list = (0..photon_count)
            .collect::<Vec<_>>()
            .into_par_iter()
            .progress_with(pb)
            .map(|_|
          // shoot photon from a random light
          {
              let mut rng = StdRng::from_entropy();
              let power = self.watts / photon_count as f64;
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
        let photon_map = match kind {
            PhotonRenderKind::PhotonMap => PhotonMap::new_point_map_for_point_estimate(photon_list),
            PhotonRenderKind::PhotonPointBeam => {
                PhotonMap::new_point_map_for_beam_estimate(photon_list)
            }
            PhotonRenderKind::PhotonBeamBeam => {
                PhotonMap::new_beam_map_for_beam_estimate(photon_list)
            }
        };

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
        // FIXME: sample random light based on area instead of looping
        for light in self.scene.lights.iter() {
            // sample a random point on the light and a random direction in the hemisphere
            if let Light::Object(object) = light {
                // the `target` arg isn't used when sampling a triangle, so it can be a dummy value
                // Sample a location on the light
                let target = glm::vec3(0., 0., 0.);
                let (pos, n, _pdf) = object.shape.sample(&target, rng);

                // sample random hemisphere direction
                let phi = 2. * glm::pi::<f64>() * rng.gen::<f64>();
                let theta = (1. - rng.gen::<f64>()).acos();
                let _pdf_of_sample = 0.5 * glm::one_over_pi::<f64>();
                let random_hemisphere_dir = glm::vec3(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );

                // rotate direction towards normal
                let rotation: nalgebra::Rotation3<f64> =
                    nalgebra::Rotation3::rotation_between(&glm::vec3(0., 1., 0.), &n)
                        .unwrap_or_else(|| {
                            nalgebra::Rotation3::rotation_between(
                                &glm::vec3(0., 1., 0.00000001),
                                &n,
                            )
                            .unwrap()
                        });
                let direction = rotation * random_hemisphere_dir;

                // recursively gather photons
                let photons = self.trace_photon(
                    Ray {
                        origin: pos,
                        dir:    direction,
                    },
                    power * object.material.color(),
                    // / pdf / pdf_of_sample,
                    rng,
                    0,
                );

                return photons;
            }
        }
        panic!("Only found non-object lights while photon mapping")
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
        let wo = -glm::normalize(&ray.dir);

        let trace_on_surface =
            |power: glm::DVec3, h: &HitRecord, object: &Object, rng: &mut StdRng| {
                let world_pos = ray.at(h.time);
                let material = object.material;
                let wo = -glm::normalize(&ray.dir);

                // page 16 of siggraph course on photon mapping
                // let specular = 1. - material.roughness;
                // TODO: make these dependant on material
                let specular = glm::vec3(0.0, 0.0, 0.0);
                let diffuse = glm::vec3(0.7, 0.7, 0.7);
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
                let _p_s = specular_sum / (diffuse_sum + specular_sum) * p_r;
                // only do diffuse russian roulette for now (no specular)
                let russian_roulette: f64 = rng.gen();

                if russian_roulette < p_d {
                    // diffuse reflection
                    if let Some((wi, pdf)) = material.sample_f(&h.normal, &wo, rng) {
                        let f = material.bsdf(&h.normal, &wo, &wi);
                        let ray = Ray {
                            origin: world_pos,
                            dir:    wi,
                        };
                        // gather recursive photons with scaled down power
                        let mut next_photons = self.trace_photon(
                            ray,
                            power.component_mul(&f) * wi.dot(&h.normal) / pdf / p_d,
                            rng,
                            num_bounces + 1,
                        );
                        // add photon from current step
                        if pdf != 1. {
                            next_photons.add_surface(Photon {
                                position: world_pos,
                                direction: wo,
                                power,
                                starting_position: ray.origin,
                            });
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
            };

        let trace_in_volume =
            |power: glm::DVec3, medium: &Medium, d: f64, _d_pdf: f64, rng: &mut StdRng| {
                let collision = ray.at(d);
                let abs = medium.absorption(&collision);
                let _emm = medium.emission(&collision);
                let medium_color = medium.color(&collision);
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
                        attenuated_power.component_mul(&medium_color) * medium.phase(&wo, &wi)
                            / ph_p,
                        rng,
                        num_bounces + 1,
                    )
                } else {
                    PhotonList::default()
                };

                // add current photon
                next_photons.add_volume(Photon {
                    position:          collision,
                    direction:         wo,
                    power:             attenuated_power
                        + self.sample_lights_for_media(&medium, &collision, &wo, rng),
                    starting_position: ray.origin,
                });

                next_photons
            };

        match self.get_closest_hit(ray) {
            None => {
                if let Some(medium) = self.scene.media.get(0) {
                    // no hit, but in medium
                    let (d, d_pdf, _d_cdf) = medium.sample_d(&ray, rng);
                    trace_in_volume(power, medium, d, d_pdf, rng)
                } else {
                    // no hit, no medium
                    PhotonList::default()
                }
            }
            Some((h, object)) => {
                if let Some(medium) = self.scene.media.get(0) {
                    // hit, but in medium
                    let (d, d_pdf, _d_cdf) = medium.sample_d(&ray, rng);

                    if d < h.time {
                        // scattering event
                        trace_in_volume(power, medium, d, d_pdf, rng)
                    } else {
                        // no scattering event
                        trace_on_surface(
                            power * medium.transmittence(&ray, h.time, 0.0, rng), // / (d_cdf),
                            &h,
                            object,
                            rng,
                        )
                    }
                } else {
                    // hit, no medium
                    trace_on_surface(power, &h, object, rng)
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
        _num_bounces: u32,
        rng: &mut StdRng,
        photon_map: &PhotonMap,
    ) -> Color {
        photon_map.estimate_indirect(self, &ray, self.scene.media.get(0), rng)
    }
}