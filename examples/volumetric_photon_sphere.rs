//! This is an example of a standard Cornell box, for testing volumetric global illumination
//! with participating media
//!
//! Reference: https://www.graphics.cornell.edu/online/box/data.html

use std::fs::{self, File};
use std::time::Instant;

use rpt::*;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    for i in 0..30 {
        f(i);
    }

    Ok(())
}

fn f(i: i32) -> color_eyre::Result<()> {
    let mut scene = Scene::new();

    let camera = Camera {
        eye: glm::vec3(278.0, 273.0, -800.0),
        direction: glm::vec3(0.0, 0.0, 1.0),
        up: glm::vec3(0.0, 1.0, 0.0),
        fov: 0.686,
        ..Default::default()
    };

    let white = Material::diffuse(hex_color(0xAAAAAA));
    let red = Material::diffuse(hex_color(0xBC0000));
    let green = Material::diffuse(hex_color(0x00BC00));
    let light_mtl = Material::light(hex_color(0xFFFEFA), 100.0); // 6500 K

    let floor = polygon(&[
        glm::vec3(0.0, 0.0, 0.0),
        glm::vec3(0.0, 0.0, 559.2),
        glm::vec3(556.0, 0.0, 559.2),
        glm::vec3(556.0, 0.0, 0.0),
    ]);
    let ceiling = polygon(&[
        glm::vec3(0.0, 548.9, 0.0),
        glm::vec3(556.0, 548.9, 0.0),
        glm::vec3(556.0, 548.9, 559.2),
        glm::vec3(0.0, 548.9, 559.2),
    ]);
    let light_rect = polygon(&[
        glm::vec3(343.0, 548.8, 227.0),
        glm::vec3(343.0, 548.8, 332.0),
        glm::vec3(213.0, 548.8, 332.0),
        glm::vec3(213.0, 548.8, 227.0),
    ])
    .translate(&glm::vec3(0.0, 0.0, 0.0))
    .rotate(3.14159265 / 2., &glm::vec3(1.0, 0.0, 0.0))
    .translate(&glm::vec3(0., -10., 0.));

    let percent = i as f64 / 30.;
    let rads = percent * 2. * 3.14159265;

    let x = 100. * rads.sin() + 300.;
    let y = 100. * rads.cos() + 400.;

    let light_rect = load_obj(File::open("examples/cube.obj")?)?.translate(&glm::vec3(x, y, 227.0));

    let back_wall = polygon(&[
        glm::vec3(0.0, 0.0, 559.2),
        glm::vec3(0.0, 548.9, 559.2),
        glm::vec3(556.0, 548.9, 559.2),
        glm::vec3(556.0, 0.0, 559.2),
    ]);
    let right_wall = polygon(&[
        glm::vec3(0.0, 0.0, 0.0),
        glm::vec3(0.0, 548.9, 0.0),
        glm::vec3(0.0, 548.9, 559.2),
        glm::vec3(0.0, 0.0, 559.2),
    ]);
    let left_wall = polygon(&[
        glm::vec3(556.0, 0.0, 0.0),
        glm::vec3(556.0, 0.0, 559.2),
        glm::vec3(556.0, 548.9, 559.2),
        glm::vec3(556.0, 548.9, 0.0),
    ]);

    let large_box = cube()
        .scale(&glm::vec3(165.0, 330.0, 165.0))
        .rotate_y(glm::two_pi::<f64>() * (-253.0 / 360.0))
        .translate(&glm::vec3(368.0, 165.0, 351.0));
    let small_box = cube()
        .scale(&glm::vec3(165.0, 165.0, 165.0))
        .rotate_y(glm::two_pi::<f64>() * (-197.0 / 360.0))
        .translate(&glm::vec3(185.0, 82.5, 169.0));

    scene.add(Object::new(floor).material(white));
    scene.add(Object::new(ceiling).material(white));
    scene.add(Object::new(back_wall).material(white));
    scene.add(Object::new(left_wall).material(red));
    scene.add(Object::new(right_wall).material(green));
    scene.add(Object::new(large_box).material(white));
    scene.add(Object::new(small_box).material(white));
    scene.add((light_rect, light_mtl)); // add light and object at the same time

    let absorb = 0.00008;
    let scat = 0.00008;
    let size = 256;
    let bounce = 10;
    let sample = 500;
    let photons = 1_000_000;
    let gather_size = 200;
    let gather_size_volume = 100;

    scene.add(Medium::homogeneous_isotropic(absorb, scat)); // foggy

    let image = Renderer::new(&scene, camera)
        .width(size)
        .height(size)
        // .filter(Filter::Box(1))
        .max_bounces(bounce)
        .num_samples(sample)
        .gather_size(gather_size)
        .gather_size_volume(gather_size_volume)
        .photon_map_render(photons);

    image
        .save(format!(
            "vpm/sphere/{}_{}_{}_{}_{}_{}_{}_{}_{}_{:0>3}.png",
            size, bounce, sample, photons, 100, gather_size, gather_size_volume, absorb, scat, i
        ))
        .expect("Failed to save image");

    Ok(())
}
