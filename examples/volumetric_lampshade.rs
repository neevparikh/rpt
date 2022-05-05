//! This is an example of a standard Cornell box, for testing volumetric global illumination
//! with participating media
//!
//! Reference: https://www.graphics.cornell.edu/online/box/data.html

use std::fs;
use std::time::Instant;

use rpt::*;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

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
    let yellow = Material::diffuse(hex_color(0x34EB7A));
    let green = Material::diffuse(hex_color(0x00BC00));
    let light_mtl = Material::light(hex_color(0xFFFEFA), 120.0); // 6500 K

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
    ]);

    let shift = glm::vec3(0.0, 70.0, 0.0);

    let front_shade = polygon(&[
        glm::vec3(343.0, 548.8, 227.0) - shift,
        glm::vec3(343.0, 548.8, 332.0) - shift,
        glm::vec3(343.0, 548.8, 332.0),
        glm::vec3(343.0, 548.8, 227.0),
    ]);
    let left_shade = polygon(&[
        glm::vec3(343.0, 548.8, 332.0) - shift,
        glm::vec3(213.0, 548.8, 332.0) - shift,
        glm::vec3(213.0, 548.8, 332.0),
        glm::vec3(343.0, 548.8, 332.0),
    ]);
    let back_shade = polygon(&[
        glm::vec3(213.0, 548.8, 332.0) - shift,
        glm::vec3(213.0, 548.8, 227.0) - shift,
        glm::vec3(213.0, 548.8, 227.0),
        glm::vec3(213.0, 548.8, 332.0),
    ]);
    let right_shade = polygon(&[
        glm::vec3(213.0, 548.8, 227.0) - shift,
        glm::vec3(343.0, 548.8, 227.0) - shift,
        glm::vec3(343.0, 548.8, 227.0),
        glm::vec3(213.0, 548.8, 227.0),
    ]);

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

    scene.add(Object::new(right_shade).material(yellow));
    scene.add(Object::new(left_shade).material(yellow));
    scene.add(Object::new(front_shade).material(yellow));
    scene.add(Object::new(back_shade).material(yellow));

    scene.add((light_rect, light_mtl));

    scene.add(Medium::homogeneous_isotropic(0.00002, 0.0002)); // foggy

    let mut time = Instant::now();
    fs::create_dir_all("volumetric_results/")?;
    Renderer::new(&scene, camera)
        .width(512)
        .height(512)
        .filter(Filter::Box(1))
        .max_bounces(4)
        .num_samples(2000)
        .iterative_render(100, |iteration, buffer| {
            let millis = time.elapsed().as_millis();
            println!(
                "Finished iteration {}, took {} ms, variance: {}",
                iteration,
                millis,
                buffer.variance()
            );
            buffer
                .image()
                .save(format!(
                    "volumetric_results/output_{:03}.png",
                    iteration - 1
                ))
                .expect("Failed to save image");
            time = Instant::now();
        });

    Ok(())
}
