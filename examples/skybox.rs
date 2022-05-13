//! This is an example of a standard Cornell box, for testing volumetric global illumination
//! with participating media
//!
//! Reference: https://www.graphics.cornell.edu/online/box/data.html

use std::fs;
use std::time::Instant;

use rpt::*;

const SCALE: f64 = 1.0;

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
    let green = Material::diffuse(hex_color(0x00BC00));
    let light_mtl = Material::light(hex_color(0xFFFEFA), 50000.0);

    let floor = polygon(&[
        SCALE * glm::vec3(0.0, 0.0, -850.0),
        SCALE * glm::vec3(0.0, 0.0, 559.2),
        SCALE * glm::vec3(556.0, 0.0, 559.2),
        SCALE * glm::vec3(556.0, 0.0, -850.0),
    ]);

    let p1 = SCALE * glm::vec3(343.0 - 50.0, 548.9, 227.0);
    let p2 = SCALE * glm::vec3(343.0 - 50.0, 548.9, 332.0);
    let p3 = SCALE * glm::vec3(213.0 + 50.0, 548.9, 332.0);
    let p4 = SCALE * glm::vec3(213.0 + 50.0, 548.9, 227.0);

    let c1 = SCALE * glm::vec3(0.0, 548.9, -850.0);
    let c2 = SCALE * glm::vec3(556.0, 548.9, -850.0);
    let c3 = SCALE * glm::vec3(556.0, 548.9, 559.2);
    let c4 = SCALE * glm::vec3(0.0, 548.9, 559.2);

    let br = glm::vec3(p3[0], c4[1], c4[2]);
    let bl = glm::vec3(p2[0], c3[1], c3[2]);
    let fr = glm::vec3(p4[0], c1[1], c1[2]);
    let fl = glm::vec3(p1[0], c2[1], c2[2]);

    let ceiling_1 = polygon(&[c1, fr, br, c4]);
    let ceiling_2 = polygon(&[p3, p2, bl, br]);
    let ceiling_3 = polygon(&[fl, c2, c3, bl]);
    let ceiling_4 = polygon(&[fr, fl, p1, p4]);

    let shift = glm::vec3(0.0, 500.0, 0.0);
    let b1 = p1 + shift;
    let b2 = p2 + shift;
    let b3 = p3 + shift;
    let b4 = p4 + shift;

    let light_rect = polygon(&[b1, b2, b3, b4]).translate(&glm::vec3(-50.0, 0.0, 50.0));

    let back_wall = polygon(&[
        SCALE * glm::vec3(0.0, 0.0, 559.2),
        SCALE * glm::vec3(0.0, 548.9, 559.2),
        SCALE * glm::vec3(556.0, 548.9, 559.2),
        SCALE * glm::vec3(556.0, 0.0, 559.2),
    ]);
    let front_wall = polygon(&[
        SCALE * glm::vec3(0.0, 0.0, -850.0),
        SCALE * glm::vec3(556.0, 0.0, -850.0),
        SCALE * glm::vec3(556.0, 548.9, -850.0),
        SCALE * glm::vec3(0.0, 548.9, -850.0),
    ]);
    let right_wall = polygon(&[
        SCALE * glm::vec3(0.0, 0.0, -850.0),
        SCALE * glm::vec3(0.0, 548.9, -850.0),
        SCALE * glm::vec3(0.0, 548.9, 559.2),
        SCALE * glm::vec3(0.0, 0.0, 559.2),
    ]);
    let left_wall = polygon(&[
        SCALE * glm::vec3(556.0, 0.0, -850.0),
        SCALE * glm::vec3(556.0, 0.0, 559.2),
        SCALE * glm::vec3(556.0, 548.9, 559.2),
        SCALE * glm::vec3(556.0, 548.9, -850.0),
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
    scene.add(Object::new(ceiling_1).material(white));
    scene.add(Object::new(ceiling_2).material(white));
    scene.add(Object::new(ceiling_3).material(white));
    scene.add(Object::new(ceiling_4).material(white));
    scene.add(Object::new(back_wall).material(white));
    scene.add(Object::new(front_wall).material(white));
    scene.add(Object::new(left_wall).material(red));
    scene.add(Object::new(right_wall).material(green));

    scene.add(Object::new(large_box).material(white));
    scene.add(Object::new(small_box).material(white));

    scene.add((light_rect, light_mtl));
    scene.environment = Environment::Color(hex_color(0x87CEEB));

    scene.add(Medium::homogeneous_isotropic(0.0003, 0.0003)); // foggy

    let mut time = Instant::now();
    fs::create_dir_all("skybox/")?;
    Renderer::new(&scene, camera)
        .width(512)
        .height(512)
        .max_bounces(4)
        .num_samples(5000)
        .iterative_render(1000, |iteration, buffer| {
            let millis = time.elapsed().as_millis();
            println!(
                "Finished iteration {}, took {} ms, variance: {}",
                iteration,
                millis,
                buffer.variance()
            );
            buffer
                .image()
                .save(format!("skybox/output_{:03}.png", iteration - 1))
                .expect("Failed to save image");
            time = Instant::now();
        });

    Ok(())
}
